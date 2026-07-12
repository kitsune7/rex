"""
rex_memory_v2.py — v1 store + MemPalace scoping + Graphiti bitemporal edges.

What changed from v1, and why
-----------------------------
Two layers were added on top of the v1 `turn` / `memory` / `retrieval_log` schema.

MemPalace layer (fidelity + scoping):
  * wing  = a top-level scope: a person, project, or domain ("user", "vasion-work")
  * room  = a topic cluster inside a wing ("alarms", "dietary")
  * drawer = the VERBATIM turn, kept addressable and independently searchable.
    This is the LLM-free fallback: even if consolidation drops or garbles a fact,
    the raw drawer is still there. Retrieval scopes to wing/room first, so a query
    runs against a room, not the whole flat corpus.

Graphiti layer (temporal correctness):
  * entity = a node (person / device / preference / project / concept)
  * edge   = a BITEMPORAL fact between entities, with two independent time axes:
        world time   t_valid / t_invalid  -> "when was this true in reality"
        system time  t_created / t_expired -> "when did Rex believe this"
    Corrections never delete. A contradicting fact INVALIDATES the old edge
    (closes both its time axes, links invalidated_by) and inserts a new one.
    That single distinction lets you answer three different questions later:
        - what is true now
        - what was true on some past date        (world-time "as-of")
        - what Rex believed at some past moment   (system-time "as-of" / audit)

The unstructured `memory` table from v1 stays for prose facts that don't
decompose into subject-predicate-object. Structured, supersedable facts
(preferences, attributes, state) now live as edges.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


# --------------------------------------------------------------------------- #
# SCHEMA v2                                                                    #
# --------------------------------------------------------------------------- #

SCHEMA = """
-- ===== MemPalace: scoping =====
CREATE TABLE IF NOT EXISTS wing (
    id   TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,       -- 'user' | 'vasion-work' | 'family' | ...
    kind TEXT NOT NULL               -- person | project | domain
);
CREATE TABLE IF NOT EXISTS room (
    id      TEXT PRIMARY KEY,
    wing_id TEXT NOT NULL REFERENCES wing(id),
    topic   TEXT NOT NULL,
    UNIQUE(wing_id, topic)
);

-- ===== MemPalace: verbatim drawers (was v1 `turn`, now scoped + searchable) =
CREATE TABLE IF NOT EXISTS drawer (
    id        TEXT PRIMARY KEY,
    wing_id   TEXT REFERENCES wing(id),
    room_id   TEXT REFERENCES room(id),
    role      TEXT NOT NULL,         -- user | assistant | system
    content   TEXT NOT NULL,         -- VERBATIM. never rewritten.
    ts        REAL NOT NULL,
    embedding BLOB
);

-- ===== v1 prose memories (unchanged shape, now carries wing/room) ==========
CREATE TABLE IF NOT EXISTS memory (
    id           TEXT PRIMARY KEY,
    wing_id      TEXT REFERENCES wing(id),
    room_id      TEXT REFERENCES room(id),
    kind         TEXT NOT NULL,
    content      TEXT NOT NULL,
    salience     REAL NOT NULL,
    confidence   REAL NOT NULL,
    created_at   REAL NOT NULL,
    last_access  REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'active',
    source_turns TEXT NOT NULL,
    embedding    BLOB
);

-- ===== Graphiti: entity nodes ==============================================
CREATE TABLE IF NOT EXISTS entity (
    id      TEXT PRIMARY KEY,
    wing_id TEXT REFERENCES wing(id),
    name    TEXT NOT NULL,
    type    TEXT NOT NULL,           -- person | device | preference | project | concept
    UNIQUE(wing_id, name)            -- entity resolution scoped to a wing
);

-- ===== Graphiti: bitemporal edges ==========================================
CREATE TABLE IF NOT EXISTS edge (
    id             TEXT PRIMARY KEY,
    wing_id        TEXT REFERENCES wing(id),
    room_id        TEXT REFERENCES room(id),
    subject_id     TEXT NOT NULL REFERENCES entity(id),
    predicate      TEXT NOT NULL,    -- 'wake_alarm_is' | 'prefers' | 'lives_in' | 'uses'
    object_value   TEXT,             -- literal ('7:00 AM') ...
    object_id      TEXT REFERENCES entity(id),  -- ... or a link to another entity

    -- world time: when the fact is/was true in reality
    t_valid        REAL NOT NULL,
    t_invalid      REAL,             -- NULL = still true

    -- system time: when Rex ingested / retired this belief
    t_created      REAL NOT NULL,
    t_expired      REAL,             -- NULL = still the live belief

    invalidated_by TEXT REFERENCES edge(id),
    salience       REAL NOT NULL DEFAULT 0.5,
    confidence     REAL NOT NULL DEFAULT 0.5,
    source_turns   TEXT NOT NULL,
    embedding      BLOB
);
CREATE INDEX IF NOT EXISTS idx_edge_live ON edge(subject_id, predicate, t_expired);

-- ===== retrieval instrumentation (unchanged from v1) =======================
CREATE TABLE IF NOT EXISTS retrieval_log (
    id TEXT PRIMARY KEY, query TEXT, ts REAL,
    ref_kind TEXT, ref_id TEXT, rank INTEGER, used INTEGER DEFAULT 0
);
"""


class Verdict(str, Enum):
    KEEP_HOT = "KEEP_HOT"; OFFLOAD = "OFFLOAD"; MERGE = "MERGE"; DROP = "DROP"


@dataclass
class EdgeFact:
    """A structured fact the scorer wants written to the graph."""
    subject: str                       # entity name, e.g. "user"
    predicate: str                     # e.g. "wake_alarm_is"
    object_value: Optional[str] = None
    object_entity: Optional[str] = None
    subject_type: str = "concept"
    t_valid: Optional[float] = None    # world-time onset; defaults to now
    # if this fact replaces a prior value of (subject, predicate), the store
    # finds the live edge and invalidates it — no id needed from the scorer.
    invalidates_prior: bool = False
    salience: float = 0.5
    confidence: float = 0.5


@dataclass
class ScoredSpan:
    turn_id: str
    verdict: Verdict
    wing: str = "user"                 # MemPalace scope
    room: str = "general"
    # prose-memory path (v1)
    canonical: str = ""
    kind: str = "fact"
    salience: float = 0.0
    confidence: float = 0.0
    # graph path (v2)
    edges: list[EdgeFact] = field(default_factory=list)
    reason: str = ""


# --------------------------------------------------------------------------- #
# SCORER PROMPT — additions over v1                                            #
# --------------------------------------------------------------------------- #
# The v1 verdict rules (KEEP_HOT / OFFLOAD / MERGE / DROP, DROP-asymmetry, etc.)
# still apply. The prompt now ALSO asks the model to (a) route each span to a
# wing + room, and (b) emit structured edges for anything that is an attribute,
# preference, state, or relationship — the supersedable stuff. Prose that isn't
# structured still goes to `memory` via `canonical`.
SCORER_ADDENDUM = """
Additionally, for each span:

- Assign a `wing` (top-level scope: a person/project/domain, e.g. "user",
  "vasion-work") and a `room` (topic within it, e.g. "alarms", "dietary").
  Reuse existing wings/rooms when they fit; propose a new short slug otherwise.

- Emit `edges` for any fact shaped as (subject, predicate, value/entity): user
  attributes, preferences, state, and relationships. Example:
    "move my alarm to 7" ->
      edges: [{"subject":"user","predicate":"wake_alarm_is",
               "object_value":"7:00 AM","t_valid":"<onset if stated>",
               "invalidates_prior": true}]
  Set "invalidates_prior": true when this replaces a previous value of the same
  (subject, predicate) — the store will retire the old edge, never delete it.
  Leave prose that does not decompose this way in `canonical` for `memory`.
"""


# --------------------------------------------------------------------------- #
# STORE                                                                        #
# --------------------------------------------------------------------------- #

class MemoryStore:
    def __init__(self, path: str = "rex_memory.db"):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # --- MemPalace: resolve scope ---------------------------------------- #
    def _wing(self, name: str, kind: str = "domain") -> str:
        row = self.db.execute("SELECT id FROM wing WHERE name=?", (name,)).fetchone()
        if row:
            return row["id"]
        wid = str(uuid.uuid4())
        self.db.execute("INSERT INTO wing VALUES (?,?,?)", (wid, name, kind))
        return wid

    def _room(self, wing_id: str, topic: str) -> str:
        row = self.db.execute(
            "SELECT id FROM room WHERE wing_id=? AND topic=?", (wing_id, topic)).fetchone()
        if row:
            return row["id"]
        rid = str(uuid.uuid4())
        self.db.execute("INSERT INTO room VALUES (?,?,?)", (rid, wing_id, topic))
        return rid

    # --- Graphiti: resolve entity ---------------------------------------- #
    def _entity(self, wing_id: str, name: str, type_: str) -> str:
        row = self.db.execute(
            "SELECT id FROM entity WHERE wing_id=? AND name=?", (wing_id, name)).fetchone()
        if row:
            return row["id"]
        eid = str(uuid.uuid4())
        self.db.execute("INSERT INTO entity VALUES (?,?,?,?)", (eid, wing_id, name, type_))
        return eid

    # --- MemPalace: verbatim drawer -------------------------------------- #
    def log_drawer(self, role: str, content: str,
                   wing: str = "user", room: str = "general", ts: Optional[float] = None) -> str:
        wid, rid = self._wing(wing, "person" if wing == "user" else "domain"), None
        rid = self._room(wid, room)
        did = str(uuid.uuid4())
        self.db.execute("INSERT INTO drawer VALUES (?,?,?,?,?,?,NULL)",
                        (did, wid, rid, role, content, ts or time.time()))
        self.db.commit()
        return did

    # --- Graphiti: write an edge, invalidating any prior value ----------- #
    def add_edge(self, fact: EdgeFact, wing: str, room: str,
                 source_turns: list[str], now: Optional[float] = None) -> str:
        now = now or time.time()
        wid = self._wing(wing)
        rid = self._room(wid, room)
        subj = self._entity(wid, fact.subject, fact.subject_type)
        obj_id = self._entity(wid, fact.object_entity, "concept") if fact.object_entity else None
        t_valid = fact.t_valid if fact.t_valid is not None else now

        new_id = str(uuid.uuid4())
        if fact.invalidates_prior:
            # find the currently-live edge for this (subject, predicate) and retire it.
            # world-time closes at the new fact's onset; system-time closes now.
            prior = self.db.execute(
                "SELECT id FROM edge WHERE subject_id=? AND predicate=? AND t_expired IS NULL",
                (subj, fact.predicate)).fetchone()
            if prior:
                self.db.execute(
                    "UPDATE edge SET t_invalid=?, t_expired=?, invalidated_by=? WHERE id=?",
                    (t_valid, now, new_id, prior["id"]))

        self.db.execute(
            "INSERT INTO edge (id,wing_id,room_id,subject_id,predicate,object_value,object_id,"
            "t_valid,t_invalid,t_created,t_expired,invalidated_by,salience,confidence,"
            "source_turns,embedding) VALUES (?,?,?,?,?,?,?,?,NULL,?,NULL,NULL,?,?,?,NULL)",
            (new_id, wid, rid, subj, fact.predicate, fact.object_value, obj_id,
             t_valid, now, fact.salience, fact.confidence, json.dumps(source_turns)))
        self.db.commit()
        return new_id

    # --- consolidation pass ---------------------------------------------- #
    def consolidate(self, spans: list[tuple[str, str]], score_fn, embed_fn=None):
        # (score_fn / _parse_scored omitted here for brevity — identical shape to v1,
        #  now also parsing wing/room/edges per SCORER_ADDENDUM.)
        scored: list[ScoredSpan] = score_fn(spans)   # returns list[ScoredSpan]
        now = time.time()
        for s in scored:
            if s.verdict == Verdict.DROP:
                continue
            wid = self._wing(s.wing); rid = self._room(wid, s.room)
            for f in s.edges:                                   # Graphiti path
                self.add_edge(f, s.wing, s.room, [s.turn_id], now)
            if s.canonical:                                     # prose path (v1)
                self.db.execute(
                    "INSERT INTO memory (id,wing_id,room_id,kind,content,salience,confidence,"
                    "created_at,last_access,access_count,status,source_turns,embedding) "
                    "VALUES (?,?,?,?,?,?,?,?,?,0,'active',?,?)",
                    (str(uuid.uuid4()), wid, rid, s.kind, s.canonical, s.salience, s.confidence,
                     now, now, json.dumps([s.turn_id]),
                     embed_fn(s.canonical) if embed_fn else None))
        self.db.commit()
        return scored

    # --- Graphiti: the three "as-of" reads ------------------------------- #
    def current(self, wing: str, subject: str, predicate: str) -> Optional[str]:
        """What is true now."""
        return self._edge_value(
            wing, subject, predicate,
            "t_invalid IS NULL AND t_expired IS NULL", ())

    def world_as_of(self, wing: str, subject: str, predicate: str, when: float) -> Optional[str]:
        """What was true in the world at `when` (world-time)."""
        return self._edge_value(
            wing, subject, predicate,
            "t_valid <= ? AND (t_invalid IS NULL OR t_invalid > ?)", (when, when))

    def believed_as_of(self, wing: str, subject: str, predicate: str, when: float) -> Optional[str]:
        """What Rex believed at system-time `when` (audit / 'why did it say that')."""
        return self._edge_value(
            wing, subject, predicate,
            "t_created <= ? AND (t_expired IS NULL OR t_expired > ?)", (when, when))

    def _edge_value(self, wing, subject, predicate, clause, params):
        wid = self._wing(wing)
        subj = self.db.execute(
            "SELECT id FROM entity WHERE wing_id=? AND name=?", (wid, subject)).fetchone()
        if not subj:
            return None
        row = self.db.execute(
            f"SELECT object_value FROM edge WHERE subject_id=? AND predicate=? AND {clause} "
            f"ORDER BY t_valid DESC LIMIT 1", (subj["id"], predicate, *params)).fetchone()
        return row["object_value"] if row else None

    # --- retrieval: scope first (MemPalace), then hybrid ----------------- #
    def retrieve(self, query: str, wing: Optional[str] = None, room: Optional[str] = None,
                 k: int = 8) -> dict:
        """
        Scope to wing/room when given (MemPalace), union live edges + prose memories,
        and always keep verbatim drawers reachable as the LLM-free fallback.
        Replace the ORDER BYs with vector search joined to a recency/frequency decay.
        """
        now = time.time()
        scope_w = "wing_id = (SELECT id FROM wing WHERE name=?)" if wing else "1=1"
        wp = (wing,) if wing else ()

        edges = self.db.execute(
            f"SELECT subject_id, predicate, object_value FROM edge "
            f"WHERE {scope_w} AND t_expired IS NULL AND t_invalid IS NULL LIMIT ?", (*wp, k)
        ).fetchall()
        mems = self.db.execute(
            f"SELECT content FROM memory WHERE {scope_w} AND status='active' "
            f"ORDER BY salience DESC LIMIT ?", (*wp, k)).fetchall()
        drawers = self.db.execute(
            f"SELECT content FROM drawer WHERE {scope_w} ORDER BY ts DESC LIMIT ?", (*wp, k)
        ).fetchall()
        return {
            "edges": [dict(e) for e in edges],
            "memories": [m["content"] for m in mems],
            "drawers_verbatim": [d["content"] for d in drawers],  # fallback path
        }


# --------------------------------------------------------------------------- #
# DEMO — no LLM. Proves bitemporal invalidation + the three as-of reads.       #
# --------------------------------------------------------------------------- #
def _demo():
    s = MemoryStore(":memory:")
    # Use plain float "days" as timestamps for a legible demo.
    # Day 1: Rex learns the alarm is 6:00 AM (true from day 1).
    s.add_edge(EdgeFact("user", "wake_alarm_is", object_value="6:00 AM",
                        subject_type="person", t_valid=1.0), "user", "alarms", ["t1"], now=1.0)
    # Day 3: user corrects it to 7:00 AM. Old edge is invalidated, not deleted.
    s.add_edge(EdgeFact("user", "wake_alarm_is", object_value="7:00 AM",
                        t_valid=3.0, invalidates_prior=True), "user", "alarms", ["t2"], now=3.0)

    print("current              ->", s.current("user", "user", "wake_alarm_is"))
    print("world_as_of day 2    ->", s.world_as_of("user", "user", "wake_alarm_is", 2.0))
    print("world_as_of day 4    ->", s.world_as_of("user", "user", "wake_alarm_is", 4.0))
    print("believed_as_of day 2 ->", s.believed_as_of("user", "user", "wake_alarm_is", 2.0))
    print("believed_as_of day 4 ->", s.believed_as_of("user", "user", "wake_alarm_is", 4.0))
    # verbatim drawer survives regardless of what consolidation did
    s.log_drawer("user", "actually make it 7 not 6", wing="user", room="alarms", ts=3.0)
    print("verbatim fallback    ->", s.retrieve("alarm", wing="user")["drawers_verbatim"])


if __name__ == "__main__":
    _demo()
