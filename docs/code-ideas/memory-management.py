"""
rex_memory.py — a tiered, forgetting-aware memory module for Rex.

Design stance
-------------
"Forget" almost never means "delete." It means "evict from the hot context but
keep addressable." The only irreversible verdict is DROP, and it is reserved for
true noise. Everything else is soft: OFFLOAD (move to store + leave a breadcrumb),
MERGE (fold into / supersede an existing memory), KEEP_HOT (stays in context).

The two schema features that make the hard benchmark categories work:
  * supersedes / valid_from / valid_to  -> knowledge-update + temporal questions
  * status = active|archived|dropped (soft delete) + retrieval_log
    -> lets you measure "forgetting regret": was DROPed content later needed?

LLM calls are abstracted behind score_spans() and judge_answer(). Wire them to
your sanctioned Anthropic API access or a local model; the module is agnostic.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Optional


# --------------------------------------------------------------------------- #
# 1. STORE SCHEMA                                                             #
# --------------------------------------------------------------------------- #

SCHEMA = """
-- Append-only raw log. Never mutated. Ground truth for audit + reconstruction.
CREATE TABLE IF NOT EXISTS turn (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,          -- 'user' | 'assistant' | 'system'
    content     TEXT NOT NULL,
    ts          REAL NOT NULL           -- unix seconds
);

-- Consolidated memories. This is what gets tiered / decayed / retrieved.
CREATE TABLE IF NOT EXISTS memory (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,        -- fact|preference|event|task|entity|procedure
    content       TEXT NOT NULL,        -- canonical, self-contained NL statement
    salience      REAL NOT NULL,        -- model-assigned base importance [0,1]
    confidence    REAL NOT NULL,        -- extraction confidence [0,1]

    created_at    REAL NOT NULL,
    last_access   REAL NOT NULL,
    access_count  INTEGER NOT NULL DEFAULT 0,

    -- temporal validity: lets "what's my CURRENT X" differ from history
    valid_from    REAL,
    valid_to      REAL,                 -- NULL = still valid

    -- supersession chain for knowledge updates
    supersedes    TEXT REFERENCES memory(id),
    superseded_by TEXT REFERENCES memory(id),

    status        TEXT NOT NULL DEFAULT 'active',  -- active|archived|dropped
    source_turns  TEXT NOT NULL,        -- json list of turn.id (breadcrumb/provenance)

    -- embeddings: store here as a blob, or (better) in sqlite-vec / a sidecar index.
    embedding     BLOB
);

CREATE INDEX IF NOT EXISTS idx_memory_status ON memory(status);
CREATE INDEX IF NOT EXISTS idx_memory_kind   ON memory(kind);

-- Every retrieval, logged. This is the instrument that tells you whether
-- forgetting decisions were good in production (not just on a benchmark).
CREATE TABLE IF NOT EXISTS retrieval_log (
    id          TEXT PRIMARY KEY,
    query       TEXT NOT NULL,
    ts          REAL NOT NULL,
    memory_id   TEXT REFERENCES memory(id),
    rank        INTEGER,                -- position in retrieved set
    used        INTEGER DEFAULT 0       -- did the answer actually depend on it?
);
"""


class Verdict(str, Enum):
    KEEP_HOT = "KEEP_HOT"   # stays in working context verbatim
    OFFLOAD  = "OFFLOAD"    # -> store, replace in-context with a breadcrumb
    MERGE    = "MERGE"      # fold into / supersede an existing memory
    DROP     = "DROP"       # true noise; the only irreversible verdict


class Kind(str, Enum):
    FACT = "fact"; PREFERENCE = "preference"; EVENT = "event"
    TASK = "task"; ENTITY = "entity"; PROCEDURE = "procedure"


@dataclass
class ScoredSpan:
    turn_id: str
    verdict: Verdict
    kind: Optional[Kind] = None
    salience: float = 0.0
    confidence: float = 0.0
    canonical: str = ""              # rewritten self-contained statement
    supersedes_id: Optional[str] = None
    valid_from: Optional[float] = None
    reason: str = ""


# --------------------------------------------------------------------------- #
# 2. SCORER PROMPT (tuned for a conversational voice assistant)               #
# --------------------------------------------------------------------------- #

SCORER_SYSTEM = """You are the memory-consolidation module for a personal voice \
assistant. You run periodically over recent conversation turns and decide what \
should persist. You are NOT summarizing — you are predicting FUTURE RETRIEVAL \
UTILITY: how likely is this content to be needed again in a later conversation?

For each numbered span, assign exactly one verdict:

  KEEP_HOT  Immediately and repeatedly relevant to the CURRENT thread. Rare.
  OFFLOAD   Worth remembering long-term but not needed in active context now.
            The default for anything durable: user facts, standing preferences,
            open tasks/reminders, decisions, named entities and their attributes.
  MERGE     Updates or contradicts something the user told you before. Set
            supersedes_id to the memory being replaced. This is how corrections
            and preference changes are handled — DO NOT create a duplicate.
  DROP      True transient noise: acknowledgments, filler, chit-chat, resolved
            micro-requests, TTS/ASR artifacts. When unsure, prefer OFFLOAD, never
            DROP. Wrongly dropping is expensive; wrongly keeping is only cheap.

Priorities for a voice assistant, high to low:
  1. Explicit user preferences and standing instructions ("always...", "I hate...")
  2. Durable facts about the user, their people, environment, and recurring tasks
  3. Corrections / updates to earlier statements  (-> MERGE + supersedes_id)
  4. Commitments, reminders, deadlines            (-> OFFLOAD, kind=task)
  5. One-off factual answers the user may re-ask

Rules:
- Rewrite each kept span as a CANONICAL, self-contained statement with no
  pronouns or conversational framing. "yeah make it 7am not 6" (correcting an
  earlier alarm) -> canonical: "User's wake alarm is set to 7:00 AM." + MERGE.
- Capture temporal cues. If a statement has a clear onset, set valid_from
  (ISO date or relative like "2026-07-11"). If it ends a prior fact's validity,
  MERGE and the old memory's valid_to closes automatically.
- salience in [0,1] = predicted long-run reuse, not emotional weight.
- Output ONLY a JSON array, no prose, no markdown fences. One object per span:
  {"turn_id","verdict","kind","salience","confidence","canonical",
   "supersedes_id","valid_from","reason"}
  Use null for absent fields. "reason" is <=12 words for auditing.
"""

SCORER_USER_TEMPLATE = """Existing memories that these spans might update \
(id :: content):
{existing_memories}

Recent spans to score:
{spans}
"""


# --------------------------------------------------------------------------- #
# 3. STORE + CONSOLIDATION LOOP                                               #
# --------------------------------------------------------------------------- #

# You supply these. See the docstring — keep them model-agnostic.
ScoreFn  = Callable[[str, str], str]   # (system, user) -> raw JSON text
EmbedFn  = Callable[[str], bytes]      # text -> embedding blob (optional)


class MemoryStore:
    def __init__(self, path: str = "rex_memory.db"):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # --- raw log ---------------------------------------------------------- #
    def log_turn(self, session_id: str, role: str, content: str) -> str:
        tid = str(uuid.uuid4())
        self.db.execute(
            "INSERT INTO turn VALUES (?,?,?,?,?)",
            (tid, session_id, role, content, time.time()),
        )
        self.db.commit()
        return tid

    # --- consolidation pass ---------------------------------------------- #
    def consolidate(
        self,
        spans: list[tuple[str, str]],      # (turn_id, text)
        score_fn: ScoreFn,
        embed_fn: Optional[EmbedFn] = None,
        candidate_supersede_k: int = 20,
    ) -> list[ScoredSpan]:
        existing = self._recent_active(candidate_supersede_k)
        existing_str = "\n".join(f"{m['id']} :: {m['content']}" for m in existing) or "(none)"
        spans_str = "\n".join(f"[{tid}] {txt}" for tid, txt in spans)

        raw = score_fn(SCORER_SYSTEM,
                       SCORER_USER_TEMPLATE.format(existing_memories=existing_str, spans=spans_str))
        scored = _parse_scored(raw)

        now = time.time()
        for s in scored:
            if s.verdict == Verdict.DROP:
                continue  # noise: never written to `memory`. Still recoverable from `turn`.

            if s.verdict == Verdict.MERGE and s.supersedes_id:
                self.db.execute(
                    "UPDATE memory SET superseded_by=?, valid_to=?, status='archived' "
                    "WHERE id=? AND status='active'",
                    (None, now, s.supersedes_id),
                )
            emb = embed_fn(s.canonical) if embed_fn else None
            mid = str(uuid.uuid4())
            self.db.execute(
                "INSERT INTO memory (id,kind,content,salience,confidence,created_at,"
                "last_access,access_count,valid_from,valid_to,supersedes,superseded_by,"
                "status,source_turns,embedding) VALUES (?,?,?,?,?,?,?,0,?,NULL,?,NULL,'active',?,?)",
                (mid, (s.kind or Kind.FACT).value, s.canonical, s.salience, s.confidence,
                 now, now, s.valid_from, s.supersedes_id, json.dumps([s.turn_id]), emb),
            )
            if s.verdict == Verdict.MERGE and s.supersedes_id:
                self.db.execute("UPDATE memory SET superseded_by=? WHERE id=?", (mid, s.supersedes_id))
        self.db.commit()
        return scored

    # --- retrieval (hybrid: recency + frequency + salience, then rerank) -- #
    def retrieve(self, query: str, embed_fn: Optional[EmbedFn] = None, k: int = 8) -> list[sqlite3.Row]:
        # Placeholder ranking: salience * recency-decay * log(freq). Replace the
        # ORDER BY with a vector search (sqlite-vec / faiss) joined to this decay.
        now = time.time()
        rows = self.db.execute(
            "SELECT *, salience * (1.0/(1.0+(?-last_access)/86400.0)) "
            "* (1.0+access_count) AS score "
            "FROM memory WHERE status='active' ORDER BY score DESC LIMIT ?",
            (now, k),
        ).fetchall()
        qid = str(uuid.uuid4())
        for rank, r in enumerate(rows):
            self.db.execute("INSERT INTO retrieval_log VALUES (?,?,?,?,?,0)",
                            (str(uuid.uuid4()), query, now, r["id"], rank))
            self.db.execute("UPDATE memory SET access_count=access_count+1, last_access=? WHERE id=?",
                            (now, r["id"]))
        self.db.commit()
        return rows

    def _recent_active(self, k: int) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT id, content FROM memory WHERE status='active' "
            "ORDER BY last_access DESC LIMIT ?", (k,)).fetchall()


def _parse_scored(raw: str) -> list[ScoredSpan]:
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    out = []
    for o in json.loads(raw):
        out.append(ScoredSpan(
            turn_id=o["turn_id"], verdict=Verdict(o["verdict"]),
            kind=Kind(o["kind"]) if o.get("kind") else None,
            salience=float(o.get("salience") or 0), confidence=float(o.get("confidence") or 0),
            canonical=o.get("canonical") or "", supersedes_id=o.get("supersedes_id"),
            valid_from=o.get("valid_from"), reason=o.get("reason") or "",
        ))
    return out


# --------------------------------------------------------------------------- #
# 4. EVAL HARNESS  (LongMemEval / LoCoMo compatible)                          #
# --------------------------------------------------------------------------- #
#
# Both benchmarks give you: a haystack of dialogue sessions + a question whose
# evidence lives in specific session(s). Protocol:
#   1. stream every session's turns through consolidate() to BUILD the store
#   2. answer the question using retrieve() + your answering LLM
#   3. judge answer vs oracle with an LLM judge (LongMemEval ships evaluate_qa.py)
#
# What you report is the whole point. Don't report one accuracy number; report:
#   * per-question-type accuracy (the forgetting bugs hide in specific types)
#   * abstention accuracy      -> catches OVER-dropping (you answer when you shouldn't)
#   * knowledge-update accuracy -> catches broken MERGE/supersession
#   * temporal accuracy        -> catches dropped timestamps / valid_from
#   * avg context tokens used   -> the cost axis of the Pareto curve
#   * forgetting_regret         -> custom, below: evidence you DROPed
#
# Run the SAME questions through baselines to get the frontier:
#   full_context | recency_window(n) | rag_only | your_policy
# Your policy "helps" iff it dominates: >= accuracy at fewer tokens.

@dataclass
class EvalResult:
    by_type: dict = field(default_factory=dict)     # {qtype: [correct...]}
    tokens: list = field(default_factory=list)
    forgetting_regret: int = 0                       # #Qs whose evidence we DROPed


def run_eval(
    dataset: Iterable[dict],       # each: {question, question_type, answer, sessions:[{turns:[...]}], evidence_turn_ids}
    score_fn: ScoreFn,
    answer_fn: Callable[[str, list], str],   # (question, retrieved_memories) -> answer
    judge_fn: Callable[[str, str, str], bool],  # (question, oracle, answer) -> correct?
    token_count: Callable[[str], int],
) -> EvalResult:
    res = EvalResult()
    for item in dataset:
        store = MemoryStore(":memory:")            # fresh store per question
        dropped_turns: set[str] = set()

        for sess in item["sessions"]:
            spans = [(store.log_turn(sess.get("id", "s"), t["role"], t["content"]), t["content"])
                     for t in sess["turns"]]
            for s in store.consolidate(spans, score_fn):
                if s.verdict == Verdict.DROP:
                    dropped_turns.add(s.turn_id)

        mems = store.retrieve(item["question"])
        ctx = "\n".join(m["content"] for m in mems)
        ans = answer_fn(item["question"], mems)
        correct = judge_fn(item["question"], item["answer"], ans)

        qtype = item["question_type"]
        res.by_type.setdefault(qtype, []).append(correct)
        res.tokens.append(token_count(ctx))
        # forgetting regret: did we DROP a turn that was this question's evidence?
        if dropped_turns & set(item.get("evidence_turn_ids", [])):
            res.forgetting_regret += 1
    return res


def summarize(res: EvalResult) -> dict:
    acc = {t: sum(v) / len(v) for t, v in res.by_type.items()}
    overall = sum(sum(v) for v in res.by_type.values()) / sum(len(v) for v in res.by_type.values())
    return {
        "overall_accuracy": round(overall, 3),
        "accuracy_by_type": {t: round(a, 3) for t, a in acc.items()},
        "avg_context_tokens": round(sum(res.tokens) / len(res.tokens), 1),
        "forgetting_regret": res.forgetting_regret,   # lower is better; 0 is the goal
    }
