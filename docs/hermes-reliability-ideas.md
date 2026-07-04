# Reliability Ideas From Hermes For Rex

## Context

[Hermes Agent](https://hermes-agent.nousresearch.com/docs) is a self-hosted autonomous
agent from Nous Research. It targets a very different shape than Rex — long-running,
multi-platform, 60+ tools — but it has spent a lot of design effort on making a
tool-calling loop *dependable*, and several of those patterns transfer directly to
Rex's voice assistant.

This document is a companion to [`forge-reliability-plan.md`](forge-reliability-plan.md).
Where Forge contributes low-level loop guardrails (rescue parsing, retry nudges,
response validation), Hermes contributes higher-level reliability patterns:
**bounded persistent memory, clarify-before-acting, fail-closed confirmation,
smart approval, self-improvement, and command safety floors.** The two are
complementary — nothing here replaces the Forge sequence.

Every recommendation below is mapped to concrete Rex code so it can be picked up
incrementally.

## What Rex looks like today (reliability surfaces)

- Two near-duplicate agent drivers: `src/agent/agent.py` (`run_voice_agent`) and
  `src/server/voice_agent.py` (`VoiceAgent`). Logic has already drifted between them
  (prompts, confirmation formatting, follow-up detection). Drift is a reliability risk.
- History is trimmed with a hard `MAX_HISTORY_MESSAGES = 20` slice in both drivers
  (`src/agent/agent.py:210`, and the constant in `src/server/voice_agent.py:65`).
  A blind tail slice can drop tool results — or a mid-flight tool/confirmation pair —
  and corrupt the message sequence.
- Only `create_reminder` is confirmable (`CONFIRMABLE_TOOLS` in
  `src/agent/tools/reminder.py:415`). `delete_reminder` and `update_reminder` mutate
  state with no confirmation and no prerequisite that an ID was actually looked up.
- Malformed / empty model output is not repaired — it is passed through as text
  (`extract_text_response`), and any exception collapses to a single generic line,
  `"Sorry, I encountered an error processing your request."`
  (`src/rex/states/processing.py:80`). This is exactly the failure the reliability
  plan wants to reduce.
- No timeout or retry budget around the LLM call. A slow or hung local model stalls
  the whole voice turn while the thinking tone plays.
- Memory is per-thread and in-process only (`MemorySaver`). Nothing about the user or
  environment survives a restart, so Rex re-learns the same things every session.
- Rescue parsing already exists (`src/agent/rescue_parsing.py`) — good, and it lines
  up with Hermes/Forge thinking.

## Recommendations

Each item lists the Hermes source, the Rex touch points, and a rough priority.

---

### 1. Bounded, persistent memory instead of a 20-message tail slice

**Hermes:** [Persistent Memory](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)
keeps two small, char-bounded stores — `MEMORY.md` (environment/agent notes) and
`USER.md` (user profile) — injected once at session start as a *frozen snapshot*.
Critically, memory **does not silently auto-compact**: a write that would exceed the
limit returns an error and the agent must consolidate first. Past conversations are
recalled separately via `session_search` (SQLite FTS5), so the always-in-prompt budget
stays tiny while full history stays searchable.

**Why it matters for Rex:** A voice assistant that forgets your name, your timezone,
and how you like to be spoken to between every launch feels broken even when it is
technically working. And the current `history[-20:]` slice is itself a correctness
bug: it can sever a tool-call / tool-result pair.

**Recommendation:**
- Add a tiny persisted user-profile store (name, timezone, verbosity preference, how
  to pronounce things) loaded into the system prompt in `base_agent.get_system_prompt`
  and `voice_agent._server_system_prompt`. Keep it small and bounded like Hermes
  (~500 tokens) so voice latency and prompt cost stay flat.
- Replace the blind tail slice with structure-aware compaction (already Plan 5 in the
  Forge doc): never split a tool-call/result pair, always keep the pending confirmation
  and the current turn, and summarize older turns rather than dropping them.
- Borrow the **frozen-snapshot** rule: compute the profile/system block once per session
  to preserve the local model's prefix cache; persist writes to disk immediately but
  only surface them in the prompt next session.

**Priority:** High. This is the single biggest perceived-reliability win and it
directly fixes an existing history-trimming bug.

---

### 2. Fail-closed confirmation with a timeout

**Hermes:** dangerous-command approval is
[**fail-closed**](https://hermes-agent.nousresearch.com/docs/user-guide/security#approval-timeout):
if the user does not answer within `approvals.timeout` (default 60s), the action is
*denied*, not executed.

**Why it matters for Rex:** `VoiceAgent.confirm` and `confirm_tool_call` assume a yes/no
eventually arrives. In a voice loop, the user can walk away, the STT can mis-hear, or
the session can be interrupted mid-confirmation. Today a pending `create_reminder` can
sit in `MemorySaver` indefinitely, and an ambiguous audio reply could be misread as
approval.

**Recommendation:**
- Give pending confirmations an explicit timeout in the `AWAITING_CONFIRMATION` state.
  On expiry, resolve as *declined* and speak a short "I'll skip that for now."
- Only treat a clear affirmative as approval; anything ambiguous re-prompts once, then
  fails closed.

**Priority:** High. Low effort, and it closes a real "accidental mutation" gap.

---

### 3. A `clarify` step for ambiguous or ID-based tool calls

**Hermes:** ships a dedicated `clarify` tool
([Tools & Toolsets](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools))
so the agent asks a question instead of guessing when a request is under-specified.

**Why it matters for Rex:** `delete_reminder`/`update_reminder` take a `reminder_id`
the user rarely says out loud, and `parse_datetime` silently picks "the soonest am/pm"
for ambiguous times (`src/agent/tools/reminder.py`). Guessing wrong here means deleting
the wrong reminder or setting an alarm for the wrong half of the day — high-annoyance,
hard-to-undo failures.

**Recommendation:**
- Add a first-class "ask a clarifying question" path (a small tool, or a structured
  result the voice loop already understands) for: unknown reminder IDs, multiple
  matching reminders, and genuinely ambiguous times.
- Pair it with a **prerequisite policy** (Forge Plan 4): `delete_reminder` /
  `update_reminder` should require a recent `list_reminders` in-thread or an explicit,
  unambiguous match before executing.

**Priority:** High. Directly targets the "risky/ambiguous mutation" goal in the Forge plan.

---

### 4. "Smart" tiered approval instead of one hardcoded tool

**Hermes:** approvals have three
[modes](https://hermes-agent.nousresearch.com/docs/user-guide/security#approval-modes) —
`manual`, `smart` (an auxiliary model auto-approves clearly-safe, auto-denies clearly-
dangerous, escalates only the uncertain middle), and `off` — plus an always-on
**hardline blocklist** that cannot be overridden.

**Why it matters for Rex:** confirmation is currently a single hardcoded set
(`CONFIRMABLE_TOOLS = {"create_reminder"}`). Read-only tools (`get_current_time`,
`check_timers`, `list_reminders`, `calculate`) never need a prompt; destructive tools
(`delete_reminder`) arguably always should. Confirming everything is annoying;
confirming nothing is unsafe.

**Recommendation:**
- Classify tools by risk (read-only / reversible / destructive) next to their
  definitions, and drive confirmation off that classification instead of a name set.
- Adopt the **floor** idea: a small "always confirm, never auto-run" category for
  destructive tools that no config path can bypass.
- A full auxiliary-model risk assessor is likely overkill for Rex's tiny toolset;
  static risk tiers capture most of the value.

**Priority:** Medium. Builds cleanly on recommendation 3.

---

### 5. LLM-call timeout + retry budget (turn a hang into a graceful reply)

**Hermes:** every terminal command has a
[timeout](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools#configuration)
and long work runs as a managed background `process` (poll/wait/kill), so a single slow
operation never wedges the loop.

**Why it matters for Rex:** there is no timeout on `agent.invoke`. With a local
OpenAI-compatible backend, a stalled model leaves Rex playing the thinking tone forever.
And a malformed response goes straight to text instead of getting one corrective retry.

**Recommendation:**
- Wrap the model call (`RescueParsingChatOpenAI` / `create_chat_model`) with a request
  timeout and a small per-turn retry budget (this is Forge Plan 3, and the guardrail
  wrapper is the natural home for it).
- On timeout/exhaustion, speak a *specific* recoverable message ("I'm having trouble
  reaching my brain right now, try again in a sec") instead of the single generic
  `processing.py` error line — and log the guardrail event.

**Priority:** High. Small change, removes a total-hang failure mode.

---

### 6. Consolidate the two agent drivers behind one guardrail wrapper

**Hermes:** one tool-execution and safety pipeline serves every surface (CLI, gateway,
subagents). Rex has two hand-maintained copies of confirmation, follow-up detection,
and history trimming.

**Recommendation:** implement the Forge Plan 3 guardrail wrapper *once* and have both
`run_voice_agent` and `VoiceAgent.chat` call it, so retry budget, compaction, and
prerequisite checks can't drift between desktop and server. This is a prerequisite for
almost every other item here landing in both places.

**Priority:** High (enabler). Mostly refactor, low behavioral risk.

---

### 7. A background self-improvement / memory-capture review

**Hermes:** after a turn, a
[background review](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory#background-review-notifications-displaymemory_notifications)
can quietly capture durable facts to memory, optionally on a *cheaper auxiliary model*,
gated by `write_approval`.

**Why it matters for Rex:** once recommendation 1 exists, Rex can get better over time —
"user always means PM for gym reminders," "user's kid is named Sam" — without asking the
user to configure anything.

**Recommendation:** after a turn completes (off the voice-latency path), run a small,
optional capture step that proposes profile updates. Keep it **off by default or
approval-gated** — silently writing wrong assumptions about a user is itself a
reliability failure, which is exactly why Hermes added `write_approval`.

**Priority:** Low / later. Only worth it after persistent memory (1) lands.

---

### 8. Guardrail/security hygiene if Rex ever persists or ingests text

**Hermes:** memory entries are
[scanned for prompt-injection and exfiltration](https://hermes-agent.nousresearch.com/docs/user-guide/security#context-file-injection-protection)
before being trusted, and fail-closed on ambiguity.

**Why it matters for Rex:** most of Hermes' security surface (containers, SSRF,
allowlists) doesn't apply to a local single-user voice assistant. But the moment Rex
persists a user profile (1) or captures memory (7), anything spoken becomes text that
gets re-injected into the system prompt.

**Recommendation:** if/when persisted memory lands, add a light sanitization pass
(strip control/zero-width characters, cap length, reject obvious "ignore previous
instructions" content) before writing. Skip the rest of Hermes' security model — it's
scoped to a multi-user, tool-heavy deployment Rex isn't.

**Priority:** Low, but a hard prerequisite for shipping 1 and 7 safely.

---

## Suggested sequence

Interleaves with the existing Forge plan rather than competing with it:

1. **Consolidate drivers behind one guardrail wrapper** (6) — enables everything else.
2. **LLM timeout + retry budget** (5) and **fail-closed confirmation** (2) — both small,
   both remove real hang/mutation failure modes.
3. **Clarify + prerequisite policies for ID/ambiguous tools** (3), then **risk-tiered
   approval** (4).
4. **Persistent bounded user profile + structure-aware compaction** (1), with the
   **sanitization pass** (8) shipped alongside it.
5. **Optional background memory capture** (7) once 1 is proven.

Validate each step against the existing scenario evals (`src/tests/test_scenario_evals.py`),
adding new fixtures for: confirmation timeout, ambiguous-time clarify, delete-without-ID
refusal, model-timeout recovery, and profile-survives-restart.

## What to deliberately skip

Hermes carries a lot that Rex should *not* adopt — it would add fragility, not remove it:

- Multi-agent delegation / subagents (also a Forge non-goal).
- Container/SSH/Modal terminal backends, SSRF and website blocklists, supply-chain
  scanning — these secure a multi-user, shell-executing agent. Rex runs a fixed, tiny,
  non-shell toolset for one local user.
- Messaging-gateway authorization, DM pairing, and per-platform allowlists.
- The full external memory-provider ecosystem (Honcho, Mem0, etc.) — a single bounded
  profile file is the right size for Rex.
