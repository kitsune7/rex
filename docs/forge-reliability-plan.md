# Forge Reliability Plan For Rex

## Context

Forge is a reliability layer for self-hosted LLM tool-calling. It focuses on making one agentic loop more dependable through response validation, rescue parsing, retries, required-step enforcement, context compaction, inference-slot serialization, and scenario evals.

Rex already has its own orchestration shape: a voice state machine, LangGraph/LangChain agent, human confirmation flow, audio coordination, reminders, timers, and an HTTP voice agent. The best fit is to borrow Forge's reliability patterns without replacing Rex's state machine or agent stack.

## Goals

- Make tool use more reliable for local OpenAI-compatible models.
- Reduce silent or generic failures during voice turns.
- Prevent risky or ambiguous tool mutations without enough context.
- Keep long conversations useful without unbounded history growth.
- Add repeatable reliability tests for voice-assistant workflows.

## Non-Goals

- Do not replace Rex's state machine with Forge's `WorkflowRunner`.
- Do not copy Forge wholesale into Rex.
- Do not make Forge a hard runtime dependency until a proxy experiment proves useful.
- Do not add multi-agent orchestration. Forge explicitly does not target that layer, and Rex does not need it yet.

## Recommended Sequence

1. Add transcript-driven scenario evals.
2. Run a low-cost Forge proxy experiment in front of Rex's configured LLM endpoint.
3. Add a Rex-native guardrail wrapper around agent invocation.
4. Add prerequisite policies for risky and ID-based tools.
5. Replace fixed message trimming with compacted voice-session memory.
6. Add inference serialization and priority handling if concurrent model calls become a real issue.

## Plan 1: Scenario Evals

Add deterministic tests that simulate user transcripts and assert expected outcomes. This should come first because it gives every later reliability change a baseline.

Initial scenarios:

- Set a timer with a simple duration.
- Ask for current timers after setting one.
- Stop a ringing or named timer.
- Create a reminder and require confirmation before execution.
- Reject a reminder confirmation and ensure it is not created.
- Update or delete a reminder only after an ID is known.
- Handle malformed or empty LLM output.
- Continue a long conversation without losing important tool results.

Implementation notes:

- Use a fake chat model or mocked agent response rather than a live model for unit-level reliability tests.
- Keep a smaller optional integration eval suite that can run against the configured local model.
- Store scenario fixtures as plain data so new workflows can be added cheaply.

## Plan 2: Forge Proxy Experiment

Rex already reads the OpenAI-compatible base URL from `settings.toml`, so Forge proxy mode can be tested without code changes or with minimal settings changes.

Experiment:

1. Start the current backend as usual.
2. Run Forge proxy in front of it.
3. Point Rex's `llm.api_base` at the Forge proxy URL.
4. Run the scenario evals against both direct backend and proxy backend.

Success criteria:

- Fewer malformed tool-call failures.
- Fewer generic "encountered an error" voice responses.
- No meaningful latency regression for common timer/reminder requests.
- No breakage in LangGraph confirmation pauses.

If the proxy improves reliability, keep it as an optional documented deployment mode. If it does not, still borrow the patterns directly in Rex.

## Plan 3: Rex-Native Guardrail Wrapper

Add a small reliability layer around Rex's agent invocation paths. Rex has both desktop and server agent flows, so the wrapper should be shared or mirrored deliberately.

Behavior to add:

- Track a per-turn retry budget for malformed responses or unusable tool calls.
- Turn validation failures into corrective model messages rather than immediately surfacing a generic error.
- Detect unknown tools and nudge the model toward the available tool names.
- Preserve a clear fatal path after repeated failures.
- Log guardrail events for debugging and future eval reports.

This is inspired by Forge's `ResponseValidator`, `ErrorTracker`, and nudge templates, but should use LangChain/LangGraph message types and Rex's existing tool objects.

Open design question:

- Decide whether retries should happen inside `run_voice_agent` and `VoiceAgent.chat`, or whether a shared lower-level helper should own both desktop and server behavior.

## Plan 4: Tool Prerequisite Policies

Use Forge's required-step and prerequisite idea for Rex tools that are risky, ambiguous, or depend on user-visible IDs.

Recommended policies:

- `create_reminder` remains confirmable before execution.
- `delete_reminder` should require either a fresh `list_reminders` context or a highly explicit reminder ID from the user.
- `update_reminder` should require a known reminder ID or a prior list/lookup step.
- Future destructive tools should default to confirmation.
- Future lookup-then-act tools should declare prerequisites next to their tool definitions.

Keep prerequisite state outside message history so context trimming or compaction cannot erase what tools have actually run.

## Plan 5: Context Compaction

Replace fixed `MAX_HISTORY_MESSAGES = 20` trimming with a voice-session memory policy.

Suggested behavior:

- Always preserve the current user request, pending confirmation state, and the most recent turns.
- Preserve concise summaries of executed tools and their outcomes.
- Drop retry nudges and transient correction messages first.
- Compress long tool outputs into one-line summaries.
- Keep enough recent context for follow-up questions after Rex asks a question.

This should be simpler than Forge's full tiered compaction at first. Rex's conversations are voice-oriented, so a short structured summary is likely more valuable than preserving raw old messages.

## Plan 6: Inference Serialization And Priority

Forge's `SlotWorker` is useful if Rex has multiple possible LLM callers competing for one local model slot.

Apply this only if needed. Likely triggers:

- The HTTP server and desktop flow can run at the same time.
- Reminder delivery starts using LLM-generated speech or tool decisions.
- Interruptions need to cancel in-flight lower-priority model work.
- Local backend calls conflict or queue poorly.

Priority model:

- Highest: user interruption and safety/stop commands.
- High: due reminders and ringing timers.
- Normal: direct user questions.
- Low: background summarization or maintenance work.

## Risks

- Retrying agent calls can accidentally duplicate tool execution if placed after tools run. Retries must happen before execution or must be idempotency-aware.
- Confirmation flow must remain human-in-the-loop. Guardrails should not auto-approve confirmable tools.
- Proxy mode may change timing or streaming behavior. Treat it as optional until evals prove it.
- Context compaction can lose user intent if summaries are too aggressive.

## First Implementation Slice

The smallest useful slice is:

1. Add scenario fixtures and a fake-model test path.
2. Add tests for malformed output, unknown tools, and reminder confirmation.
3. Add a simple retry/error budget around pre-tool agent responses.
4. Compare direct backend versus Forge proxy with the same scenarios.

That slice should make the next architectural decisions evidence-based instead of speculative.
