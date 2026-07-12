## Run as a background daemon

I'd like to be able to run Rex in the background as a daemon at startup and have it use the Bedrock models by default.

Its logs should be easily accessible.

## Text-insertion tool

I want to make it so that Rex can insert text at wherever my cursor is currently.

## Conversation flow

I can start a new conversation by saying, "Hey Rex."
There should be a kill conversation tool that ends the conversation naturally, but it will also deterministically end if there's no response within some time limit, like, say, 10 seconds.

At the end of the conversation, it should be moved into memory storage.

There should also be a way to test conversations or to mark a conversation as a test so that it doesn't persist memory of the conversation or attribute importance to memory retreival. Basically, it should be a clean run that doesn't affect any future sessions.

## Memory Storage

I need some way to store, index, and retreive memories. Short-term memory should be super-fast retrieval and should have a regular loop to decay short-term memory based on time, frequency, and importance. An LLM can make a "dreaming" pass.

Long-term memory should be more persistent and safe from trimming passes. Things should only get moved into long-term memory when explicitly requested or when accessed frequently.

Test memory on LoCoMo (Maharana et al. 2024) benchmark first. Use LongMemEval (Wu et al. 2024) for a more serious test.

### Storage Methodologies

Mempalace for the store and indexing.
Graphiti for the bitemporal supersession and edge-invalidation.

MemCog for high-thinking mode where a more powerful model can actively explore memory. This can be an upgrade once the memory store is solid, but it should have a highly capable model driving it every turn.

### Failure modes

1. Over-aggressive DROP shows up as abstention false-answers (you confidently answer a question whose evidence you deleted) and **information-extraction/single-session misses**.
2. Broken MERGE shows up as **knowledge-update** failures (stale values win).
3. Dropping metadata shows up as **temporal-reasoning** failures

## Other tools that could help Rex

- Use computer:
  - Switch window
  - View window
  - Move mouse
  - Type
  - Run terminal command
  - Run script
- FS General:
  - Write file
  - Read file
  - Search computer
- Notes:
  - Search notes
  - Create note(s)
  - Read note(s)
  - Move note(s)
  - Edit note(s)
  - Delete note(s)

Work-specific:
- Jira:
  - Create ticket (epic, story, or bug)
  - Read ticket
  - Update ticket
  - Delete ticket
- Slack:
  - Post in Slack channel
  - Send DM
  - Reply to message thread
- Github
  - ...

## Animated Rex on screen

I think it would be very interesting to have Rex on the computer screen and wrap emotion tags around certain texts that can trigger specific animations. That would be more reliable than the robot hardware that I have and would have a similar effect in terms of how I perceive it.