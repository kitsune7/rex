# runs every N turns / on a timer — the "consolidation pass"
def consolidate(working_ctx, memory_store):
    # 1. LLM scores each span for future-utility + type
    #    types: DROP (true noise), KEEP_HOT (stays in context),
    #           OFFLOAD (move to store, replace with a pointer/cue),
    #           MERGE (fold into an existing memory)
    verdicts = llm_classify(working_ctx, task_profile)

    for span, v in zip(working_ctx, verdicts):
        if v.type == "OFFLOAD" or v.type == "MERGE":
            memory_store.upsert(span, embedding, metadata={
                "ts": now(), "access_count": 0, "salience": v.score
            })
    # 2. rebuild the hot context: KEEP_HOT + a compact index of what got offloaded
    return rebuild(working_ctx, keep=KEEP_HOT, breadcrumbs=offloaded_cues)

# at retrieval time: hybrid (vector + recency + access_count boost),
# and bump access_count so frequently-pulled memories resist decay