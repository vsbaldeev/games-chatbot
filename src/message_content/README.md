# Message content

Per-message record of what the bot actually put into the chat — the counterpart
of `src/feedback`, which records how members reacted. One row per bot message
that answers a pipeline turn, in `bot_message_content` (schema in
`src/store/README.md`). `bot_message_feedback` is unchanged; the two tables join
on `(chat_id, message_id)`. No rates are stored: every figure is a `COUNT`
taken at query time.

The unit is a **message**. Things that never become a message — the 😴 cooldown
reaction, emoji-only wind-down, silent drops — are not recorded here.

## What a row holds

```
content_kind    why the message exists (see below)
worker_status   how the tool-calling worker stage ended
response_label  did the bot actually answer? (explicit free-form replies only)
sent_at         when the row was written
```

### content_kind

LLM-written:

```
response               normal reply, the model chose freely
link_summary           Shorts / Instagram / TikTok / YouTube summary — nobody asked a question
wind_down_reply        written under the wind-down directive (also the wind-down
                       meme / group-profile refusals)
insult_reply           written under the BOT_INSULT directive (full tier only)
```

Canned (fixed text, no LLM):

```
prompt_injection_reply guard flagged the message — includes guard false positives
limit_notice           Groq daily quota / rate limit
context_limit_notice   context window exceeded
cap_notice             Instagram + TikTok shared daily cap, group-profile cooldown
media_failure_notice   could not transcribe / see / fetch (voice, photo, Shorts, links)
error                  generic failure line
```

Canned kinds are stamped on `BotState.content_kind` by the node that writes the
text (guard, filter, router). The LLM-written kinds are derived at delivery by
`kinds.content_kind_for`, with precedence wind-down > insult > link summary >
response — a wound-down insult carries both flags, but the wind-down directive
is the one the model was told to obey.

### worker_status

Set by `WorkerNode` on every run, carried in `BotState.worker_status`:

```
ok                the worker ran
skipped           deliberately not used (insult, wind-down, meme, group profile, link summaries)
context_overflow  ContextLengthError absorbed — the answer has no worker facts behind it
failed            unexpected exception absorbed — same
NULL              the worker never finished: a canned early exit (router / filter /
                  guard) or a pipeline that aborted before the worker returned
```

`context_overflow` and `failed` are the silent failures: the pipeline carries on
with empty worker output, the user sees a normal-looking reply, and nothing but
a log line recorded it before this table.

### response_label

One background call (`response_labeller.label_response`) judges the member's
question against the delivered reply:

```
ANSWERED    engages with the question and gives something usable
REFUSED     explicitly declines
NON_ANSWER  no refusal, nothing usable: dodge, vague "don't know", off-topic,
            or only a counter-question
UNCLEAR     the model could not tell, answered outside the contract, or the call failed
NULL        not labelled yet, or not applicable
```

Only `content_kind = 'response'` with `trigger = 'explicit'` and a non-blank
question is labelled. The question is the transcript (`processed_text`) for
voice and video notes, else the typed text. Labelling runs in a background task
(`recorder.record` spawns it) on OpenRouter only (`REPLY_LABEL_MODELS`), so it
never competes with chat replies for Groq quota and never sits on the delivery
path. The call is not retried.

## Modules

```
kinds.py              content_kind_for(state), question_text(state)
recorder.py           record(...) — registers the row, spawns labelling when
                      should_label(); label_and_store(...). Never raises.
response_labeller.py  build_payload / parse_label / label_response — strict JSON
                      contract {"label": ...}; an answer outside it returns None,
                      a call that never reached a model raises LabellingCallFailed
```

Write sites: `deliver_and_record` → `record_content` (every delivered pipeline
reply) and `notify_pipeline_failure` (limit, context and error notices), both in
`src/events/messages.py`. Retention: `src/jobs/cleanup.py` (60 days).

## Explicit limits

- Memes, weekly roles, group-profile messages and their failure lines are not
  recorded — they come out the same way every time.
- `prompt_injection_reply` counts what the guard flagged, not confirmed attacks:
  gaming slang looks like jailbreak phrasing and the guard is a small model.
- `UNCLEAR` mixes "the labeller could not tell" with "the labelling call failed".
  A message with a NULL label is either not labelled yet or never eligible.
- Probes in `tests/integration/message_content/` are cleaner than real chat
  messages and read optimistically. Before trusting the REFUSED / NON_ANSWER
  rates, hand-label ~50 real `response` rows from `bot_llm_log` and compare.
- The labeller sees only the member's message and the reply, not the thread. A
  terse follow-up («а почему?») judged against a reply that leans on the
  conversation can come out NON_ANSWER or UNCLEAR, so those two labels slightly
  overstate real misses on follow-up questions.
- No backfill: the table starts empty at deploy, and retention is 60 days.
- A search-notification message edited into its final text is recorded once,
  under its original id; a later edit does not overwrite the row.

## Reading the numbers

Content mix and how members reacted to each kind:

```sql
SELECT content.content_kind, COUNT(*) AS messages,
       COUNT(*) FILTER (WHERE feedback.ignored) AS ignored,
       SUM(feedback.positive_reactions + feedback.positive_replies) AS positive,
       SUM(feedback.negative_reactions + feedback.negative_replies) AS negative
FROM bot_message_content AS content
JOIN bot_message_feedback AS feedback USING (chat_id, message_id)
WHERE content.sent_at >= now() - interval '7 days'
GROUP BY content.content_kind ORDER BY messages DESC;
```

Silent worker failures against answer quality:

```sql
SELECT worker_status, response_label, COUNT(*) AS messages
FROM bot_message_content
WHERE content_kind = 'response' AND sent_at >= now() - interval '7 days'
GROUP BY worker_status, response_label;
```

Refusal rate by model:

```sql
SELECT log.model, COUNT(*) AS labelled,
       COUNT(*) FILTER (WHERE content.response_label = 'REFUSED') AS refused
FROM bot_message_content AS content
JOIN bot_llm_log AS log
  ON log.chat_id = content.chat_id AND log.bot_message_id = content.message_id
WHERE content.response_label IS NOT NULL AND content.sent_at >= now() - interval '7 days'
GROUP BY log.model;
```
