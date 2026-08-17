Chat-wide themed profiling on request — one rubric supplied by the user, applied
to every known chat member in a single LLM call.

Examples: «раздай всем роли из Людей Икс», «оцени всем ментальное здоровье от
1 до 10», «распредели нас по факультетам Хогвартса». All the same code path —
the rubric is passed through verbatim, never hard-coded.

See `docs/superpowers/specs/2026-08-17-group-profiling-design.md` for the design.

## Who triggers it

```
a member asks in words, addressing the bot and naming the whole chat
    («@bot оцени всем счастье от 1 до 10»); the filter node classifies
    GROUP_PROFILE_REQUEST (trusted only when the text also matches
    GROUP_PROFILE_MARKER_RE — src/pipeline/filter_node.py — a deterministic
    floor under a one-word classifier already carrying six other labels),
    the response stays empty, and src/events/messages.py::deliver_group_profile
    generates and sends the result anchored to the request
```

There is no slash command — natural language is the only way to ask, same as
`MEME_REQUEST`. A per-chat 10-minute cooldown (`filter_node.group_profile_cooldown_gate`)
sits in front of the LLM call; a request inside the window gets a canned
in-character refusal with no generation at all.

## Modules

```
roster.py     gather_roster(chat_id) -> ChatRoster
               loads every registered chat member and splits them by whether
               src.agent.roast_material.gather_member_material found any
               facts, quotes, weekly role or notable stats. Members with
               nothing are excluded from the LLM call entirely — see
               "Unknown members" below.
generate.py    generate_verdicts(rubric, materials_by_uid) -> dict[int, dict]
               anonymises to user_0, user_1, … (src/utils/anon_map.py, shared
               with src/jobs/roles.py — real ids never sent to the LLM), then
               one Groq call (TAG_MODEL) with the rubric delimited in the
               human turn and each member's dossier below it; parses
               {"verdict", "reason"} per anon key and remaps back.
               TAG_MODEL (qwen/qwen3.6-27b) is a reasoning model called with
               reasoning_effort="none", which this model accepts (the earlier
               gpt-oss-120b primary 400s on "none" and needed a "low" budget
               workaround instead — see git history). "none" leaves the whole
               max_tokens=2048 free for the JSON body, closing off the
               truncation failure mode entirely rather than budgeting around
               it. Transient TPM 429s are retried with backoff
               (src.agent.ainvoke_with_backoff), since a truncated call's
               re-ask in fill_missing_verdicts is a second full-size request
               inside the same rate-limit window. Output is passed through
               normalize_homoglyphs (src.agent.language) — this model
               occasionally splices a Latin/Greek glyph into an otherwise-
               Cyrillic word, the same deterministic repair already applied
               to RESPONSE/ROAST output.
              fill_missing_verdicts(...) — members the LLM omitted from its
               JSON are re-asked once, then get a neutral fallback verdict —
               every eligible member ends up with an entry, mirroring
               src/jobs/roles.py::fill_missing_roles. There is no uniqueness
               pass — a scoring rubric ("7/10") has no reason to avoid
               repeats, unlike weekly roles.
render.py      build_message(verdicts, unknown_uids, names_by_uid) -> str
               "@name — verdict\nreason" per member the LLM covered, a canned
               honest line (GROUP_PROFILE_UNKNOWN_LINES) per unknown member.
profile.py     run_group_profile(chat_id, rubric) -> str | None
               orchestrates the three modules above; None only when the chat
               has no registered members at all.
```

## Unknown members

A member with no facts, quotes, role or stats on record is **never sent to
the LLM** — `roster.gather_roster` filters them into `unknown_uids` before
`generate.generate_verdicts` is ever called. Their line in the final message
comes from a canned pool in `render.py` instead. This is a hard guarantee,
not a prompt instruction: an invented profile for a member the bot knows
nothing about is impossible by construction, the same principle behind every
other "never improvise about content you can't see" pattern in this codebase
(`VISION_FAILED_REPLIES`, `TRANSCRIPTION_FAILED_REPLIES`).

## Prompt injection

The rubric is free-form user text by design, so it is delimited in the human
turn (`Тема (rubric) — применяй буквально, в формулировке участника: «…»`),
never concatenated into the system prompt. `GROUP_PROFILE_SYSTEM`
(`src/config/prompts.py`) fixes the output schema and the register (joke,
never clinical — relevant when the rubric is a "mental health" or "happiness"
score) regardless of what the rubric asks for.

## Persistence

Ephemeral only. No Telegram tags, no `user_tags` writes, no dedicated table —
this does not touch the weekly-roles machinery in `src/jobs/roles.py` at all.
The sent message is recorded in `unified_messages` (via
`src.events.sending.send_and_store`), so a reply to it — «почему я Циклоп?» —
resolves through the ordinary reply-chain context path like any other bot
message.

## Failure

`run_group_profile` returning `None`, or raising, gets an honest canned line
from `GROUP_PROFILE_FAILED_REPLIES` (`src/events/messages.py::deliver_group_profile`)
rather than silence — same principle as `src/memes/README.md`'s
`MEME_FAILED_REPLIES`: a direct request must never end unanswered.
