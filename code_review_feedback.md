# Code Review: `src/dnd.py`

Scope: full module review focused on correctness (state leaks / stuck games), LLM prompt + parsing robustness, error handling, structure, and mode branching. Line numbers below refer to the current `src/dnd.py`.

---

## Critical issues (genuine bugs)

### 1. Lobby auto-start race — concurrent join clicks can crash with `KeyError`
**Location:** `__handle_join` lines 597-634.

The function appends a player synchronously, checks the count, then performs several `await`s (`query.answer`, `query.edit_message_text`) before doing `del __lobbies[chat_id]` (line 617). asyncio is single-threaded, but callbacks suspend at every `await`, so two concurrent join clicks while the lobby has 2 players already can both enter the auto-start branch:

- Click A appends → `len(players) == 3` → enters auto-start → `await query.answer(...)` → suspends.
- Click B appends → `len(players) == 4` → also enters auto-start → `await query.answer(...)` → suspends.
- Click A resumes → `__lobby_timeout_jobs.pop(...)` succeeds, `del __lobbies[chat_id]` succeeds, schedules start job.
- Click B resumes → `__lobby_timeout_jobs.pop(...)` returns None, then `del __lobbies[chat_id]` raises `KeyError`.

Even before the crash, the snapshot `players = list(lobby.players)` (line 616) sees different lists for the two clicks — both could schedule `__start_game_job` with overlapping but different player rosters.

**Fix:** pop the lobby and the timeout job *before* any `await`:

```python
entry = __lobbies.get(chat_id)
if not entry:
    await query.answer("Лобби уже закрыто.", show_alert=True)
    return
lobby, max_rounds, mode = entry
if any(player_id == clicker_id for player_id, _ in lobby.players):
    await query.answer("Ты уже в отряде!", show_alert=True)
    return
lobby.players.append((clicker_id, clicker_username))
if len(lobby.players) < DND_MIN_PLAYERS:
    ... # existing branch
    return
# Auto-start: pop synchronously before any await
popped = __lobbies.pop(chat_id, None)
if popped is None:
    await query.answer()       # someone else already triggered start
    return
players = list(lobby.players)
job = __lobby_timeout_jobs.pop(chat_id, None)
if job:
    job.schedule_removal()
# now safe to await
```

### 2. `__active_chats` leaked in error paths inside `__start_game_job` and `__next_round_job`
**Locations:** `__start_game_job` lines 741-745, `__next_round_job` lines 982-986.

When the post-generation `edit_message_text` fails:

```python
except TelegramError as error:
    logger.warning(...)
    __active_chats.discard(chat_id)
    __active_games.pop(chat_id, None)
    return
```

`__active_chats` and `__active_games` are cleaned up — but **no message is shown to the chat**. Players see "🎲 Generating..." or a stale loading screen forever, the message has no buttons, and the chat is silently released. From the player's POV the game is stuck.

The outer `__next_round_job` `except` (lines 957-968) handles this correctly by sending an "Приключение прервано" message. Apply the same pattern to lines 741-745 and 982-986.

### 3. Silent edit failures in the live game loop leave the message frozen
**Locations:** `__handle_action` lines 665-672, `__start_game_job` line 734, `__next_round_job` line 975.

These three call-sites use `edit_message_text` directly with `parse_mode="Markdown"` and swallow `TelegramError` with `pass` (or warn-and-return). The narrative paths use `__edit_safe` (lines 992-1006), which retries without parse_mode on `BadRequest`. The live-game-render paths do not.

If the LLM-generated scenario contains a stray `*` or `_`, Telegram returns 400 "Can't parse entities" → `BadRequest` → silently dropped. The result:
- In `__handle_action` (line 666): the player's `✅` mark never appears even though their choice is recorded.
- In `__start_game_job` (line 734): the entire game show fails; `__active_chats.discard` runs and the chat is released without the user knowing why.
- In `__next_round_job` (line 975): same.

**Fix:** route every LLM-text-bearing edit through `__edit_safe`. Keep the keyboard parameter on the safe path (it currently doesn't accept one — extend it).

### 4. Lobby/scenario text is not Markdown-escaped before being passed with `parse_mode="Markdown"`
**Locations:** `__build_lobby_text`, `__build_game_text`, scenario rendering throughout.

`__build_game_text` interpolates `game.scenario` directly (line 134). The LLM is told to produce plain text but routinely emits underscores (Russian usernames in lore, math, etc.) and asterisks. `to_telegram_md` only converts `**bold**` → `*bold*` and strips table separators — it does **not** balance stray `*` / `_` / `[` / `(`.

This is the root cause of #3 and gets noticed any time the LLM hallucinates emphasis. Either:
- Switch to `parse_mode="MarkdownV2"` and escape all reserved characters in scenario / actions before interpolation, or
- Pre-sanitize LLM output: strip lone `*` / `_` characters or balance them.

### 5. Action regex matches `Д0:` and `Д5:`, breaking action ordering
**Location:** `__parse_scenario` line 247, `__parse_coop_scenario` line 343.

```python
elif re.match(r"^Д\d:", line):
    actions.append(re.sub(r"^Д\d:\s*", "", line).strip())
```

If the LLM emits `Д0: ...` (rare but happens), it's appended *before* the legitimate `Д1: ...`, and the resulting `actions[:4]` returns four buttons in the wrong order — i.e. the button labels won't match the LLM's intent for the narrative phase.

**Fix:** `r"^Д[1-4]:"` and ideally key the parsed actions by their digit so order is determined by the prompt format, not the order of lines:

```python
parsed = {}
for line in text.splitlines():
    match = re.match(r"^Д([1-4]):\s*(.*)$", line.strip())
    if match:
        parsed[int(match.group(1))] = match.group(2).strip()
actions = [parsed.get(idx, "") for idx in range(1, 5)]
```

### 6. `handle_dnd_callback` doesn't acknowledge unknown callback data
**Location:** lines 563-575.

If `data` matches the pattern `^dnd_` but is neither `dnd_join` nor an action prefix (e.g. a stale callback after a code change), the function falls through without `query.answer()`. Telegram shows a loading spinner on the player's button for up to 30 seconds.

**Fix:** add `await query.answer()` as the default tail.

### 7. `BadRequest("Message is not modified")` is treated as a real error
**Location:** `__edit_safe` lines 992-1006 and other edit paths.

When two players click in immediate succession, the second edit can produce identical text (same player order, same `✅` distribution). Telegram returns `BadRequest: Message is not modified`. `__edit_safe`'s retry-without-parse-mode also fails with the same error. Other edit paths log a warning that's pure noise.

**Fix:**
```python
except BadRequest as error:
    if "not modified" in str(error).lower():
        return
    ...
```

### 8. `__active_chats` not released when the action-timeout job fires but the game was already popped
**Location:** `__expire_actions` lines 751-757.

```python
async def __expire_actions(context):
    chat_id = context.job.data
    game = __active_games.pop(chat_id, None)
    __action_timeout_jobs.pop(chat_id, None)
    if not game:
        return
    context.job_queue.run_once(__resolve_game_job, 0, data=(chat_id, game))
```

If the game was already popped (last action click won the race against the timer), this returns cleanly. But there's a different scenario: the game ran to completion through `__resolve_*_round` but the next-round job fails to **send** a new message (lines 942-945 in `__next_round_job`). At that point `__active_chats.discard(chat_id)` runs but the action-timeout job for the prior round is gone (already cancelled by `__handle_action` or `__expire_actions`). So this is fine. ✅

What is **not** fine: if `__edit_safe` in `__resolve_*_round` (lines 844, 910) fails when `is_final` is True, the chat is released BUT the message is left in its "Подводим итоги..." state. Same UX bug as #2 / #3: silent stuck game.

### 9. `boss_max_hp` formula gives ~80% win rate, not the documented 70%
**Location:** `__start_game_job` line 700, with documentation in `dnd.md:120`.

`boss_max_hp = random.randint(N*15, N*20)`. For N=3: range 45-60, mean 52.5. Mean party damage per round: each of 3 players rolls d20 (mean 10.5), so per round 31.5 damage; over 2 rounds, 63. The probability that two rounds of 3d20 sums each ≥ boss_max_hp is closer to ~80% than 70%. Either re-tune the constant or update the doc.

### 10. `game.history` does not capture the boss-HP trajectory for coop continuation prompts
**Location:** `__resolve_coop_round` line 850, `__generate_coop_round` lines 318-324.

The history dict stores `{scenario, narrative, results}`. The continuation prompt for round 2 of coop (line 320) re-reads `boss_hp` from the live `ActiveGame`, so this works. But the prompt only includes narratives, not the HP delta per round. If the LLM is told `"У босса осталось 22 из 50 HP"` and the prior narrative was already triumphant ("отряд сокрушает противника!"), the LLM has to reconcile a near-victorious narrative with a still-alive boss. This causes inconsistent narratives in practice.

**Fix (improvement, not bug):** include `damage_this_round` and HP-after in the history line for prompt context.

---

## Important improvements

### 11. Mode branching is duplicated in seven places — needs a strategy pattern
You asked specifically about this. The `if game.mode == ...` branches are:

| Location | Lines | Purpose |
|---|---|---|
| `__build_lobby_text` | 85-93 | Lobby header per mode |
| `__build_game_text` | 116-129 | Round header + boss HP line |
| `__start_game_job` | 695-704 | Scenario generator dispatch |
| `__resolve_game_job` | 772-775 | Resolver dispatch |
| `__resolve_standard_round` | 866-868, 896-900 | Heist phase names + result title (twice) |
| `__next_round_job` | 926-933 | Loading-text dispatch |
| `__generate_round` | 182-206, 213-231 | Mode-specific user prompt |

Heist phase names `{1: "Проникновение", 2: "Дело", 3: "Побег"}` are duplicated three times verbatim (lines 124, 866, 929). Adding a new mode (e.g. `/dnd_horror`) means touching 7 distinct places.

**Recommendation:** introduce a `ModeStrategy` registry. Sketch:

```python
@dataclass
class ModeStrategy:
    name: str                         # "pvp" | "coop" | "heist" | "adventure"
    lobby_header: Callable[[int], str]
    round_header: Callable[[int, int], str]
    loading_header: Callable[[int, int], str]
    result_title: Callable[[int, int, bool], str]
    system_prompt: str
    first_round_prompt: Callable[[int], str]
    continuation_prompt: Callable[[int, int, int, list[dict]], str]
    narrative_ending: Callable[[bool], str]
    has_boss: bool = False
    fallback_narrative: str = "Летописец выронил перо..."

MODE_STRATEGIES: dict[str, ModeStrategy] = {
    "pvp": ModeStrategy(...),
    "coop": ModeStrategy(..., has_boss=True),
    "heist": ModeStrategy(...),
}
```

This collapses `__resolve_coop_round` and `__resolve_standard_round` into a single function gated on `strategy.has_boss`, eliminates the heist phase-name duplication, and reduces the cost of adding a new mode from 7 sites to 1 strategy entry.

It's a real refactor (~150-200 lines moved), but the module is at the threshold where it pays off.

### 12. `__resolve_coop_round` and `__resolve_standard_round` share ~40% of their bodies
**Lines:** 778-854 vs 857-920.

Both compute `roll_lines` identically (817-821 vs 890-894). Both append to `game.history`, increment `round_number`, reset `choices`, and schedule next round (850-854 vs 916-920). Extract:

```python
def __finalize_round(game, narrative, player_results) -> None:
    game.history.append({"scenario": game.scenario, "narrative": narrative, "results": player_results})
    game.round_number += 1
    game.choices = {}

def __format_roll_lines(player_results) -> list[str]: ...
```

Shrinks each resolver by half.

### 13. No timeout on LLM calls
**Locations:** every `await llm.ainvoke(...)`.

If Groq stalls for 90 seconds, the round freezes — and the action-timer for the next round eventually fires while the message still says "Подводим итоги...". Wrap with `asyncio.wait_for(..., timeout=30)`:

```python
try:
    response = await asyncio.wait_for(
        llm.ainvoke([SystemMessage(...), HumanMessage(...)]),
        timeout=30,
    )
except asyncio.TimeoutError:
    raise  # caller handles as generation failure
```

The 45-second action timer assumes the resolve path completes in well under that window.

### 14. ChatGroq client constructed on every LLM call
**Lines:** 164-169, 271-276, 368-373, 447-452.

Four call sites instantiate `ChatGroq(...)` with identical config. Hoist a module-level factory or a single client:

```python
def __llm(max_tokens: int = 300) -> ChatGroq:
    return ChatGroq(model=DND_MODEL, api_key=config.GROQ_API_KEY,
                    temperature=0.95, max_tokens=max_tokens)
```

Also makes the client trivially mockable for tests.

### 15. History prompt formatting duplicated four times
**Lines:** 208-211, 314-317, 382-384, 461-463.

All four build "Раунд N: ..." blocks for LLM prompts in nearly identical ways. Extract `__format_history_lines(history, with_scenario: bool) -> str`.

### 16. Hard-coded fallback narratives drift apart
**Lines:** 815, 888.

`"Летописец выронил перо..."` appears in two places with slightly different text. With the strategy refactor (#11) these belong on the strategy.

### 17. Action button labels are not length-checked
**Location:** `__parse_scenario` returns whatever the LLM emits. Lines on Telegram inline buttons get awkwardly truncated past ~30 chars on mobile. The system prompt says "2-5 слов" but doesn't enforce it. Add a hard length cap (e.g. 40 chars) in `__parse_scenario`:

```python
actions = [action[:40] for action in actions]
```

### 18. `tuple` annotation for `__lobbies` is too loose
**Line 68:** `__lobbies: dict[int, tuple] = {}`. Use `tuple[LobbyState, int, str]` for type-checker visibility, or replace with a small dataclass `LobbyEntry`.

### 19. Dead `mode == "adventure"` branch
**Locations:** `__build_lobby_text` lines 91-93, `__build_game_text` lines 127-129.

The `else` branch corresponds to `mode == "adventure"`, but no entry point sets that mode — `cmd_dnd_pvp`/`coop`/`dnd3` set `pvp`/`coop`/`heist`. The branch is unreachable. Either wire up a `/dnd` adventure command or remove the branch (and the `mode: str = "adventure"` default in `ActiveGame` line 60).

### 20. `__start_game_job` builds an `ActiveGame` from a tuple it could have received whole
**Location:** line 633, 689, 718-730.

The job is scheduled with a positional tuple `(chat_id, message_id, players, max_rounds, mode)`. Inside the job, an `ActiveGame` is built. Cleaner: build the `ActiveGame` (without scenario/actions) at the call site and pass it as `data=game`. Avoids the boss-field zero-init at lines 691-692.

---

## Suggestions (style, minor)

### 21. Compile regexes once at module level
**Lines:** 247, 248, 343, 344. `re.match(r"^Д\d:", ...)` and `re.sub(r"^Д\d:\s*", "", ...)` are called per line per parse. Module-level `__ACTION_RE = re.compile(r"^Д([1-4]):\s*(.*)$")`.

### 22. `re.compile` is missing for `DND_CALLBACK_PATTERN`
**Line 35.** It's a string passed to `CallbackQueryHandler`, which compiles it once internally — this is fine and idiomatic for python-telegram-bot. Just confirming this is intentional.

### 23. Variable shadowing in `__parse_scenario`
**Lines:** 243-244.
```python
for line in text.splitlines():
    line = line.strip()
```
Shadowing the loop variable. Minor, but `stripped = line.strip()` is clearer.

### 24. Logging level inconsistency
**Lines:** 706, 814, 887 use `logger.error` for LLM failures (recoverable, fallback narrative kicks in). `logger.warning` is used for Telegram edit failures. LLM failures that fall through to a usable fallback are arguably warnings — reserve `error` for cases that kill the game.

### 25. `scenario, _, actions = await __generate_coop_round(...)` discards the boss name
**Line 949.** Re-uses `__generate_coop_round` for round 2+ but discards the boss name (it's fixed at round 1). Add a comment, or split into `__generate_coop_first_round` and `__generate_coop_continuation`.

### 26. `__build_game_keyboard` builds 2-wide rows manually
**Lines 141-150.** With Python 3.12+, `itertools.batched(actions, 2)` is cleaner. Stylistic.

### 27. Module-level mutable globals
The `__lobbies`, `__active_games`, `__active_chats`, `__lobby_timeout_jobs`, `__action_timeout_jobs` dicts are all mutable module state. Consider wrapping in a `DnDState` class to make testing and state-reset easier. Not critical for a small bot.

### 28. No unit tests for `__parse_scenario` / `__parse_coop_scenario`
These are pure functions with clear contracts and known fallbacks — ideal candidates for tests covering: empty input, missing СЦЕНАРИЙ, missing actions, extra unrelated lines, mixed-case prefix, action numbers outside 1-4, BOM-prefixed first line, leading whitespace.

### 29. Bot restart loses all in-flight state
All state is in-memory; restart wipes `__active_chats`, lobbies, and games. Live messages still show buttons that error out with "Лобби уже закрыто" / "Игра уже завершена". Acceptable for an in-memory bot but worth noting in `dnd.md` ("game state is volatile across bot restarts"). No code change needed unless persistence is in scope.

---

## Things done well

- **`__edit_safe`** (lines 992-1006) is a nice fallback pattern for narrative messages — the dual-attempt with parse_mode then plain is the right call.
- **Pop-then-check on resolve** (lines 675-676): `__active_games.pop(chat_id, None)` before scheduling resolve correctly handles the race between last-action-click and timer expiry. Only one resolve job runs.
- **Job-queue everywhere** for timers; no `asyncio.create_task` floating around. Consistent and inspectable.
- **Strict LLM prompt format** with explicit per-field markers (`СЦЕНАРИЙ:`, `Д1:` ...) and parser fallbacks — much more robust than free-form parsing.
- **History threading** into prompts gives narrative continuity without state explosion. The token cost stays bounded for ≤3 rounds.
- **Mode-specific narrative endings** (lines 387-410) are well-thought-out — pvp / heist-final / heist-mid / adventure-final / adventure-regular all distinct and appropriate.
- **Separation of `LobbyState` and `ActiveGame`** dataclasses keeps lobby and in-game concerns cleanly distinct.
- **Atomic boss-name capture** at round 1 only (line 949 discards subsequent boss-name parses) — correct, prevents the LLM from renaming the boss mid-fight.

---

## Suggested priority ordering

1. **#1** — lobby auto-start race (real `KeyError` under concurrent clicks).
2. **#2 / #3 / #4** — silent edit failures leave the chat with a stuck "Generating..." message; player has no recourse but to wait 5 minutes for lobby timeout (which won't fire because the game already started).
3. **#6 / #7** — callback hygiene and "not modified" handling (low effort, removes UX paper-cuts and log noise).
4. **#5** — tighten the action regex to `[1-4]` and key by digit.
5. **#13** — `asyncio.wait_for` around LLM calls.
6. **#11** — strategy registry refactor for mode branching (biggest maintainability win, prerequisite for adding more modes cheaply).
7. **#12 / #14 / #15 / #16** — extract shared resolve logic, module-level LLM client, shared history formatter, fallback narratives on the strategy.
8. **#28** — unit tests for the parse functions.
9. Remaining style cleanup (#19, #20, #21, #23, #24, #25, #26).
