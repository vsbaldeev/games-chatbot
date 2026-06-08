"""
Roast feature tests.

Covers five units in isolation:
  - pick_roast_facts: stochastic numpy selection logic
  - pick_roast_mode: recent-mode-avoiding angle picker
  - pop_roast_target: shuffle-bag queue against a mocked DB
  - Roaster.generate: LLM + embedding path with all I/O mocked
  - Roaster.cmd_roast: Telegram command edge cases
"""

import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from src.commands.fun.roast import (
    ROAST_HEADERS,
    ROAST_MODES,
    SELECTION_SIZE,
    Roaster,
    pick_roast_facts,
    pick_roast_mode,
)
from src.store.roast_store import pop_roast_target

CHAT_ID = 1000
USER_ID = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(vec: list) -> np.ndarray:
    arr = np.array(vec, dtype=float)
    return arr / np.linalg.norm(arr)


def _make_db_conn(fetch_return: list | None = None) -> AsyncMock:
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=fetch_return or [])
    conn.execute = AsyncMock()
    conn.executemany = AsyncMock()
    txn = MagicMock()
    txn.__aenter__ = AsyncMock(return_value=None)
    txn.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=txn)
    return conn


def _db_acquire(conn: AsyncMock) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# pick_roast_facts
# ---------------------------------------------------------------------------

class TestPickRoastFacts:
    def test_fewer_than_selection_size_returns_all(self):
        facts = [("f1", _unit([1, 0, 0])), ("f2", _unit([0, 1, 0]))]
        result = pick_roast_facts(facts, _unit([1, 0, 0]))
        assert set(result) == {"f1", "f2"}

    def test_exactly_selection_size_returns_all(self):
        facts = [
            ("f1", _unit([1, 0, 0])),
            ("f2", _unit([0, 1, 0])),
            ("f3", _unit([0, 0, 1])),
        ]
        result = pick_roast_facts(facts, _unit([1, 0, 0]))
        assert set(result) == {"f1", "f2", "f3"}

    def test_reaches_for_varied_facts_over_near_duplicates(self):
        # T1, T2, T3 are near-duplicate facts pointing the same way (a dense
        # "fandom" cluster); V is a different direction but still embarrassing.
        # The redundancy penalty must make the distinct fact V a frequent pick
        # rather than stacking all three near-duplicates, so a single cluster
        # cannot reliably fill every slot.
        anchor = _unit([1.0, 0.0, 0.0])
        facts = [
            ("T1", _unit([0.99, 0.10, 0.0])),
            ("T2", _unit([0.98, 0.14, 0.0])),
            ("T3", _unit([0.97, 0.17, 0.0])),
            ("V", _unit([0.6, 0.8, 0.0])),
        ]
        results = [tuple(pick_roast_facts(facts, anchor)) for _ in range(200)]
        assert all(len(set(result)) == SELECTION_SIZE for result in results)
        # V is distinct, so the diversity penalty makes it a common pick.
        assert sum("V" in result for result in results) > 100

    def test_selection_varies_across_runs(self):
        # Stochastic selection is the whole point: the same candidates must not
        # always yield the same set, otherwise repeat roasts stay stale.
        anchor = _unit([1.0, 0.0])
        facts = [(f"f{idx}", _unit([float(idx + 1), 1.0])) for idx in range(8)]
        seen = {tuple(sorted(pick_roast_facts(facts, anchor))) for _ in range(200)}
        assert len(seen) > 1

    def test_returns_exactly_selection_size_when_more_facts(self):
        facts = [(f"f{idx}", _unit([float(idx + 1), 0.0])) for idx in range(8)]
        result = pick_roast_facts(facts, _unit([1.0, 0.0]))
        assert len(result) == SELECTION_SIZE


# ---------------------------------------------------------------------------
# pick_roast_mode
# ---------------------------------------------------------------------------

class TestPickRoastMode:
    def test_excludes_recent_mode_when_alternatives_exist(self):
        excluded = ROAST_MODES[0]
        picks = {pick_roast_mode([excluded]) for _ in range(200)}
        assert excluded not in picks
        assert picks.issubset(set(ROAST_MODES))

    def test_falls_back_to_full_set_when_all_modes_recent(self):
        # If every mode was used recently, exclusion would leave nothing — the
        # picker must still return a valid mode rather than failing.
        result = pick_roast_mode(list(ROAST_MODES))
        assert result in ROAST_MODES

    def test_empty_history_allows_any_mode(self):
        picks = {pick_roast_mode([]) for _ in range(200)}
        assert picks == set(ROAST_MODES)


# ---------------------------------------------------------------------------
# pop_roast_target
# ---------------------------------------------------------------------------

class TestPopRoastTarget:
    async def test_picks_from_remaining_queue(self):
        members = [(1, "alice"), (2, "bob"), (3, "carol")]
        conn = _make_db_conn(fetch_return=[{"user_id": 2}])
        with patch("src.store.roast_store.database.acquire", return_value=_db_acquire(conn)):
            target_id, target_username = await pop_roast_target(CHAT_ID, members)
        assert target_id == 2
        assert target_username == "bob"

    async def test_refills_and_picks_when_queue_empty(self):
        members = [(1, "alice"), (2, "bob")]
        conn = _make_db_conn(fetch_return=[])
        with patch("src.store.roast_store.database.acquire", return_value=_db_acquire(conn)):
            target_id, target_username = await pop_roast_target(CHAT_ID, members)
        assert target_id in {1, 2}
        assert target_username in {"alice", "bob"}
        conn.executemany.assert_called_once()

    async def test_deletes_picked_user_from_queue(self):
        conn = _make_db_conn(fetch_return=[{"user_id": 1}])
        with patch("src.store.roast_store.database.acquire", return_value=_db_acquire(conn)):
            await pop_roast_target(CHAT_ID, [(1, "alice")])
        conn.execute.assert_called_once_with(
            "DELETE FROM roast_queue WHERE chat_id = $1 AND user_id = $2",
            CHAT_ID, 1,
        )


# ---------------------------------------------------------------------------
# Roaster.generate
# ---------------------------------------------------------------------------

class TestRoasterGenerate:
    async def test_returns_header_text_mode_tuple(self):
        with patch("src.commands.fun.roast.get_recent_modes", AsyncMock(return_value=[])), \
             patch("src.commands.fun.roast.get_facts_with_embeddings", AsyncMock(return_value=[])), \
             patch("src.commands.fun.roast.get_facts", AsyncMock(return_value=[])), \
             patch("src.commands.fun.roast.roast_agent.invoke_roast", AsyncMock(return_value="прожарка")):
            header, text, mode = await Roaster().generate(CHAT_ID, USER_ID, "vasya")
        assert header in ROAST_HEADERS
        assert text == "прожарка"
        assert mode in ROAST_MODES

    async def test_uses_embedding_facts_when_available(self):
        fake_emb = np.ones(4)
        embed_mock = AsyncMock(return_value=list(fake_emb))
        with patch("src.commands.fun.roast.get_recent_modes", AsyncMock(return_value=[])), \
             patch("src.commands.fun.roast.pick_roast_mode", return_value="shame"), \
             patch("src.commands.fun.roast.get_facts_with_embeddings",
                   AsyncMock(return_value=[("играет ночью", fake_emb)])), \
             patch("src.commands.fun.roast.embedder.embed", embed_mock), \
             patch("src.commands.fun.roast.roast_agent.invoke_roast", AsyncMock(return_value="эмбеддинг-прожарка")):
            header, text, mode = await Roaster().generate(CHAT_ID, USER_ID, "vasya")
        assert header in ROAST_HEADERS
        assert text == "эмбеддинг-прожарка"
        assert mode == "shame"
        embed_mock.assert_awaited_once()

    async def test_silent_member_gets_silence_fallback_prompt(self):
        """When a user has no facts, the prompt must call out their silence."""
        mock_invoke = AsyncMock(return_value="молчун")
        with patch("src.commands.fun.roast.get_recent_modes", AsyncMock(return_value=[])), \
             patch("src.commands.fun.roast.get_facts_with_embeddings", AsyncMock(return_value=[])), \
             patch("src.commands.fun.roast.get_facts", AsyncMock(return_value=[])), \
             patch("src.commands.fun.roast.roast_agent.invoke_roast", mock_invoke):
            await Roaster().generate(CHAT_ID, USER_ID, "silentuser")
        user_prompt = mock_invoke.call_args[0][0]
        assert "silentuser" in user_prompt
        assert "ничего не пишет" in user_prompt


# ---------------------------------------------------------------------------
# Roaster.cmd_roast
# ---------------------------------------------------------------------------

class TestCmdRoast:
    def _make_update(self) -> MagicMock:
        update = MagicMock()
        update.message.reply_text = AsyncMock(return_value=MagicMock(message_id=999))
        update.message.chat.send_action = AsyncMock()
        update.effective_chat.id = CHAT_ID
        return update

    def _make_context(self, bot_id: int = 999) -> MagicMock:
        context = MagicMock()
        context.bot.id = bot_id
        return context

    async def test_no_members_sends_empty_chat_message(self):
        update = self._make_update()
        with patch("src.commands.fun.roast.achievements.get_chat_members", AsyncMock(return_value=[])):
            await Roaster().cmd_roast(update, self._make_context())
        update.message.reply_text.assert_called_once_with(
            "В базе нет участников. Пусть сначала кто-нибудь напишет в чат."
        )

    async def test_generate_exception_sends_error_message(self):
        roaster = Roaster()
        update = self._make_update()
        with patch("src.commands.fun.roast.achievements.get_chat_members",
                   AsyncMock(return_value=[(1, "alice")])), \
             patch("src.commands.fun.roast.pop_roast_target", AsyncMock(return_value=(1, "alice"))), \
             patch.object(roaster, "generate", AsyncMock(side_effect=RuntimeError("groq down"))):
            await roaster.cmd_roast(update, self._make_context())
        update.message.reply_text.assert_called_once_with(
            "Прожарка не задалась. Groq на перекуре — попробуй позже."
        )
