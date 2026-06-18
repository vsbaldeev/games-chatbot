"""
Roast feature tests.

Covers three units in isolation:
  - pick_roast_mode: recent-mode-avoiding angle picker
  - pop_roast_target: shuffle-bag queue against a mocked DB
  - Roaster.generate: LLM path with all I/O mocked (used by the offense auto-roast)
"""

from unittest.mock import AsyncMock, MagicMock, patch

from src.commands.fun.roast import (
    ROAST_HEADERS,
    ROAST_MODES,
    Roaster,
    pick_roast_mode,
)
from src.store.roast_store import pop_roast_target

CHAT_ID = 1000
USER_ID = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
             patch("src.commands.fun.roast.get_facts", AsyncMock(return_value=[])), \
             patch("src.commands.fun.roast.roast_agent.invoke_roast", AsyncMock(return_value="прожарка")):
            header, text, mode = await Roaster().generate(CHAT_ID, USER_ID, "vasya")
        assert header in ROAST_HEADERS
        assert text == "прожарка"
        assert mode in ROAST_MODES

    async def test_passes_all_facts_to_the_model(self):
        """Every stored fact must reach the prompt, with no pre-selection window."""
        mock_invoke = AsyncMock(return_value="прожарка")
        facts = ["гоняет на мотоцикле 200 км/ч", "боится быстрой езды", "живёт в Москве"]
        with patch("src.commands.fun.roast.get_recent_modes", AsyncMock(return_value=[])), \
             patch("src.commands.fun.roast.pick_roast_mode", return_value="shame"), \
             patch("src.commands.fun.roast.get_facts", AsyncMock(return_value=facts)), \
             patch("src.commands.fun.roast.roast_agent.invoke_roast", mock_invoke):
            await Roaster().generate(CHAT_ID, USER_ID, "vasya")
        user_prompt = mock_invoke.call_args[0][0]
        assert all(fact in user_prompt for fact in facts)

    async def test_silent_member_gets_silence_fallback_prompt(self):
        """When a user has no facts, the prompt must call out their silence."""
        mock_invoke = AsyncMock(return_value="молчун")
        with patch("src.commands.fun.roast.get_recent_modes", AsyncMock(return_value=[])), \
             patch("src.commands.fun.roast.get_facts", AsyncMock(return_value=[])), \
             patch("src.commands.fun.roast.roast_agent.invoke_roast", mock_invoke):
            await Roaster().generate(CHAT_ID, USER_ID, "silentuser")
        user_prompt = mock_invoke.call_args[0][0]
        assert "silentuser" in user_prompt
        assert "ничего не пишет" in user_prompt
