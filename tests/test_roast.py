"""
Roast feature tests.

Covers four units in isolation:
  - pick_roast_cluster: pure numpy selection logic
  - pop_roast_target: shuffle-bag queue against a mocked DB
  - Roaster.generate: LLM + embedding path with all I/O mocked
  - Roaster.cmd_roast: Telegram command edge cases
"""

import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from src.commands.fun.roast import ROAST_ANCHORS, ROAST_HEADERS, Roaster, pick_roast_cluster
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
# pick_roast_cluster
# ---------------------------------------------------------------------------

class TestPickRoastCluster:
    def test_fewer_than_cluster_size_returns_all(self):
        facts = [("f1", _unit([1, 0, 0])), ("f2", _unit([0, 1, 0]))]
        result = pick_roast_cluster(facts, _unit([1, 0, 0]))
        assert set(result) == {"f1", "f2"}

    def test_exactly_cluster_size_returns_all(self):
        facts = [
            ("f1", _unit([1, 0, 0])),
            ("f2", _unit([0, 1, 0])),
            ("f3", _unit([0, 0, 1])),
        ]
        result = pick_roast_cluster(facts, _unit([1, 0, 0]))
        assert set(result) == {"f1", "f2", "f3"}

    def test_selects_tightest_cluster_closest_to_anchor(self):
        # A, B, C: close to anchor [1,0,0,0] and mutually similar.
        # D, E: far from the anchor and from A/B/C — no triplet containing
        # them can outscore (A, B, C) on average pairwise similarity.
        facts = [
            ("A", np.array([0.9, 0.1, 0.0, 0.0])),
            ("B", np.array([0.85, 0.15, 0.0, 0.0])),
            ("C", np.array([0.8, 0.2, 0.0, 0.0])),
            ("D", np.array([0.0, 0.9, 0.1, 0.0])),
            ("E", np.array([0.0, 0.1, 0.9, 0.0])),
        ]
        result = pick_roast_cluster(facts, np.array([1.0, 0.0, 0.0, 0.0]))
        assert set(result) == {"A", "B", "C"}

    def test_returns_exactly_cluster_size_when_more_facts(self):
        facts = [(f"f{idx}", _unit([float(idx + 1), 0.0])) for idx in range(8)]
        result = pick_roast_cluster(facts, _unit([1.0, 0.0]))
        assert len(result) == 3


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
    async def test_returns_header_text_anchor_key_tuple(self):
        mock_response = MagicMock(content="прожарка")
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        with patch("src.commands.fun.roast.get_facts_with_embeddings", AsyncMock(return_value=[])), \
             patch("src.commands.fun.roast.get_facts", AsyncMock(return_value=[])), \
             patch("src.commands.fun.roast.ChatGroq", return_value=mock_llm), \
             patch("src.commands.fun.roast.apply_language_correction", AsyncMock(return_value=mock_response)):
            header, text, anchor_key = await Roaster().generate(CHAT_ID, USER_ID, "vasya")
        assert header in ROAST_HEADERS
        assert text == "прожарка"
        assert anchor_key in ROAST_ANCHORS

    async def test_uses_embedding_facts_when_available(self):
        fake_emb = np.ones(4)
        mock_response = MagicMock(content="эмбеддинг-прожарка")
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        with patch("src.commands.fun.roast.get_facts_with_embeddings",
                   AsyncMock(return_value=[("играет ночью", fake_emb)])), \
             patch("src.commands.fun.roast.embedder.embed", AsyncMock(return_value=list(fake_emb))), \
             patch("src.commands.fun.roast.ChatGroq", return_value=mock_llm), \
             patch("src.commands.fun.roast.apply_language_correction", AsyncMock(return_value=mock_response)):
            header, text, anchor_key = await Roaster().generate(CHAT_ID, USER_ID, "vasya")
        assert header in ROAST_HEADERS
        assert text == "эмбеддинг-прожарка"
        assert anchor_key in ROAST_ANCHORS

    async def test_silent_member_gets_silence_fallback_prompt(self):
        """When a user has no facts, the prompt must call out their silence."""
        mock_response = MagicMock(content="молчун")
        captured: list = []
        mock_llm = AsyncMock()

        async def capture_invoke(messages):
            captured.extend(messages)
            return mock_response

        mock_llm.ainvoke = capture_invoke
        with patch("src.commands.fun.roast.get_facts_with_embeddings", AsyncMock(return_value=[])), \
             patch("src.commands.fun.roast.get_facts", AsyncMock(return_value=[])), \
             patch("src.commands.fun.roast.ChatGroq", return_value=mock_llm), \
             patch("src.commands.fun.roast.apply_language_correction", AsyncMock(return_value=mock_response)):
            await Roaster().generate(CHAT_ID, USER_ID, "silentuser")
        user_prompt = captured[1].content
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
