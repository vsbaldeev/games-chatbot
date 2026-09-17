"""llm_system_prompts.hash_prompt is pure and worth pinning directly —
ensure()/cleanup_unreferenced() are raw SQL, verified manually per this
repo's store-layer convention (see tests/unit/store/test_unified_messages.py)."""

from src.store.llm_system_prompts import hash_prompt


class TestHashPrompt:
    def test_same_content_hashes_the_same(self):
        assert hash_prompt("Ты — Жора.") == hash_prompt("Ты — Жора.")

    def test_different_content_hashes_differently(self):
        assert hash_prompt("Ты — Жора.") != hash_prompt("Ты — не Жора.")

    def test_returns_sha256_hex_digest(self):
        result = hash_prompt("test")
        assert len(result) == 64
        assert all(char in "0123456789abcdef" for char in result)
