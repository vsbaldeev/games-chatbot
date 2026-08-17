"""Tests for the opaque user_id <-> anon-key mapping shared by roles and group_profile."""

from src.utils.anon_map import anonymise


class TestAnonymise:
    def test_maps_each_user_id_to_a_positional_anon_key(self):
        uid_to_anon, anon_to_uid = anonymise([10, 20, 30])
        assert uid_to_anon == {10: "user_0", 20: "user_1", 30: "user_2"}

    def test_anon_to_uid_is_the_exact_inverse(self):
        uid_to_anon, anon_to_uid = anonymise([10, 20, 30])
        for user_id, anon in uid_to_anon.items():
            assert anon_to_uid[anon] == user_id

    def test_empty_input_returns_empty_maps(self):
        assert anonymise([]) == ({}, {})
