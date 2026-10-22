"""Reply text cleanup tests."""

from src.pipeline.respond.reply.reply_text import strip_speaker_prefix, strip_writing_tics


class TestStripSpeakerPrefix:
    def test_strips_own_label(self):
        assert strip_speaker_prefix("Ты (бот): Ну да, еблан.") == "Ну да, еблан."

    def test_strips_username_label(self):
        assert strip_speaker_prefix("@alice: сам такой") == "сам такой"

    def test_strips_username_label_with_reply_arrow(self):
        text = "@alice (↳ Ты (бот)): сам такой"
        assert strip_speaker_prefix(text) == "сам такой"

    def test_only_strips_leading_label_not_ones_mid_text(self):
        text = "он написал @alice: и заткнулся"
        assert strip_speaker_prefix(text) == text

    def test_leaves_text_without_a_label_unchanged(self):
        assert strip_speaker_prefix("Обычный ответ без метки.") == "Обычный ответ без метки."


class TestStripWritingTics:
    def test_removes_single_word_scare_quotes(self):
        assert strip_writing_tics("Ну да, «шедевр».") == "Ну да, шедевр."

    def test_keeps_multi_word_quotes(self):
        text = 'Он сказал «отвали от меня».'
        assert strip_writing_tics(text) == text

    def test_normalizes_non_breaking_hyphen(self):
        assert strip_writing_tics("чурка‑титуле") == "чурка-титуле"

    def test_normalizes_en_dash(self):
        assert strip_writing_tics("да так – по делам") == "да так - по делам"

    def test_strips_trailing_emoji(self):
        assert strip_writing_tics("Ну и ладно 🙄") == "Ну и ладно"

    def test_strips_trailing_emoji_with_zwj_sequence(self):
        assert strip_writing_tics("Не еблан 🤷‍♂️") == "Не еблан"

    def test_leaves_inline_emoji_alone(self):
        text = "Ну 🙄 и ладно"
        assert strip_writing_tics(text) == text

    def test_leaves_clean_text_unchanged(self):
        text = "Обычный ответ без всяких меток."
        assert strip_writing_tics(text) == text
