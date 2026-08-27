"""Daily yt-dlp self-update job tests.

The install must request the same extras as requirements.txt/entrypoint.sh
(curl-cffi + default) — a bare "yt-dlp" install here would upgrade past its
matching yt-dlp-ejs (JS challenge-solver scripts, exact-pinned by yt-dlp to
its own version) and curl-cffi (Instagram extraction), silently breaking
both the moment versions drift apart.
"""

from unittest.mock import AsyncMock, patch

from src.jobs.ytdlp_update import (
    YTDLP_PACKAGE_SPEC,
    newer_version_available,
    ytdlp_update_job,
)

ISDIR_TARGET = "src.jobs.ytdlp_update.os.path.isdir"
VERSION_TARGET = "src.jobs.ytdlp_update.importlib.metadata.version"
RUN_PIP_TARGET = "src.jobs.ytdlp_update.run_pip"
KILL_TARGET = "src.jobs.ytdlp_update.os.kill"


class TestNewerVersionAvailable:
    def test_detects_a_different_version_in_dry_run_output(self):
        output = "Would install yt-dlp-2026.8.20 yt-dlp-ejs-0.8.1\n"
        assert newer_version_available(output, "2026.8.19") is True

    def test_current_version_is_not_newer(self):
        output = "Would install yt-dlp-2026.8.19 yt-dlp-ejs-0.8.0\n"
        assert newer_version_available(output, "2026.8.19") is False

    def test_no_would_install_line_is_not_newer(self):
        assert newer_version_available("Requirement already satisfied\n", "2026.8.19") is False


class TestYtdlpUpdateJobPackageSpec:
    async def test_dry_run_and_install_both_use_the_full_package_spec(self):
        run_pip = AsyncMock(side_effect=[
            (0, "Would install yt-dlp-2026.8.20\n"),
            (0, "Successfully installed yt-dlp-2026.8.20\n"),
        ])
        with patch(ISDIR_TARGET, return_value=True), \
             patch(VERSION_TARGET, return_value="2026.8.19"), \
             patch(RUN_PIP_TARGET, run_pip), \
             patch(KILL_TARGET):
            await ytdlp_update_job(None)
        dry_run_args = run_pip.await_args_list[0].args
        install_args = run_pip.await_args_list[1].args
        assert YTDLP_PACKAGE_SPEC in dry_run_args
        assert YTDLP_PACKAGE_SPEC in install_args

    async def test_no_newer_version_skips_install(self):
        run_pip = AsyncMock(return_value=(0, "Requirement already satisfied\n"))
        with patch(ISDIR_TARGET, return_value=True), \
             patch(VERSION_TARGET, return_value="2026.8.19"), \
             patch(RUN_PIP_TARGET, run_pip), \
             patch(KILL_TARGET) as mock_kill:
            await ytdlp_update_job(None)
        assert run_pip.await_count == 1
        mock_kill.assert_not_called()

    async def test_missing_runtime_deps_skips_entirely(self):
        run_pip = AsyncMock()
        with patch(ISDIR_TARGET, return_value=False), patch(RUN_PIP_TARGET, run_pip):
            await ytdlp_update_job(None)
        run_pip.assert_not_awaited()

    async def test_newer_version_installed_restarts_via_sigterm(self):
        run_pip = AsyncMock(side_effect=[
            (0, "Would install yt-dlp-2026.8.20\n"),
            (0, "Successfully installed yt-dlp-2026.8.20\n"),
        ])
        with patch(ISDIR_TARGET, return_value=True), \
             patch(VERSION_TARGET, return_value="2026.8.19"), \
             patch(RUN_PIP_TARGET, run_pip), \
             patch(KILL_TARGET) as mock_kill:
            await ytdlp_update_job(None)
        mock_kill.assert_called_once()
