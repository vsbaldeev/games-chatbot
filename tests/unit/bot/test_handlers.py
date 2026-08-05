"""
Part 1 — retirement of the command-driven roast surface.

The /roast command and the weekly scheduled roast are removed in favour of
autonomous humor, while the offense auto-roast generation path
(``Roaster.generate`` / ``generate_roast_text``) is preserved because the
offense clap-back still uses it.
"""

import importlib
from unittest.mock import MagicMock

from telegram.ext import CommandHandler

from src.bot.handlers import CommandHandlerManager


def registered_command_names() -> set[str]:
    """Collect every command name registered by ``CommandHandlerManager``.

    Returns:
        The union of command strings across all registered ``CommandHandler``s.
    """
    recorded: set[str] = set()

    def capture(handler, *args, **kwargs) -> None:
        if isinstance(handler, CommandHandler):
            recorded.update(handler.commands)

    app = MagicMock()
    app.add_handler.side_effect = capture
    CommandHandlerManager().add_handlers(app)
    return recorded


class TestRoastCommandRemoved:
    def test_roast_command_not_registered(self):
        assert "roast" not in registered_command_names()

    def test_core_commands_still_registered(self):
        names = registered_command_names()
        assert {"start", "help", "meme", "duel"}.issubset(names)


class TestWeeklyRoastJobRemoved:
    def test_jobs_module_has_no_roast_job_manager(self):
        jobs = importlib.import_module("src.bot.jobs")
        assert not hasattr(jobs, "RoastJobManager")

    def test_weekly_roast_job_not_imported(self):
        jobs = importlib.import_module("src.bot.jobs")
        assert not hasattr(jobs, "weekly_roast_job")


class TestOffenseRoastPathPreserved:
    def test_generate_roast_text_still_exported(self):
        roast = importlib.import_module("src.commands.fun.roast")
        assert hasattr(roast, "generate_roast_text")
        assert hasattr(roast.Roaster, "generate")

    def test_cmd_roast_removed(self):
        roast = importlib.import_module("src.commands.fun.roast")
        assert not hasattr(roast, "cmd_roast")
        assert not hasattr(roast.Roaster, "cmd_roast")
