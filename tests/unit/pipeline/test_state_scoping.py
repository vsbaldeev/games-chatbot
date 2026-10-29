"""Stage state schemas: each stage's source names only keys its schema declares.

LangGraph hides undeclared keys from a subgraph and silently drops undeclared keys a
node returns, so a stray ``state.get("x")`` would read ``None`` forever and a stray
``{"x": ...}`` update would vanish. The check is static: every string that is a ``BotState``
key and is used as a dict key, a subscript or a ``.get()`` argument in a stage's modules
must belong to that stage's schema.
"""

import ast
from pathlib import Path

import pytest

from src.pipeline import state as state_module
from src.pipeline.state import BotState, IngestState, RespondState

PIPELINE_ROOT = Path(state_module.__file__).parent

STAGE_SCHEMAS = {
    "ingest": (IngestState, ["ingest/graph.py", "ingest/media", "ingest/filter"]),
    "respond": (
        RespondState,
        [
            "respond/graph.py",
            "respond/guard_node.py",
            "respond/context_builder.py",
            "respond/worker_node.py",
            "respond/reply",
        ],
    ),
}


def source_files(relative_paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for relative in relative_paths:
        path = PIPELINE_ROOT / relative
        files.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    return files


def state_keys_used(path: Path) -> set[str]:
    """String constants in ``path`` that are used as a dict key, subscript or ``.get()`` argument."""
    used: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        candidates: list[ast.expr | None] = []
        if isinstance(node, ast.Dict):
            candidates = list(node.keys)
        elif isinstance(node, ast.Subscript):
            candidates = [node.slice]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            candidates = node.args[:1]
        used.update(
            candidate.value
            for candidate in candidates
            if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str)
        )
    return used & set(BotState.__annotations__)


@pytest.mark.parametrize("stage", STAGE_SCHEMAS)
def test_stage_sources_only_use_keys_of_their_schema(stage):
    schema, relative_paths = STAGE_SCHEMAS[stage]
    declared = set(schema.__annotations__)
    strays = {
        str(path.relative_to(PIPELINE_ROOT)): sorted(state_keys_used(path) - declared)
        for path in source_files(relative_paths)
        if state_keys_used(path) - declared
    }
    assert strays == {}


def test_bot_state_is_exactly_the_union_of_the_stage_schemas_and_parent_keys():
    stage_keys = set(IngestState.__annotations__) | set(RespondState.__annotations__)
    assert set(BotState.__annotations__) - stage_keys == {"link_cap_remaining", "link_message_is_bare"}
