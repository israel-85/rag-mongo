import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import config

REPO_ROOT = Path(__file__).resolve().parent.parent


def _example_secret_names() -> list[str]:
    """Top-level names assigned in key_param.example.py, in file order.

    Parsed rather than imported: the example file is a template, and importing it
    would shadow the real key_param that conftest stubs.
    """
    tree = ast.parse((REPO_ROOT / "key_param.example.py").read_text())
    return [
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    ]


def _key_param(**overrides) -> SimpleNamespace:
    values: dict[str, object] = {name: "set" for name in config.REQUIRED_SECRETS}
    values.update(overrides)
    return SimpleNamespace(**values)


class TestMissingSecrets:
    def test_a_complete_module_is_missing_nothing(self):
        assert config.missing_secrets(_key_param()) == []

    def test_reports_an_absent_attribute(self):
        module = _key_param()
        del module.VOYAGE_API_KEY

        assert config.missing_secrets(module) == ["VOYAGE_API_KEY"]

    def test_reports_a_none_value(self):
        assert config.missing_secrets(_key_param(LLM_MODEL=None)) == ["LLM_MODEL"]

    def test_reports_an_empty_string(self):
        assert config.missing_secrets(_key_param(MONGODB_URI="")) == ["MONGODB_URI"]

    def test_reports_a_whitespace_only_string(self):
        """A copy-paste that left a blank line is still unset."""
        assert config.missing_secrets(_key_param(LLM_BASE_URL="   ")) == ["LLM_BASE_URL"]

    def test_reports_every_missing_name_at_once(self):
        """One run should tell you everything to fix, not one name per attempt."""
        module = _key_param(MONGODB_URI="", LLM_API_KEY=None)

        assert config.missing_secrets(module) == ["MONGODB_URI", "LLM_API_KEY"]

    def test_both_halves_need_the_llm_settings(self):
        """Ingestion tags pages with the LLM; retrieval generates the answer with it."""
        assert set(config.REQUIRED_SECRETS) == {
            "MONGODB_URI",
            "VOYAGE_API_KEY",
            "LLM_API_KEY",
            "LLM_BASE_URL",
            "LLM_MODEL",
        }


class TestDocumentationStaysInSync:
    """The template, the validator, and the docs are three copies of one list.

    Drift between them is silent and user-facing: a secret added to the example but
    not to REQUIRED_SECRETS is never validated, and one documented but not templated
    sends people hunting for a field that does not exist.
    """

    def test_the_example_file_defines_exactly_the_required_secrets(self):
        assert set(_example_secret_names()) == set(config.REQUIRED_SECRETS)

    def test_the_example_file_lists_them_in_the_validated_order(self):
        """Same order keeps the file, the error message, and the docs table readable."""
        assert _example_secret_names() == list(config.REQUIRED_SECRETS)

    def test_the_claude_md_secrets_table_covers_every_required_secret(self):
        table = (REPO_ROOT / "CLAUDE.md").read_text()
        documented = set(re.findall(r"^\| `([A-Z_]+)` \|", table, re.MULTILINE))

        assert set(config.REQUIRED_SECRETS) <= documented


class TestRequireSecrets:
    def test_passes_silently_when_everything_is_set(self):
        assert config.require_secrets(_key_param()) is None

    def test_exits_naming_the_missing_settings(self):
        """The point is a readable line at startup, not a driver traceback 30s in."""
        with pytest.raises(SystemExit) as exit_info:
            config.require_secrets(_key_param(MONGODB_URI=""))

        assert "MONGODB_URI" in str(exit_info.value)
        assert "key_param.example.py" in str(exit_info.value)
