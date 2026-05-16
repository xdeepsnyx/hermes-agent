"""Tests for the tool_output_reducer plugin.

Covers the bundled plugin at ``plugins/tool_output_reducer/``:

  * ``rules.ansi.strip_ansi`` — CSI/OSC/two-char escape sequence removal
  * ``rules.dedupe.collapse_adjacent_duplicates`` — adjacent-line collapse
  * ``reducer.reduce_tool_output`` — dispatch, thresholds, error/allowlist
    passthrough, rule-chain composition, fail-open on rule error
  * Plugin ``__init__.register`` — hook registration + env-var disable
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    """Redirect HERMES_HOME so log writes don't pollute the real ~/.hermes/."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    yield hermes_home


def _plugin_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "plugins" / "tool_output_reducer"


def _load_module(rel_path: str, name: str):
    """Load a module from the plugin tree under a given test-only module name."""
    spec = importlib.util.spec_from_file_location(name, _plugin_dir() / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_reducer():
    """Load reducer.py with its `from .rules import ansi, dedupe` working.

    Builds a fake `hermes_plugins.tool_output_reducer` package namespace
    so relative imports resolve, mirroring the disk-cleanup test loader.
    """
    plugin_dir = _plugin_dir()
    pkg_name = "hermes_plugins.tool_output_reducer"

    # Ensure parent namespace package
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns

    # Package the plugin itself
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(plugin_dir)]
    sys.modules[pkg_name] = pkg

    # Subpackage `.rules`
    rules_pkg_name = f"{pkg_name}.rules"
    rules_pkg = types.ModuleType(rules_pkg_name)
    rules_pkg.__path__ = [str(plugin_dir / "rules")]
    sys.modules[rules_pkg_name] = rules_pkg

    # Load .rules.ansi and .rules.dedupe so reducer.py's import sees them
    ansi_spec = importlib.util.spec_from_file_location(
        f"{rules_pkg_name}.ansi", plugin_dir / "rules" / "ansi.py"
    )
    ansi_mod = importlib.util.module_from_spec(ansi_spec)
    ansi_spec.loader.exec_module(ansi_mod)
    sys.modules[f"{rules_pkg_name}.ansi"] = ansi_mod
    rules_pkg.ansi = ansi_mod

    dedupe_spec = importlib.util.spec_from_file_location(
        f"{rules_pkg_name}.dedupe", plugin_dir / "rules" / "dedupe.py"
    )
    dedupe_mod = importlib.util.module_from_spec(dedupe_spec)
    dedupe_spec.loader.exec_module(dedupe_mod)
    sys.modules[f"{rules_pkg_name}.dedupe"] = dedupe_mod
    rules_pkg.dedupe = dedupe_mod

    # Finally load reducer.py
    reducer_spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.reducer",
        plugin_dir / "reducer.py",
    )
    reducer_mod = importlib.util.module_from_spec(reducer_spec)
    reducer_mod.__package__ = pkg_name
    sys.modules[f"{pkg_name}.reducer"] = reducer_mod
    reducer_spec.loader.exec_module(reducer_mod)
    return reducer_mod


def _load_plugin_init():
    """Load the plugin's __init__.py so register() can be called."""
    plugin_dir = _plugin_dir()
    pkg_name = "hermes_plugins.tool_output_reducer"
    # Reuse _load_reducer's package setup
    _load_reducer()
    init_spec = importlib.util.spec_from_file_location(
        pkg_name,
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    init_mod = importlib.util.module_from_spec(init_spec)
    init_mod.__package__ = pkg_name
    init_mod.__path__ = [str(plugin_dir)]
    sys.modules[pkg_name] = init_mod
    init_spec.loader.exec_module(init_mod)
    return init_mod


# ---------------------------------------------------------------------------
# rules.ansi
# ---------------------------------------------------------------------------

class TestAnsiStrip:
    def test_strips_csi_color_codes(self):
        ansi = _load_module("rules/ansi.py", "_t_ansi_csi")
        assert ansi.strip_ansi("\x1b[32mOK\x1b[0m") == "OK"

    def test_strips_csi_cursor_movement(self):
        ansi = _load_module("rules/ansi.py", "_t_ansi_cursor")
        assert ansi.strip_ansi("up\x1b[2Adown") == "updown"

    def test_strips_osc_with_bel_terminator(self):
        ansi = _load_module("rules/ansi.py", "_t_ansi_osc_bel")
        assert ansi.strip_ansi("\x1b]0;title\x07hello") == "hello"

    def test_strips_osc_with_esc_terminator(self):
        ansi = _load_module("rules/ansi.py", "_t_ansi_osc_esc")
        assert ansi.strip_ansi("\x1b]8;;url\x1b\\link") == "link"

    def test_passes_clean_text_unchanged(self):
        ansi = _load_module("rules/ansi.py", "_t_ansi_clean")
        clean = "no escape sequences here, just\nplain text with newlines"
        assert ansi.strip_ansi(clean) == clean

    def test_strips_real_world_git_log_output(self):
        ansi = _load_module("rules/ansi.py", "_t_ansi_realworld")
        sample = "\x1b[33mcommit abc123\x1b[m\nAuthor: x\n\n    \x1b[32m+added\x1b[m line"
        out = ansi.strip_ansi(sample)
        assert "\x1b" not in out
        assert "commit abc123" in out
        assert "+added" in out


# ---------------------------------------------------------------------------
# rules.dedupe
# ---------------------------------------------------------------------------

class TestDedupe:
    def test_collapses_run_of_identical_lines(self):
        dedupe = _load_module("rules/dedupe.py", "_t_dedupe_run")
        text = "a\nb\nb\nb\nb\nb\nc"
        out = dedupe.collapse_adjacent_duplicates(text)
        assert out == "a\nb\n  (... 4 more identical lines ...)\nc"

    def test_preserves_non_adjacent_duplicates(self):
        dedupe = _load_module("rules/dedupe.py", "_t_dedupe_non_adj")
        text = "a\nb\na\nb"
        assert dedupe.collapse_adjacent_duplicates(text) == text

    def test_passes_unique_lines_unchanged(self):
        dedupe = _load_module("rules/dedupe.py", "_t_dedupe_unique")
        text = "alpha\nbeta\ngamma"
        assert dedupe.collapse_adjacent_duplicates(text) == text

    def test_handles_empty_string(self):
        dedupe = _load_module("rules/dedupe.py", "_t_dedupe_empty")
        assert dedupe.collapse_adjacent_duplicates("") == ""

    def test_handles_single_line_no_newlines(self):
        dedupe = _load_module("rules/dedupe.py", "_t_dedupe_single")
        assert dedupe.collapse_adjacent_duplicates("hello") == "hello"

    def test_collapses_two_identical_lines(self):
        dedupe = _load_module("rules/dedupe.py", "_t_dedupe_two")
        # Boundary case: N=2 should still collapse.
        out = dedupe.collapse_adjacent_duplicates("x\nx")
        assert out == "x\n  (... 1 more identical lines ...)"


# ---------------------------------------------------------------------------
# reducer.reduce_tool_output
# ---------------------------------------------------------------------------

class TestReducerPassthroughs:
    def test_passes_through_output_below_threshold(self):
        reducer = _load_reducer()
        short = "\x1b[32mtiny\x1b[0m"  # would reduce, but under threshold
        assert reducer.reduce_tool_output("terminal", short) is None

    def test_passes_through_error_json(self):
        reducer = _load_reducer()
        # Long enough to clear the threshold; still passes through.
        err = '{"error": "' + ("x" * 300) + '"}'
        assert reducer.reduce_tool_output("terminal", err) is None

    def test_passes_through_allowlisted_tool(self):
        reducer = _load_reducer()
        # Output that would otherwise reduce, but tool is allowlisted.
        payload = "\x1b[32m" + ("a\n" * 200) + "\x1b[0m"
        assert reducer.reduce_tool_output("read_file", payload) is None

    def test_returns_none_for_non_string_input(self):
        reducer = _load_reducer()
        assert reducer.reduce_tool_output("terminal", None) is None  # type: ignore[arg-type]
        assert reducer.reduce_tool_output("terminal", {"x": 1}) is None  # type: ignore[arg-type]

    def test_returns_none_when_no_actual_reduction(self):
        reducer = _load_reducer()
        # Long clean text with no ANSI AND no adjacent duplicates — no-op.
        clean = "\n".join(f"unique line number {i}" for i in range(50))
        assert reducer.reduce_tool_output("terminal", clean) is None


class TestReducerChain:
    def test_strips_ansi_and_dedupes_on_long_terminal_output(self):
        reducer = _load_reducer()
        # 300+ bytes, has ANSI, has dupes
        raw = "\x1b[32mhello\x1b[0m\n" + ("repeated line\n" * 30) + "tail"
        out = reducer.reduce_tool_output("terminal", raw)
        assert out is not None
        assert "\x1b" not in out
        assert "(... 29 more identical lines ...)" in out
        assert "tail" in out
        assert len(out) < len(raw)

    def test_unknown_tool_uses_universal_chain(self):
        reducer = _load_reducer()
        # Must clear PASSTHROUGH_THRESHOLD (240 bytes), so use 100 dup lines.
        raw = "\x1b[31merr\x1b[0m\n" + ("dup line\n" * 100)
        out = reducer.reduce_tool_output("some_unrecognized_tool", raw)
        assert out is not None
        assert "\x1b" not in out
        assert "(... 99 more identical lines ...)" in out

    def test_writes_observability_log_on_reduction(self, _isolate_env):
        reducer = _load_reducer()
        raw = "\x1b[32mhi\x1b[0m\n" + ("x\n" * 200)
        out = reducer.reduce_tool_output("terminal", raw)
        assert out is not None
        log_path = _isolate_env / "logs" / "tool_reducer.log"
        assert log_path.exists()
        content = log_path.read_text()
        assert "terminal" in content
        assert "->" in content
        assert "saved=" in content


class TestReducerFailOpen:
    def test_rule_exception_returns_none(self, monkeypatch):
        reducer = _load_reducer()
        # Monkeypatch the ansi rule to raise; reducer should swallow and return None
        from hermes_plugins.tool_output_reducer.rules import ansi as ansi_mod

        def _boom(_text):
            raise RuntimeError("boom")

        monkeypatch.setattr(ansi_mod, "strip_ansi", _boom)
        raw = "x" * 500
        # Should not raise; should return None (fail-open).
        assert reducer.reduce_tool_output("terminal", raw) is None


# ---------------------------------------------------------------------------
# Plugin register()
# ---------------------------------------------------------------------------

class TestPluginRegister:
    def test_registers_hook_when_env_var_unset(self, monkeypatch):
        monkeypatch.delenv("HERMES_REDUCER_DISABLE", raising=False)
        plugin = _load_plugin_init()
        ctx = MagicMock()
        plugin.register(ctx)
        ctx.register_hook.assert_called_once()
        args, _kwargs = ctx.register_hook.call_args
        assert args[0] == "transform_tool_result"

    def test_does_not_register_when_disable_env_set(self, monkeypatch):
        monkeypatch.setenv("HERMES_REDUCER_DISABLE", "1")
        plugin = _load_plugin_init()
        ctx = MagicMock()
        plugin.register(ctx)
        ctx.register_hook.assert_not_called()

    def test_hook_callable_returns_none_for_short_string(self, monkeypatch):
        monkeypatch.delenv("HERMES_REDUCER_DISABLE", raising=False)
        plugin = _load_plugin_init()
        # Call the hook callable directly with short input — should be no-op.
        out = plugin._on_transform_tool_result(
            tool_name="terminal", result="tiny"
        )
        assert out is None

    def test_hook_callable_returns_none_for_non_string_result(self, monkeypatch):
        monkeypatch.delenv("HERMES_REDUCER_DISABLE", raising=False)
        plugin = _load_plugin_init()
        out = plugin._on_transform_tool_result(
            tool_name="terminal", result={"not": "a string"}
        )
        assert out is None
