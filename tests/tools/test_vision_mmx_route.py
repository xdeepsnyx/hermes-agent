"""Tests for the custom-patch mmx-vision route in tools/vision_tools.py.

Verifies that when the active main provider is `minimax`, vision_analyze_tool
routes through the mmx CLI subprocess instead of async_call_llm. Failure modes
must not silently fall through to async_call_llm (which would hit MiniMax's
broken /anthropic vision and hallucinate).
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.vision_tools import (
    MMX_BIN,
    MMX_VISION_LABEL,
    _mmx_vision_describe,
    vision_analyze_tool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_png(tmp_path: Path) -> Path:
    """Write a 1x1 PNG to a temp path and return it."""
    import base64

    data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    )
    p = tmp_path / "tiny.png"
    p.write_bytes(data)
    return p


def _make_fake_proc(stdout: bytes, stderr: bytes = b"", returncode: int = 0):
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.kill = MagicMock()
    return proc


# ---------------------------------------------------------------------------
# _mmx_vision_describe: subprocess invocation + output parsing
# ---------------------------------------------------------------------------


class TestMmxVisionDescribe:
    """Direct tests of the mmx subprocess wrapper."""

    @pytest.mark.asyncio
    async def test_success_returns_content_field(self, tiny_png):
        ok = b'{"content": "Hello world", "base_resp": {"status_code": 0, "status_msg": "success"}}'
        with patch("tools.vision_tools.Path.exists", return_value=True), \
             patch(
                "tools.vision_tools.asyncio.create_subprocess_exec",
                AsyncMock(return_value=_make_fake_proc(ok)),
            ) as mock_spawn:
            result = await _mmx_vision_describe(tiny_png, "describe")
        assert result == "Hello world"
        # Verify we called mmx with the right shape
        args, _ = mock_spawn.call_args
        assert args[0] == MMX_BIN
        assert "vision" in args and "describe" in args
        assert "--image" in args and str(tiny_png) in args
        assert "--prompt" in args
        assert "--non-interactive" in args

    @pytest.mark.asyncio
    async def test_cli_error_shape_raises(self, tiny_png):
        cli_err = b'{"error": {"code": 2, "message": "File not found: /bad/path"}}'
        with patch("tools.vision_tools.Path.exists", return_value=True), \
             patch(
                "tools.vision_tools.asyncio.create_subprocess_exec",
                AsyncMock(return_value=_make_fake_proc(cli_err, returncode=2)),
            ):
            with pytest.raises(RuntimeError, match="CLI error code=2"):
                await _mmx_vision_describe(tiny_png, "describe")

    @pytest.mark.asyncio
    async def test_server_error_shape_raises(self, tiny_png):
        srv_err = b'{"content": "", "base_resp": {"status_code": 1024, "status_msg": "content policy"}}'
        with patch("tools.vision_tools.Path.exists", return_value=True), \
             patch(
                "tools.vision_tools.asyncio.create_subprocess_exec",
                AsyncMock(return_value=_make_fake_proc(srv_err)),
            ):
            with pytest.raises(RuntimeError, match="server error status_code=1024"):
                await _mmx_vision_describe(tiny_png, "describe")

    @pytest.mark.asyncio
    async def test_empty_content_with_success_status_raises(self, tiny_png):
        empty_ok = b'{"content": "", "base_resp": {"status_code": 0, "status_msg": "success"}}'
        with patch("tools.vision_tools.Path.exists", return_value=True), \
             patch(
                "tools.vision_tools.asyncio.create_subprocess_exec",
                AsyncMock(return_value=_make_fake_proc(empty_ok)),
            ):
            with pytest.raises(RuntimeError, match="empty content"):
                await _mmx_vision_describe(tiny_png, "describe")

    @pytest.mark.asyncio
    async def test_non_json_stdout_raises(self, tiny_png):
        with patch("tools.vision_tools.Path.exists", return_value=True), \
             patch(
                "tools.vision_tools.asyncio.create_subprocess_exec",
                AsyncMock(return_value=_make_fake_proc(b"<html>auth wall</html>")),
            ):
            with pytest.raises(RuntimeError, match="non-JSON output"):
                await _mmx_vision_describe(tiny_png, "describe")

    @pytest.mark.asyncio
    async def test_missing_binary_raises(self, tiny_png):
        with patch("tools.vision_tools.Path.exists", return_value=False):
            with pytest.raises(RuntimeError, match="not installed"):
                await _mmx_vision_describe(tiny_png, "describe")

    @pytest.mark.asyncio
    async def test_minimax_api_key_sourced_from_env_file_when_missing(self, tiny_png, monkeypatch):
        ok = b'{"content": "OK", "base_resp": {"status_code": 0}}'
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        captured_env: dict = {}

        async def fake_exec(*args, **kwargs):
            captured_env.update(kwargs.get("env") or {})
            return _make_fake_proc(ok)

        with patch("tools.vision_tools.Path.exists", return_value=True), \
             patch(
                "tools.vision_tools._load_minimax_api_key_from_env_file",
                return_value="sk-fromfile-test",
            ), \
             patch(
                "tools.vision_tools.asyncio.create_subprocess_exec",
                side_effect=fake_exec,
            ):
            result = await _mmx_vision_describe(tiny_png, "describe")
        assert result == "OK"
        assert captured_env.get("MINIMAX_API_KEY") == "sk-fromfile-test"


# ---------------------------------------------------------------------------
# vision_analyze_tool: end-to-end interception when provider == "minimax"
# ---------------------------------------------------------------------------


class TestVisionAnalyzeToolMmxRoute:
    """End-to-end: vision_analyze_tool routes to mmx when minimax is active."""

    @pytest.mark.asyncio
    async def test_minimax_provider_routes_to_mmx(self, tiny_png):
        ok = b'{"content": "A tiny red square", "base_resp": {"status_code": 0, "status_msg": "success"}}'
        async_call_mock = AsyncMock(side_effect=AssertionError(
            "async_call_llm must NOT be invoked when provider=minimax"
        ))
        with patch("agent.auxiliary_client._read_main_provider", return_value="minimax"), \
             patch("tools.vision_tools.Path.exists", return_value=True), \
             patch(
                "tools.vision_tools.asyncio.create_subprocess_exec",
                AsyncMock(return_value=_make_fake_proc(ok)),
            ), \
             patch("tools.vision_tools.async_call_llm", async_call_mock):
            result_json = await vision_analyze_tool(
                image_url=str(tiny_png), user_prompt="describe this"
            )
        result = json.loads(result_json)
        assert result["success"] is True
        assert result["analysis"] == "A tiny red square"
        async_call_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_minimax_mmx_failure_does_not_fall_through_to_llm(self, tiny_png):
        """Critical: if mmx fails, we MUST NOT silently call async_call_llm
        (which would hit MiniMax's broken /anthropic vision endpoint and
        hallucinate). We return a clean error response instead."""
        cli_err = b'{"error": {"code": 2, "message": "Auth required"}}'
        async_call_mock = AsyncMock(side_effect=AssertionError(
            "async_call_llm must NOT be invoked on mmx failure for minimax"
        ))
        with patch("agent.auxiliary_client._read_main_provider", return_value="minimax"), \
             patch("tools.vision_tools.Path.exists", return_value=True), \
             patch(
                "tools.vision_tools.asyncio.create_subprocess_exec",
                AsyncMock(return_value=_make_fake_proc(cli_err, returncode=2)),
            ), \
             patch("tools.vision_tools.async_call_llm", async_call_mock):
            result_json = await vision_analyze_tool(
                image_url=str(tiny_png), user_prompt="describe this"
            )
        result = json.loads(result_json)
        assert result["success"] is False
        assert "mmx vision" in result["error"]
        assert "Auth required" in result["error"]
        # Analysis text must convey the failure to the agent
        assert "unavailable" in result["analysis"].lower()
        async_call_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_minimax_provider_uses_existing_llm_path(self, tiny_png):
        """When provider != minimax, the mmx intercept must NOT trigger."""
        # Mock the LLM path to return a valid response
        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message = MagicMock()
        fake_response.choices[0].message.content = "From the real LLM"

        async_call_mock = AsyncMock(return_value=fake_response)
        spawn_mock = AsyncMock(side_effect=AssertionError(
            "mmx subprocess must NOT be spawned when provider != minimax"
        ))
        with patch("agent.auxiliary_client._read_main_provider", return_value="openai-codex"), \
             patch(
                "tools.vision_tools.asyncio.create_subprocess_exec",
                spawn_mock,
            ), \
             patch("tools.vision_tools.async_call_llm", async_call_mock), \
             patch(
                "tools.vision_tools.extract_content_or_reasoning",
                return_value="From the real LLM",
            ):
            result_json = await vision_analyze_tool(
                image_url=str(tiny_png), user_prompt="describe this"
            )
        result = json.loads(result_json)
        assert result["success"] is True
        assert result["analysis"] == "From the real LLM"
        spawn_mock.assert_not_called()
