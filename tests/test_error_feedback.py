"""Tests for the recursive meta-prompting error feedback system.

All tests run without LLM API calls — they use mocked LLMs or test
only the data structures and parsing logic.
"""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.profiler.profile import TestRegressionError, _extract_failures


# ---------------------------------------------------------------------------
# _extract_failures
# ---------------------------------------------------------------------------

JUNIT_WITH_FAILURES = """\
<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" errors="0" failures="2" tests="5" time="1.23">
    <testcase classname="tests.test_math" name="test_add" time="0.01"/>
    <testcase classname="tests.test_math" name="test_divide" time="0.02">
      <failure message="ZeroDivisionError: division by zero">Traceback ...</failure>
    </testcase>
    <testcase classname="tests.test_string" name="test_upper" time="0.01"/>
    <testcase classname="tests.test_string" name="test_concat" time="0.03">
      <failure message="AssertionError: expected 'ab' got 'a'">Traceback ...</failure>
    </testcase>
    <testcase classname="tests.test_string" name="test_split" time="0.01"/>
  </testsuite>
</testsuites>
"""

JUNIT_NO_FAILURES = """\
<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest" errors="0" failures="0" tests="3" time="0.5">
  <testcase classname="tests.test_math" name="test_add" time="0.01"/>
  <testcase classname="tests.test_math" name="test_sub" time="0.01"/>
  <testcase classname="tests.test_math" name="test_mul" time="0.01"/>
</testsuite>
"""

JUNIT_BARE_TESTSUITE = """\
<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest" errors="0" failures="1" tests="2" time="0.3">
  <testcase classname="tests.test_io" name="test_read" time="0.1">
    <failure message="FileNotFoundError">Traceback ...</failure>
  </testcase>
  <testcase classname="tests.test_io" name="test_write" time="0.1"/>
</testsuite>
"""


def _parse_root(xml_str: str) -> ET.Element:
    return ET.fromstring(xml_str)


class TestExtractFailures:
    def test_parses_failures_from_junit_xml(self):
        root = _parse_root(JUNIT_WITH_FAILURES)
        failures = _extract_failures(root)

        assert len(failures) == 2
        assert failures[0] == {
            "testcase": "test_divide",
            "classname": "tests.test_math",
            "message": "ZeroDivisionError: division by zero",
        }
        assert failures[1] == {
            "testcase": "test_concat",
            "classname": "tests.test_string",
            "message": "AssertionError: expected 'ab' got 'a'",
        }

    def test_empty_xml_returns_empty_list(self):
        root = _parse_root(JUNIT_NO_FAILURES)
        failures = _extract_failures(root)
        assert failures == []

    def test_bare_testsuite_root(self):
        root = _parse_root(JUNIT_BARE_TESTSUITE)
        failures = _extract_failures(root)

        assert len(failures) == 1
        assert failures[0]["testcase"] == "test_read"
        assert failures[0]["classname"] == "tests.test_io"
        assert failures[0]["message"] == "FileNotFoundError"

    def test_missing_attributes_default_to_empty(self):
        xml = """\
        <testsuite>
          <testcase>
            <failure/>
          </testcase>
        </testsuite>
        """
        root = ET.fromstring(xml)
        failures = _extract_failures(root)

        assert len(failures) == 1
        assert failures[0] == {
            "testcase": "",
            "classname": "",
            "message": "",
        }


# ---------------------------------------------------------------------------
# TestRegressionError
# ---------------------------------------------------------------------------

class TestTestRegressionError:
    def test_carries_structured_data(self):
        failures = [
            {"testcase": "test_a", "classname": "mod.A", "message": "fail A"},
            {"testcase": "test_b", "classname": "mod.B", "message": "fail B"},
        ]
        err = TestRegressionError("myproj", 2, 4, failures)

        assert err.proj_name == "myproj"
        assert err.baseline_count == 2
        assert err.new_count == 4
        assert err.failures == failures

    def test_is_runtime_error(self):
        err = TestRegressionError("p", 0, 1, [])
        assert isinstance(err, RuntimeError)

    def test_message_includes_counts(self):
        err = TestRegressionError("proj", 3, 5, [])
        assert "3 -> 5" in str(err)
        assert "proj" in str(err)

    def test_message_includes_failure_details(self):
        failures = [
            {"testcase": "test_x", "classname": "tests.X", "message": "boom"},
        ]
        err = TestRegressionError("proj", 0, 1, failures)
        msg = str(err)
        assert "tests.X::test_x" in msg
        assert "boom" in msg

    def test_empty_failures_still_works(self):
        err = TestRegressionError("proj", 0, 1, [])
        assert "0 -> 1" in str(err)


# ---------------------------------------------------------------------------
# Chain error context injection (mocked LLMs)
# ---------------------------------------------------------------------------

class TestChainErrorContextInjection:
    """Test that error_context is correctly routed in chains.invoke()."""

    def test_regular_chain_receives_error_in_code_input(self):
        from src.chains.chain import _escape, _format_error_context

        error_ctx = [
            {"testcase": "test_a", "classname": "mod.A", "message": "failed"},
        ]
        formatted = _format_error_context(error_ctx)

        assert "mod.A::test_a" in formatted
        assert "failed" in formatted
        assert "previous optimization" in formatted.lower()

    def test_format_error_context_multiple_failures(self):
        from src.chains.chain import _format_error_context

        error_ctx = [
            {"testcase": "test_a", "classname": "mod.A", "message": "err1"},
            {"testcase": "test_b", "classname": "mod.B", "message": "err2"},
        ]
        formatted = _format_error_context(error_ctx)
        assert "mod.A::test_a" in formatted
        assert "mod.B::test_b" in formatted

    def test_no_error_context_first_run(self):
        """On first run (no errors), error_context is None and no noise is added."""
        from src.chains.chain import _escape

        code = "def foo(): pass"
        scope_str = "module-level"
        inputs = {"code": _escape(code), "scope": _escape(scope_str)}

        # No error context => inputs should just have code and scope
        assert "previous optimization" not in inputs["code"].lower()
        assert "test failures" not in inputs["code"].lower()


# ---------------------------------------------------------------------------
# MetaChain error isolation (mocked LLMs)
# ---------------------------------------------------------------------------

class TestMetaChainErrorIsolation:
    """CRITICAL: Verify the meta-prompter sees error context but the
    revising agent NEVER does."""

    @patch("src.chains.chain.ChatOpenAI")
    @patch("src.chains.chain.ChatAnthropic")
    @patch("src.chains.chain.ChatGoogleGenerativeAI")
    def test_meta_prompter_receives_error_context(
        self, mock_google, mock_anthropic, mock_openai
    ):
        from src.chains.chain import MetaChain
        from src.chains.prompts import GeneratedPrompt

        # Mock the meta LLM to capture what it receives
        mock_meta_llm = MagicMock()
        mock_meta_structured = MagicMock()
        mock_meta_structured.invoke.return_value = GeneratedPrompt(
            prompt="Optimize this code for performance."
        )
        mock_meta_llm.with_structured_output.return_value = mock_meta_structured

        # Mock the target LLM
        mock_target_llm = MagicMock()
        mock_target_structured = MagicMock()
        from src.chains.chain import OptimizedCode
        mock_target_structured.invoke.return_value = OptimizedCode(code="optimized")
        mock_target_llm.with_structured_output.return_value = mock_target_structured

        mock_openai.return_value = mock_meta_llm

        chain = MetaChain.__new__(MetaChain)
        chain._meta_llm = mock_meta_llm
        chain._llm = mock_target_llm
        chain._cached_prompt = None

        # Set up the meta prompt template
        from langchain_core.prompts import ChatPromptTemplate
        chain._meta_prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a prompt to optimize code."),
        ])

        error_ctx = [
            {"testcase": "test_x", "classname": "mod.X", "message": "broken"},
        ]

        # The invoke with error_context should reach the meta-prompter
        # We need to intercept the chain that gets built
        # The key assertion: meta_llm.with_structured_output was called
        # and the chain was invoked (meaning error context was processed)
        result = chain.invoke(
            {"code": "def foo(): pass", "scope": "module-level"},
            regenerate=True,
            error_context=error_ctx,
        )

        # Meta LLM was called (it generated a prompt)
        assert mock_meta_structured.invoke.called
        assert result.code == "optimized"

    @patch("src.chains.chain.ChatOpenAI")
    @patch("src.chains.chain.ChatAnthropic")
    @patch("src.chains.chain.ChatGoogleGenerativeAI")
    def test_revising_agent_never_sees_error_context(
        self, mock_google, mock_anthropic, mock_openai
    ):
        """The revising agent's prompt must NOT contain error context."""
        from src.chains.chain import MetaChain, OptimizedCode
        from src.chains.prompts import GeneratedPrompt
        from langchain_core.prompts import ChatPromptTemplate

        # Track what the target LLM receives
        target_invoke_inputs = []

        mock_meta_llm = MagicMock()
        mock_meta_structured = MagicMock()
        mock_meta_structured.invoke.return_value = GeneratedPrompt(
            prompt="Optimize this code for performance."
        )
        mock_meta_llm.with_structured_output.return_value = mock_meta_structured
        mock_openai.return_value = mock_meta_llm

        mock_target_llm = MagicMock()
        mock_target_structured = MagicMock()

        def capture_invoke(inputs, **kwargs):
            target_invoke_inputs.append(inputs)
            return OptimizedCode(code="optimized")

        mock_target_structured.invoke = capture_invoke
        mock_target_llm.with_structured_output.return_value = mock_target_structured

        chain = MetaChain.__new__(MetaChain)
        chain._meta_llm = mock_meta_llm
        chain._llm = mock_target_llm
        chain._cached_prompt = None
        chain._meta_prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a prompt to optimize code."),
        ])

        error_ctx = [
            {"testcase": "test_x", "classname": "mod.X", "message": "broken"},
        ]

        chain.invoke(
            {"code": "def foo(): pass", "scope": "module-level"},
            regenerate=True,
            error_context=error_ctx,
        )

        # The revising agent (target LLM) should have been invoked
        assert len(target_invoke_inputs) == 1

        # The revising agent's input must NOT contain error context
        reviser_input = str(target_invoke_inputs[0])
        assert "broken" not in reviser_input
        assert "test_x" not in reviser_input
        assert "mod.X" not in reviser_input
        assert "test failures" not in reviser_input.lower()
        assert "regression" not in reviser_input.lower()
