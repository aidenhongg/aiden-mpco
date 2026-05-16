"""Tests for post-patch regression detection.

When an optimized snippet is patched in, the profiler re-runs the target
repo's pytest suite and compares failure counts against the baseline.
These tests cover the two pieces of that path: `_extract_failures`
(parsing JUnit XML into structured failure dicts) and the
`TestRegressionError` raised when the patched code regresses the suite.

No LLM API calls — pure data-structure and XML-parsing logic.
"""

import xml.etree.ElementTree as ET

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
