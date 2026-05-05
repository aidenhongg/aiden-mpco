from contextlib import contextmanager
from unittest.mock import patch, MagicMock

from src.chains.chain import RMPChain, OptimizedCode, build_chain, _escape
from src.chains.prompts import GeneratedPrompt
from src.chains.evaluation import is_converged, CONVERGENCE_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextmanager
def _rmp_chain(proposer_prompt="Initial prompt", refiner_prompts=None, converge_results=None):
    """Yield a fully-mocked RMPChain. All patches stay active for the lifetime of
    the `with` block, so tests can call _refine_prompt / invoke without leaking
    to real network code.

    Both the target and meta-prompt roles share the same mock LLM, since the
    migration collapsed them to one local_llm() factory call.
    """
    refiner_prompts = refiner_prompts or ["Refined prompt 1"]
    converge_results = converge_results or [(True, 1.5)]

    with patch("src.chains.chain.local_llm") as mock_local_llm, \
         patch("src.chains.chain.rmp_proposer") as mock_proposer, \
         patch("src.chains.chain.rmp_refiner") as mock_refiner, \
         patch("src.chains.chain.is_converged") as mock_converged, \
         patch("src.chains.chain.get_openai_callback") as mock_cb_ctx:

        mock_llm = MagicMock()
        mock_target_chain = MagicMock()
        mock_llm.with_structured_output.return_value = mock_target_chain
        mock_target_chain.invoke.return_value = OptimizedCode(code="optimized")
        mock_local_llm.return_value = mock_llm

        mock_proposer_template = MagicMock()
        mock_proposer_msg = MagicMock()
        mock_proposer_msg.prompt = MagicMock()
        mock_proposer_msg.prompt.template = "Meta-meta-prompt text"
        mock_proposer_template.messages = [mock_proposer_msg]
        mock_proposer.return_value = mock_proposer_template

        proposer_chain = MagicMock()
        mock_proposer_template.__or__ = MagicMock(return_value=proposer_chain)
        proposer_chain.invoke.return_value = GeneratedPrompt(prompt=proposer_prompt)

        mock_refiner_template = MagicMock()
        mock_refiner.return_value = mock_refiner_template

        refiner_chain = MagicMock()
        mock_refiner_template.__or__ = MagicMock(return_value=refiner_chain)
        refiner_chain.invoke.side_effect = [
            GeneratedPrompt(prompt=p) for p in refiner_prompts
        ]

        mock_converged.side_effect = converge_results

        mock_cb = MagicMock()
        mock_cb.prompt_tokens = 100
        mock_cb.completion_tokens = 50
        mock_cb_ctx.return_value.__enter__ = MagicMock(return_value=mock_cb)
        mock_cb_ctx.return_value.__exit__ = MagicMock(return_value=False)

        chain = RMPChain("qwen3.5-9b-q4", "test-project")
        chain._mock_converged = mock_converged
        chain._mock_refiner_chain = refiner_chain
        chain._mock_target_chain = mock_target_chain

        yield chain


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRefinementConvergesEarly:
    def test_stops_at_convergence(self):
        with _rmp_chain(
            proposer_prompt="Initial",
            refiner_prompts=["Refined 1"],
            converge_results=[(True, 1.5)],
        ) as chain:
            chain._refine_prompt()

            assert chain._cached_prompt == "Refined 1"
            assert chain._converged is True
            assert len(chain._refinement_trace) == 2  # iteration 0 + 1
            assert chain._refinement_trace[0]["iteration"] == 0
            assert chain._refinement_trace[1]["iteration"] == 1
            assert chain._refinement_trace[1]["convergence_score"] == 1.5


class TestRefinementHitsMaxIterations:
    def test_stops_at_max(self):
        with _rmp_chain(
            proposer_prompt="Initial",
            refiner_prompts=["R1", "R2", "R3"],
            converge_results=[(False, 4.0), (False, 3.5), (False, 3.0)],
        ) as chain:
            chain._refine_prompt()

            assert chain._cached_prompt == "R3"
            assert chain._converged is False
            assert len(chain._refinement_trace) == 4  # iteration 0 + 3 refinements


class TestRefinementErrorUsesPartial:
    def test_catches_error_uses_last_good(self):
        with _rmp_chain(
            proposer_prompt="Initial",
            refiner_prompts=["Refined 1"],
            converge_results=[(False, 4.0)],
        ) as chain:
            # Re-mock the refiner template's __or__ so the second iteration raises
            with patch.object(chain, "_refiner_template") as mock_template:
                refiner_chain = MagicMock()
                mock_template.__or__ = MagicMock(return_value=refiner_chain)
                call_count = [0]

                def side_effect(*args, **kwargs):
                    call_count[0] += 1
                    if call_count[0] == 1:
                        return GeneratedPrompt(prompt="Refined 1")
                    raise RuntimeError("API timeout")

                refiner_chain.invoke.side_effect = side_effect
                chain._mock_converged.side_effect = [(False, 4.0)]

                chain._refine_prompt()

            assert chain._cached_prompt == "Refined 1"
            # Trace: initial + 1 success + 1 error
            assert len(chain._refinement_trace) == 3
            assert chain._refinement_trace[2].get("error") is not None


class TestIsConvergedThreshold:
    def test_exact_match_short_circuits(self):
        result, score = is_converged("identical prompt", "identical prompt")
        assert result is True
        assert score == 0.0

    def test_threshold_boundary(self):
        with patch("src.chains.evaluation._convergence_metric") as mock_metric:
            async def mock_score(sample):
                return CONVERGENCE_THRESHOLD
            mock_metric.single_turn_ascore = mock_score
            result, score = is_converged("A", "B")
            assert result is True
            assert score == CONVERGENCE_THRESHOLD


class TestEscapeSystemMessage:
    def test_escape_braces(self):
        prompt = "Use {dict} comprehensions for {key: value} pairs"
        escaped = _escape(prompt)
        assert escaped == "Use {{dict}} comprehensions for {{key: value}} pairs"

    def test_no_braces_unchanged(self):
        prompt = "Optimize this Python function"
        assert _escape(prompt) == prompt

    def test_template_does_not_crash(self):
        from langchain_core.prompts import ChatPromptTemplate
        generated = "Use {dict} and {set} for O(1) lookups"
        prompt = ChatPromptTemplate.from_messages([
            ("system", _escape(generated)),
            ("human", "test {code}"),
        ])
        # Should not raise — braces in system message are escaped
        result = prompt.format(code="x = 1")
        assert "{{" not in result
        assert "{dict}" in result


class TestBuildChainRouting:
    def test_rmp_returns_rmp_chain(self):
        with patch("src.chains.chain.local_llm", return_value=MagicMock()), \
             patch("src.chains.chain.rmp_proposer"), \
             patch("src.chains.chain.rmp_refiner"):
            chain = build_chain("qwen3.5-9b-q4", "rmp", "test-proj")
            assert isinstance(chain, RMPChain)

    def test_base_returns_runnable(self):
        with patch("src.chains.chain.AGENTS", {"qwen3.5-9b-q4": lambda: MagicMock()}):
            chain = build_chain("qwen3.5-9b-q4", "base", "test-proj")
            assert not isinstance(chain, RMPChain)


class TestTraceStructure:
    def test_trace_entries_have_required_fields(self):
        with _rmp_chain(
            proposer_prompt="Initial",
            refiner_prompts=["Refined 1", "Refined 2"],
            converge_results=[(False, 3.5), (True, 1.8)],
        ) as chain:
            chain._refine_prompt()

            for entry in chain._refinement_trace:
                assert "iteration" in entry
                assert "prompt" in entry
                assert "convergence_score" in entry
                assert isinstance(entry["iteration"], int)

            # First entry (proposer) has no convergence score
            assert chain._refinement_trace[0]["convergence_score"] is None

            # Refinement entries have telemetry
            for entry in chain._refinement_trace[1:]:
                assert "refine_prompt_tokens" in entry
                assert "refine_completion_tokens" in entry
                assert "refine_latency" in entry


class TestRegenerateRerunsRefinement:
    def test_regenerate_calls_refine_again(self):
        with _rmp_chain(
            proposer_prompt="Initial",
            refiner_prompts=["Refined 1"],
            converge_results=[(True, 1.5)],
        ) as chain:
            # _refine_prompt is patched to set a non-None cached_prompt so
            # downstream _escape() doesn't crash on None.
            def fake_refine():
                chain._cached_prompt = "fake refined prompt"

            with patch.object(chain, "_refine_prompt", side_effect=fake_refine) as mock_refine:
                chain._cached_prompt = None
                chain.invoke({"code": "x=1", "scope": "module"})
                assert mock_refine.call_count == 1

                chain.invoke({"code": "x=1", "scope": "module"}, regenerate=True)
                assert mock_refine.call_count == 2
