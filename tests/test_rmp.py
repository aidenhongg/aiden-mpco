import asyncio
from unittest.mock import patch, MagicMock, PropertyMock
import pytest

from src.chains.chain import RMPChain, OptimizedCode, build_chain, _escape
from src.chains.prompts import GeneratedPrompt
from src.chains.evaluation import is_converged, CONVERGENCE_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_meta_llm():
    """Return a mock ChatOpenAI that can be chained with with_structured_output."""
    llm = MagicMock()
    chain = MagicMock()
    llm.with_structured_output.return_value = chain
    return llm, chain


def _make_rmp_chain(proposer_prompt="Initial prompt", refiner_prompts=None, converge_results=None):
    """Build an RMPChain with fully mocked LLMs and convergence checks."""
    refiner_prompts = refiner_prompts or ["Refined prompt 1"]
    converge_results = converge_results or [(True, 1.5)]

    with patch("src.chains.chain.ChatOpenAI") as MockOpenAI, \
         patch("src.chains.chain.rmp_proposer") as mock_proposer, \
         patch("src.chains.chain.rmp_refiner") as mock_refiner, \
         patch("src.chains.chain.is_converged") as mock_converged, \
         patch("src.chains.chain.get_openai_callback") as mock_cb_ctx:

        # Mock the target LLM (AGENTS)
        mock_target = MagicMock()
        mock_target_chain = MagicMock()
        mock_target.with_structured_output.return_value = mock_target_chain
        mock_target_chain.invoke.return_value = OptimizedCode(code="optimized")

        # Mock the meta LLM (GPT-4o)
        mock_meta = MagicMock()
        MockOpenAI.return_value = mock_meta

        # Mock proposer template
        mock_proposer_template = MagicMock()
        mock_proposer_msg = MagicMock()
        mock_proposer_msg.prompt = MagicMock()
        mock_proposer_msg.prompt.template = "Meta-meta-prompt text"
        mock_proposer_template.messages = [mock_proposer_msg]
        mock_proposer.return_value = mock_proposer_template

        # Mock proposer chain: template | llm -> GeneratedPrompt
        proposer_chain = MagicMock()
        mock_proposer_template.__or__ = MagicMock(return_value=proposer_chain)
        proposer_chain.invoke.return_value = GeneratedPrompt(prompt=proposer_prompt)

        # Mock refiner template
        mock_refiner_template = MagicMock()
        mock_refiner.return_value = mock_refiner_template

        # Mock refiner chain: template | llm -> GeneratedPrompt (one per iteration)
        refiner_chain = MagicMock()
        mock_refiner_template.__or__ = MagicMock(return_value=refiner_chain)
        refiner_chain.invoke.side_effect = [
            GeneratedPrompt(prompt=p) for p in refiner_prompts
        ]

        # Mock convergence checks
        mock_converged.side_effect = converge_results

        # Mock get_openai_callback
        mock_cb = MagicMock()
        mock_cb.prompt_tokens = 100
        mock_cb.completion_tokens = 50
        mock_cb_ctx.return_value.__enter__ = MagicMock(return_value=mock_cb)
        mock_cb_ctx.return_value.__exit__ = MagicMock(return_value=False)

        # Build the chain with mocked AGENTS
        with patch("src.chains.chain.AGENTS", {"test-agent": lambda: mock_target}):
            chain = RMPChain("test-agent", "test-project")

        # Attach mocks so tests can inspect them
        chain._mock_converged = mock_converged
        chain._mock_refiner_chain = refiner_chain
        chain._mock_target_chain = mock_target_chain

    return chain


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRefinementConvergesEarly:
    def test_stops_at_convergence(self):
        chain = _make_rmp_chain(
            proposer_prompt="Initial",
            refiner_prompts=["Refined 1"],
            converge_results=[(True, 1.5)],
        )
        chain._refine_prompt()

        assert chain._cached_prompt == "Refined 1"
        assert chain._converged is True
        assert len(chain._refinement_trace) == 2  # iteration 0 + 1
        assert chain._refinement_trace[0]["iteration"] == 0
        assert chain._refinement_trace[1]["iteration"] == 1
        assert chain._refinement_trace[1]["convergence_score"] == 1.5


class TestRefinementHitsMaxIterations:
    def test_stops_at_max(self):
        chain = _make_rmp_chain(
            proposer_prompt="Initial",
            refiner_prompts=["R1", "R2", "R3"],
            converge_results=[(False, 4.0), (False, 3.5), (False, 3.0)],
        )
        chain._refine_prompt()

        assert chain._cached_prompt == "R3"
        assert chain._converged is False
        assert len(chain._refinement_trace) == 4  # iteration 0 + 3 refinements


class TestRefinementErrorUsesPartial:
    def test_catches_error_uses_last_good(self):
        chain = _make_rmp_chain(
            proposer_prompt="Initial",
            refiner_prompts=["Refined 1"],
            converge_results=[(False, 4.0)],
        )
        # Make the second refiner call raise
        chain._mock_refiner_chain = MagicMock()

        # Re-mock: first call succeeds, second raises
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

            with patch("src.chains.chain.is_converged") as mock_conv, \
                 patch("src.chains.chain.get_openai_callback") as mock_cb_ctx:
                mock_conv.return_value = (False, 4.0)
                mock_cb = MagicMock()
                mock_cb.prompt_tokens = 100
                mock_cb.completion_tokens = 50
                mock_cb_ctx.return_value.__enter__ = MagicMock(return_value=mock_cb)
                mock_cb_ctx.return_value.__exit__ = MagicMock(return_value=False)

                chain._refine_prompt()

        assert chain._cached_prompt == "Refined 1"
        # Trace should have: initial + 1 success + 1 error
        assert len(chain._refinement_trace) == 3
        assert chain._refinement_trace[2].get("error") is not None


class TestIsConvergedThreshold:
    def test_below_threshold_is_converged(self):
        with patch("src.chains.evaluation._convergence_metric") as mock_metric:
            mock_metric.single_turn_ascore = MagicMock(
                return_value=asyncio.coroutine(lambda: 1.5)()
            )
            result, score = is_converged("prompt A", "prompt A slightly different")
            # Exact match short circuit won't fire since strings differ
            # But we need to handle the async mock properly

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
        assert "{{" not in result  # escaped braces should be rendered as single braces
        assert "{dict}" in result


class TestBuildChainRouting:
    def test_rmp_returns_rmp_chain(self):
        with patch("src.chains.chain.ChatOpenAI"), \
             patch("src.chains.chain.rmp_proposer"), \
             patch("src.chains.chain.rmp_refiner"), \
             patch("src.chains.chain.AGENTS", {"o4-mini": lambda: MagicMock()}):
            chain = build_chain("o4-mini", "rmp", "test-proj")
            assert isinstance(chain, RMPChain)

    def test_base_returns_runnable(self):
        with patch("src.chains.chain.AGENTS", {"o4-mini": lambda: MagicMock()}):
            chain = build_chain("o4-mini", "base", "test-proj")
            assert not isinstance(chain, RMPChain)


class TestTraceStructure:
    def test_trace_entries_have_required_fields(self):
        chain = _make_rmp_chain(
            proposer_prompt="Initial",
            refiner_prompts=["Refined 1", "Refined 2"],
            converge_results=[(False, 3.5), (True, 1.8)],
        )
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
        chain = _make_rmp_chain(
            proposer_prompt="Initial",
            refiner_prompts=["Refined 1"],
            converge_results=[(True, 1.5)],
        )

        with patch.object(chain, "_refine_prompt") as mock_refine:
            chain._cached_prompt = None
            chain.invoke({"code": "x=1", "scope": "module"})
            assert mock_refine.call_count == 1

            chain.invoke({"code": "x=1", "scope": "module"}, regenerate=True)
            assert mock_refine.call_count == 2
