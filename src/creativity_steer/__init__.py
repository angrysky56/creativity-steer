"""creativity-steer.

Repurpose the creativity-eval measurement metrics as a *generation-time*
control signal. Semantic entropy (divergent) and a multi-agent judge
(convergent) score the model's own candidates, and a Pareto selection rule
chooses the candidate that is both novel and task-fulfilling, instead of the
modal (greedy) one.

Key insight from the source paper: divergent and convergent creativity are
empirically *separable*, so novelty can be pushed without sacrificing task
fulfilment -- but only when optimised jointly, never semantic entropy alone.
"""

from creativity_steer.backends import (
    GenSample,
    LLMBackend,
    MockBackend,
    OllamaBackend,
    OpenAIBackend,
)
from creativity_steer.config import backend_summary, build_backend, load_env
from creativity_steer.chat import ChatConfig, chat_turn, chat_turn_stream
from creativity_steer.control import TurnController
from creativity_steer.entailment import make_entailment
from creativity_steer.judge import JudgeResult, multi_agent_judge
from creativity_steer.reference import novelty_vs_reference
from creativity_steer.stage1 import (
    Stage1Candidate,
    Stage1Config,
    Stage1Result,
    Stage1Trajectory,
    run_greedy_trajectory,
    run_stage1_trajectory,
    think_and_select,
)
from creativity_steer.variants import (
    brainstorm_variants,
    get_variants,
    independent_variants,
)
from creativity_steer.selection import (
    SelectionConfig,
    SelectionResult,
    StepRecord,
    run_trajectory,
    select_candidate,
)

__version__ = "0.3.0"

__all__ = [
    "GenSample",
    "LLMBackend",
    "MockBackend",
    "OllamaBackend",
    "OpenAIBackend",
    "build_backend",
    "backend_summary",
    "load_env",
    "make_entailment",
    "ChatConfig",
    "chat_turn",
    "chat_turn_stream",
    "TurnController",
    "JudgeResult",
    "multi_agent_judge",
    "novelty_vs_reference",
    "SelectionConfig",
    "SelectionResult",
    "StepRecord",
    "run_trajectory",
    "select_candidate",
    "Stage1Candidate",
    "Stage1Config",
    "Stage1Result",
    "Stage1Trajectory",
    "think_and_select",
    "run_stage1_trajectory",
    "run_greedy_trajectory",
    "brainstorm_variants",
    "independent_variants",
    "get_variants",
]
