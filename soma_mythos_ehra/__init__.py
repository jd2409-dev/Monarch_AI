"""SOMA-Mythos-EHRA / Monarch AI.

Monarch AI is a compact implementation of the architecture described in this
repository:

* SOMA: tensorized grid physics plus JEPA-style latent energy prediction.
* Mythos/Omnithos: paced discrete lookahead with cycle-aware metacognition.
* EHRA: async runtime orchestration, action filtering, and JSONL telemetry.
"""

from soma_mythos_ehra.agent import MonarchAI, MonarchConfig
from soma_mythos_ehra.ehra.harness import EHRARuntime, RuntimeResult, TelemetryEvent
from soma_mythos_ehra.mythos.search import MythosConfig, MythosSearch
from soma_mythos_ehra.soma.gpu_simulator import TensorGridSimulator
from soma_mythos_ehra.soma.jepa import JEPAWorldModel

__all__ = [
    "EHRARuntime",
    "JEPAWorldModel",
    "MonarchAI",
    "MonarchConfig",
    "MythosConfig",
    "MythosSearch",
    "RuntimeResult",
    "TelemetryEvent",
    "TensorGridSimulator",
]
