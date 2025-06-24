from typing import Dict, Any
from app.backend.llm_providers.base import LLMProvider # Required for Topology __init__
from .base import Topology
from .sequential_reflect import SequentialReflectTopology
from .parallel_ensemble import ParallelEnsembleTopology
from .iterative_reason_act import IterativeReasonActTopology
from .default_fallback import DefaultFallbackTopology

class TopologyFactoryError(ValueError):
    """Custom error for TopologyFactory."""
    pass

class TopologyFactory:
    """
    Factory class to create instances of execution topologies.
    """

    @staticmethod
    def get_topology(
        topology_name: str,
        llm_provider: LLMProvider, # Topologies require an LLMProvider instance
        topology_specific_config: Dict[str, Any] | None = None
    ) -> Topology:
        """
        Gets a topology instance based on its name.

        Args:
            topology_name (str): The name of the topology.
                                 (e.g., "sequential_reflect", "parallel_ensemble",
                                  "iterative_reason_act", "default_fallback")
            llm_provider (LLMProvider): The LLM provider instance to be used by the topology.
            topology_specific_config (Dict[str, Any] | None): Configuration dictionary specific
                                                              to the requested topology.
                                                              This is passed to the topology's constructor.

        Returns:
            Topology: An instance of the requested topology.

        Raises:
            TopologyFactoryError: If the topology name is unknown.
        """
        if topology_specific_config is None:
            topology_specific_config = {}

        if topology_name.lower() == "sequential_reflect":
            return SequentialReflectTopology(llm_provider, topology_specific_config)
        elif topology_name.lower() == "parallel_ensemble":
            return ParallelEnsembleTopology(llm_provider, topology_specific_config)
        elif topology_name.lower() == "iterative_reason_act":
            return IterativeReasonActTopology(llm_provider, topology_specific_config)
        elif topology_name.lower() == "default_fallback":
            # DefaultFallbackTopology has its own internal config structure for sub-topologies
            # but topology_specific_config can still override its top-level settings.
            return DefaultFallbackTopology(llm_provider, topology_specific_config)

        # Future topologies can be added here
        # elif topology_name.lower() == "new_fancy_topology":
        #     return NewFancyTopology(llm_provider, topology_specific_config)

        else:
            raise TopologyFactoryError(
                f"Unknown topology name: {topology_name}. "
                f"Available: sequential_reflect, parallel_ensemble, iterative_reason_act, default_fallback."
            )

# Example usage (assuming llm_provider is an instantiated LLMProvider):
# try:
#     # config_for_parallel = {"code_gen_models": ["model_a", "model_b"], "max_workers": 2}
#     # parallel_topo = TopologyFactory.get_topology("parallel_ensemble", llm_provider, config_for_parallel)
#
#     # react_config = {"max_iterations": 7}
#     # react_topo = TopologyFactory.get_topology("iterative_reason_act", llm_provider, react_config)
#
#     # default_topo = TopologyFactory.get_topology("default_fallback", llm_provider)
# except TopologyFactoryError as e:
#     print(e)
