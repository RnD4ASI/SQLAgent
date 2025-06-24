from abc import ABC, abstractmethod
from typing import Dict, Any
from app.backend.llm_providers.base import LLMProvider

class Topology(ABC):
    """
    Abstract base class for execution topologies.
    A topology defines how LLM calls and tool executions (like running SQL/Python/R)
    are orchestrated to fulfill a user's query.
    """

    def __init__(self, llm_provider: LLMProvider, topology_config: Dict[str, Any] | None = None):
        """
        Initializes the topology.

        Args:
            llm_provider (LLMProvider): An instance of an LLM provider to be used for LLM calls.
            topology_config (Dict[str, Any] | None): Configuration specific to the topology,
                                                     e.g., number of parallel calls, specific models for steps.
        """
        self.llm_provider = llm_provider
        self.topology_config = topology_config if topology_config else {}


    @abstractmethod
    def execute(
        self,
        natural_language_query: str,
        metadata: Dict[str, Any], # Table schema, etc.
        agent_type: str,          # 'sql', 'python_pandas', 'r_datatable'
        file_path: str,           # Path to the data file
        table_name: str,          # Name of the table/object in the file
        # Potentially other parameters like preferred LLM model for this execution, etc.
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Executes the topology to process the natural language query.

        Args:
            natural_language_query (str): The user's query.
            metadata (Dict[str, Any]): Schema information for the data.
            agent_type (str): The type of agent/code to generate ('sql', 'python_pandas', 'r_datatable').
            file_path (str): Full path to the data file.
            table_name (str): Name of the table or data object to be queried.
                               For R, this is the object name in the .Rdata file.
                               For SQL (CSV/Parquet), this is the table name DuckDB will use.
                               For Python Pandas, this is the initial DataFrame variable name.
            **kwargs: Additional arguments specific to the execution.

        Returns:
            Dict[str, Any]: A dictionary containing results, such as:
                            'executed_query_text': The actual code/query that was executed.
                            'results': The data results from the execution (e.g., list of dicts).
                            'error': Error message if any occurred.
                            'natural_language_response': A natural language summary of the results.
                            'intermediate_steps': (Optional) List of thoughts/actions taken by the topology.
        """
        pass

    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """Helper to get a value from the topology_config."""
        return self.topology_config.get(key, default)

    # Potentially, common helper methods for topologies can be added here later,
    # for example, a method to safely execute code using the existing helper functions
    # from app.py, or a common way to structure LLM calls for different sub-tasks.
    # For now, these will reside within the concrete topology implementations.
