"""
Module that implements containers for specific LLM bindings.

This module provides container implementations for various Large Language Model
bindings and integrations.
"""

from argparse import ArgumentParser, Namespace
import argparse
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

from lightrag.utils import get_env_value


# =============================================================================
# BindingOptions Base Class
# =============================================================================
#
# The BindingOptions class serves as the foundation for all LLM provider bindings
# in LightRAG. It provides a standardized framework for:
#
# 1. Configuration Management:
#    - Defines how each LLM provider's configuration parameters are structured
#    - Handles default values and type information for each parameter
#    - Maps configuration options to command-line arguments and environment variables
#
# 2. Environment Integration:
#    - Automatically generates environment variable names from binding parameters
#    - Provides methods to create sample .env files for easy configuration
#    - Supports configuration via environment variables with fallback to defaults
#
# 3. Command-Line Interface:
#    - Dynamically generates command-line arguments for all registered bindings
#    - Maintains consistent naming conventions across different LLM providers
#    - Provides help text and type validation for each configuration option
#
# 4. Extensibility:
#    - Uses class introspection to automatically discover all binding subclasses
#    - Requires minimal boilerplate code when adding new LLM provider bindings
#    - Maintains separation of concerns between different provider configurations
#
# This design pattern ensures that adding support for a new LLM provider requires
# only defining the provider-specific parameters and help text, while the base
# class handles all the common functionality for argument parsing, environment
# variable handling, and configuration management.
#
# Instances of a derived class of BindingOptions can be used to store multiple
# runtime configurations of options for a single LLM provider. using the
# asdict() method to convert the options to a dictionary.
#
# =============================================================================
@dataclass
class BindingOptions:
    """Base class for binding options."""

    # mandatory name of binding
    _binding_name: ClassVar[str]

    # optional help message for each option
    _help: ClassVar[dict[str, str]]

    @staticmethod
    def _all_class_vars(klass: type, include_inherited=True) -> dict[str, Any]:
        """Print class variables, optionally including inherited ones"""
        if include_inherited:
            # Get all class variables from MRO
            vars_dict = {}
            for base in reversed(klass.__mro__[:-1]):  # Exclude 'object'
                vars_dict.update(
                    {
                        k: v
                        for k, v in base.__dict__.items()
                        if (
                            not k.startswith("_")
                            and not callable(v)
                            and not isinstance(v, classmethod)
                        )
                    }
                )
        else:
            # Only direct class variables
            vars_dict = {
                k: v
                for k, v in klass.__dict__.items()
                if (
                    not k.startswith("_")
                    and not callable(v)
                    and not isinstance(v, classmethod)
                )
            }

        return vars_dict

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        group = parser.add_argument_group(f"{cls._binding_name} binding options")
        for arg_item in cls.args_env_name_type_value():
            group.add_argument(
                f"--{arg_item['argname']}",
                type=arg_item["type"],
                default=get_env_value(f"{arg_item['env_name']}", argparse.SUPPRESS),
                help=arg_item["help"],
            )

    @classmethod
    def args_env_name_type_value(cls):
        args_prefix = f"{cls._binding_name}".replace("_", "-")
        env_var_prefix = f"{cls._binding_name}_".upper()
        class_vars = {
            key: value
            for key, value in cls._all_class_vars(cls).items()
            if not callable(value) and not key.startswith("_")
        }
        help = cls._help

        for class_var in class_vars:
            argdef = {
                "argname": f"{args_prefix}-{class_var}",
                "env_name": f"{env_var_prefix}{class_var.upper()}",
                "type": type(class_vars[class_var]),
                "default": class_vars[class_var],
                "help": f"{cls._binding_name} -- " + help.get(class_var, ""),
            }

            yield argdef

    @classmethod
    def generate_dot_env_sample(cls):
        from io import StringIO

        sample_top = (
            "#" * 80
            + "\n"
            + (
                "# Autogenerated .env entries list for LightRAG binding options\n"
                "#\n"
                "# To generate run:\n"
                "# $ python -m lightrag.llm.binding_options\n"
            )
            + "#" * 80
            + "\n"
        )

        sample_bottom = (
            ("#\n# End of .env entries for LightRAG binding options\n")
            + "#" * 80
            + "\n"
        )

        sample_stream = StringIO()
        sample_stream.write(sample_top)
        for klass in cls.__subclasses__():
            for arg_item in klass.args_env_name_type_value():
                if arg_item["help"]:
                    sample_stream.write(f"# {arg_item['help']}\n")
                sample_stream.write(
                    f"# {arg_item['env_name']}={arg_item['default']}\n\n"
                )

        sample_stream.write(sample_bottom)
        return sample_stream.getvalue()

    @classmethod
    def options_dict(cls, args: Namespace) -> dict[str, Any]:
        """
        Extract options dictionary for a specific binding from parsed arguments.

        This method filters the parsed command-line arguments to return only those
        that belong to the specific binding class. It removes the binding prefix
        from argument names to create a clean options dictionary.

        Args:
            args (Namespace): Parsed command-line arguments containing all binding options

        Returns:
            dict[str, Any]: Dictionary mapping option names (without prefix) to their values

        Example:
            If args contains {'ollama_num_ctx': 512, 'other_option': 'value'}
            and this is called on OllamaOptions, it returns {'num_ctx': 512}
        """
        prefix = cls._binding_name + "_"
        skipchars = len(prefix)
        options = {
            key[skipchars:]: value
            for key, value in vars(args).items()
            if key.startswith(prefix)
        }

        return options

    def asdict(self) -> dict[str, Any]:
        """
        Convert an instance of binding options to a dictionary.

        This method uses dataclasses.asdict() to convert the dataclass instance
        into a dictionary representation, including all its fields and values.

        Returns:
            dict[str, Any]: Dictionary representation of the binding options instance
        """
        return asdict(self)


# =============================================================================
# Binding Options for Different LLM Providers
# =============================================================================
#
# This section contains dataclass definitions for various LLM provider options.
# Each binding option class inherits from BindingOptions and defines:
#   - _binding_name: Unique identifier for the binding
#   - Configuration parameters with default values
#   - _help: Dictionary mapping parameter names to help descriptions
#
# To add a new binding:
#   1. Create a new dataclass inheriting from BindingOptions
#   2. Set the _binding_name class variable
#   3. Define configuration parameters as class attributes
#   4. Add corresponding help strings in the _help dictionary
#
# =============================================================================


# =============================================================================
# Binding Options for Ollama
# =============================================================================
#
# Ollama binding options provide configuration for the Ollama local LLM server.
# These options control model behavior, sampling parameters, hardware utilization,
# and performance settings. The parameters are based on Ollama's API specification
# and provide fine-grained control over model inference and generation.
#
# The _OllamaOptionsMixin defines the complete set of available options, while
# OllamaEmbeddingOptions and OllamaLLMOptions provide specialized configurations
# for embedding and language model tasks respectively.
# =============================================================================
@dataclass
class _OllamaOptionsMixin:
    """Options for Ollama bindings."""

    # Core context and generation parameters
    num_ctx: int = 32768  # Context window size (number of tokens)
    num_predict: int = 128  # Maximum number of tokens to predict
    num_keep: int = 0  # Number of tokens to keep from the initial prompt
    seed: int = -1  # Random seed for generation (-1 for random)

    # Sampling parameters
    temperature: float = 0.8  # Controls randomness (0.0-2.0)
    top_k: int = 40  # Top-k sampling parameter
    top_p: float = 0.9  # Top-p (nucleus) sampling parameter
    tfs_z: float = 1.0  # Tail free sampling parameter
    typical_p: float = 1.0  # Typical probability mass
    min_p: float = 0.0  # Minimum probability threshold

    # Repetition control
    repeat_last_n: int = 64  # Number of tokens to consider for repetition penalty
    repeat_penalty: float = 1.1  # Penalty for repetition
    presence_penalty: float = 0.0  # Penalty for token presence
    frequency_penalty: float = 0.0  # Penalty for token frequency

    # Mirostat sampling
    mirostat: int = (
        # Mirostat sampling algorithm (0=disabled, 1=Mirostat 1.0, 2=Mirostat 2.0)
        0
    )
    mirostat_tau: float = 5.0  # Mirostat target entropy
    mirostat_eta: float = 0.1  # Mirostat learning rate

    # Hardware and performance parameters
    numa: bool = False  # Enable NUMA optimization
    num_batch: int = 512  # Batch size for processing
    num_gpu: int = -1  # Number of GPUs to use (-1 for auto)
    main_gpu: int = 0  # Main GPU index
    low_vram: bool = False  # Optimize for low VRAM
    num_thread: int = 0  # Number of CPU threads (0 for auto)

    # Memory and model parameters
    f16_kv: bool = True  # Use half-precision for key/value cache
    logits_all: bool = False  # Return logits for all tokens
    vocab_only: bool = False  # Only load vocabulary
    use_mmap: bool = True  # Use memory mapping for model files
    use_mlock: bool = False  # Lock model in memory
    embedding_only: bool = False  # Only use for embeddings

    # Output control
    penalize_newline: bool = True  # Penalize newline tokens
    stop: str = ""  # Stop sequences (comma-separated)

    # optional help strings
    _help: ClassVar[dict[str, str]] = {
        "num_ctx": "Context window size (number of tokens)",
        "num_predict": "Maximum number of tokens to predict",
        "num_keep": "Number of tokens to keep from the initial prompt",
        "seed": "Random seed for generation (-1 for random)",
        "temperature": "Controls randomness (0.0-2.0, higher = more creative)",
        "top_k": "Top-k sampling parameter (0 = disabled)",
        "top_p": "Top-p (nucleus) sampling parameter (0.0-1.0)",
        "tfs_z": "Tail free sampling parameter (1.0 = disabled)",
        "typical_p": "Typical probability mass (1.0 = disabled)",
        "min_p": "Minimum probability threshold (0.0 = disabled)",
        "repeat_last_n": "Number of tokens to consider for repetition penalty",
        "repeat_penalty": "Penalty for repetition (1.0 = no penalty)",
        "presence_penalty": "Penalty for token presence (-2.0 to 2.0)",
        "frequency_penalty": "Penalty for token frequency (-2.0 to 2.0)",
        "mirostat": "Mirostat sampling algorithm (0=disabled, 1=Mirostat 1.0, 2=Mirostat 2.0)",
        "mirostat_tau": "Mirostat target entropy",
        "mirostat_eta": "Mirostat learning rate",
        "numa": "Enable NUMA optimization",
        "num_batch": "Batch size for processing",
        "num_gpu": "Number of GPUs to use (-1 for auto)",
        "main_gpu": "Main GPU index",
        "low_vram": "Optimize for low VRAM",
        "num_thread": "Number of CPU threads (0 for auto)",
        "f16_kv": "Use half-precision for key/value cache",
        "logits_all": "Return logits for all tokens",
        "vocab_only": "Only load vocabulary",
        "use_mmap": "Use memory mapping for model files",
        "use_mlock": "Lock model in memory",
        "embedding_only": "Only use for embeddings",
        "penalize_newline": "Penalize newline tokens",
        "stop": "Stop sequences (comma-separated string)",
    }


# =============================================================================
# Ollama Binding Options - Specialized Configurations
# =============================================================================
#
# This section defines specialized binding option classes for different Ollama
# use cases. Both classes inherit from OllamaOptionsMixin to share the complete
# set of Ollama configuration parameters, while providing distinct binding names
# for command-line argument generation and environment variable handling.
#
# OllamaEmbeddingOptions: Specialized for embedding tasks
# OllamaLLMOptions: Specialized for language model/chat tasks
#
# Each class maintains its own binding name prefix, allowing users to configure
# embedding and LLM options independently when both are used in the same application.
# =============================================================================


@dataclass
class OllamaEmbeddingOptions(_OllamaOptionsMixin, BindingOptions):
    """Options for Ollama embeddings with specialized configuration for embedding tasks."""

    # mandatory name of binding
    _binding_name: ClassVar[str] = "ollama_embedding"


@dataclass
class OllamaLLMOptions(_OllamaOptionsMixin, BindingOptions):
    """Options for Ollama LLM with specialized configuration for LLM tasks."""

    # mandatory name of binding
    _binding_name: ClassVar[str] = "ollama_llm"


# =============================================================================
# Additional LLM Provider Bindings
# =============================================================================
#
# This section is where you can add binding options for other LLM providers.
# Each new binding should follow the same pattern as the Ollama bindings above:
#
# 1. Create a dataclass that inherits from BindingOptions
# 2. Set a unique _binding_name class variable (e.g., "openai", "anthropic")
# 3. Define configuration parameters as class attributes with default values
# 4. Add a _help class variable with descriptions for each parameter
#
# Example template for a new provider:
#
# @dataclass
# class NewProviderOptions(BindingOptions):
#     """Options for NewProvider LLM binding."""
#
#     _binding_name: ClassVar[str] = "newprovider"
#
#     # Configuration parameters
#     api_key: str = ""
#     max_tokens: int = 1000
#     model: str = "default-model"
#
#     # Help descriptions
#     _help: ClassVar[dict[str, str]] = {
#         "api_key": "API key for authentication",
#         "max_tokens": "Maximum tokens to generate",
#         "model": "Model name to use",
#     }
#
# =============================================================================

# TODO: Add binding options for additional LLM providers here
# Common providers to consider: OpenAI, Anthropic, Cohere, Hugging Face, etc.

# =============================================================================
# Main Section - For Testing and Sample Generation
# =============================================================================
#
# When run as a script, this module:
# 1. Generates and prints a sample .env file with all binding options
# 2. If "test" argument is provided, demonstrates argument parsing with Ollama binding
#
# Usage:
#   python -m lightrag.llm.binding_options           # Generate .env sample
#   python -m lightrag.llm.binding_options test      # Test argument parsing
#
# =============================================================================

if __name__ == "__main__":
    import sys
    import dotenv
    from io import StringIO

    print(BindingOptions.generate_dot_env_sample())

    env_strstream = StringIO(
        ("OLLAMA_LLM_TEMPERATURE=0.1\nOLLAMA_EMBEDDING_TEMPERATURE=0.2\n")
    )

    # Load environment variables from .env file
    dotenv.load_dotenv(stream=env_strstream)

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        parser = ArgumentParser(description="Test Ollama binding")
        OllamaEmbeddingOptions.add_args(parser)
        OllamaLLMOptions.add_args(parser)
        args = parser.parse_args(
            [
                "--ollama-embedding-num_ctx",
                "1024",
                "--ollama-llm-num_ctx",
                "2048",
            ]
        )
        print(args)

        # test LLM options
        ollama_options = OllamaLLMOptions.options_dict(args)
        print(ollama_options)
        print(OllamaLLMOptions(num_ctx=30000).asdict())

        # test embedding options
        embedding_options = OllamaEmbeddingOptions.options_dict(args)
        print(embedding_options)
        print(OllamaEmbeddingOptions(**embedding_options).asdict())
