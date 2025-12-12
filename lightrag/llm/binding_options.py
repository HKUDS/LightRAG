"""
Module that implements containers for specific LLM bindings.

This module provides container implementations for various Large Language Model
bindings and integrations.
"""

import argparse
import json
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, cast, get_args, get_origin

from lightrag.constants import DEFAULT_TEMPERATURE
from lightrag.utils import get_env_value


def _resolve_optional_type(field_type: Any) -> Any:
    """Return the concrete type for Optional/Union annotations."""
    origin = get_origin(field_type)
    if origin in (list, dict, tuple):
        return field_type

    args = get_args(field_type)
    if args:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return non_none_args[0]
    return field_type


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
                        if (not k.startswith('_') and not callable(v) and not isinstance(v, classmethod))
                    }
                )
        else:
            # Only direct class variables
            vars_dict = {
                k: v
                for k, v in klass.__dict__.items()
                if (not k.startswith('_') and not callable(v) and not isinstance(v, classmethod))
            }

        return vars_dict

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        group = parser.add_argument_group(f'{cls._binding_name} binding options')
        for arg_item in cls.args_env_name_type_value():
            # Handle JSON parsing for list types
            if arg_item['type'] is list[str]:

                def json_list_parser(value):
                    try:
                        parsed = json.loads(value)
                        if not isinstance(parsed, list):
                            raise argparse.ArgumentTypeError(f'Expected JSON array, got {type(parsed).__name__}')
                        return parsed
                    except json.JSONDecodeError as e:
                        raise argparse.ArgumentTypeError(f'Invalid JSON: {e}') from e

                # Get environment variable with JSON parsing
                env_value = get_env_value(f'{arg_item["env_name"]}', argparse.SUPPRESS)
                if env_value is not argparse.SUPPRESS:
                    try:
                        env_value = json_list_parser(env_value)
                    except argparse.ArgumentTypeError:
                        env_value = argparse.SUPPRESS

                group.add_argument(
                    f'--{arg_item["argname"]}',
                    type=json_list_parser,
                    default=env_value,
                    help=arg_item['help'],
                )
            # Handle JSON parsing for dict types
            elif arg_item['type'] is dict:

                def json_dict_parser(value):
                    try:
                        parsed = json.loads(value)
                        if not isinstance(parsed, dict):
                            raise argparse.ArgumentTypeError(f'Expected JSON object, got {type(parsed).__name__}')
                        return parsed
                    except json.JSONDecodeError as e:
                        raise argparse.ArgumentTypeError(f'Invalid JSON: {e}') from e

                # Get environment variable with JSON parsing
                env_value = get_env_value(f'{arg_item["env_name"]}', argparse.SUPPRESS)
                if env_value is not argparse.SUPPRESS:
                    try:
                        env_value = json_dict_parser(env_value)
                    except argparse.ArgumentTypeError:
                        env_value = argparse.SUPPRESS

                group.add_argument(
                    f'--{arg_item["argname"]}',
                    type=json_dict_parser,
                    default=env_value,
                    help=arg_item['help'],
                )
            # Handle boolean types specially to avoid argparse bool() constructor issues
            elif arg_item['type'] is bool:

                def bool_parser(value):
                    """Custom boolean parser that handles string representations correctly"""
                    if isinstance(value, bool):
                        return value
                    if isinstance(value, str):
                        return value.lower() in ('true', '1', 'yes', 't', 'on')
                    return bool(value)

                # Get environment variable with proper type conversion
                env_value = get_env_value(f'{arg_item["env_name"]}', argparse.SUPPRESS, bool)

                group.add_argument(
                    f'--{arg_item["argname"]}',
                    type=bool_parser,
                    default=env_value,
                    help=arg_item['help'],
                )
            else:
                resolved_type = arg_item['type']
                if resolved_type is not None:
                    resolved_type = _resolve_optional_type(resolved_type)

                group.add_argument(
                    f'--{arg_item["argname"]}',
                    type=resolved_type,
                    default=get_env_value(f'{arg_item["env_name"]}', argparse.SUPPRESS),
                    help=arg_item['help'],
                )

    @classmethod
    def args_env_name_type_value(cls):
        import dataclasses

        args_prefix = f'{cls._binding_name}'.replace('_', '-')
        env_var_prefix = f'{cls._binding_name}_'.upper()
        help = cls._help

        # Check if this is a dataclass and use dataclass fields
        if dataclasses.is_dataclass(cls):
            for field in dataclasses.fields(cls):
                # Skip private fields
                if field.name.startswith('_'):
                    continue

                # Get default value
                if field.default is not dataclasses.MISSING:
                    default_value = field.default
                elif field.default_factory is not dataclasses.MISSING:
                    default_value = field.default_factory()
                else:
                    default_value = None

                argdef = {
                    'argname': f'{args_prefix}-{field.name}',
                    'env_name': f'{env_var_prefix}{field.name.upper()}',
                    'type': _resolve_optional_type(field.type),
                    'default': default_value,
                    'help': f'{cls._binding_name} -- ' + help.get(field.name, ''),
                }

                yield argdef
        else:
            # Fallback to old method for non-dataclass classes
            all_vars = cast(dict[str, Any], cls._all_class_vars(cls))
            class_vars = {
                key: value for key, value in all_vars.items() if not callable(value) and not key.startswith('_')
            }

            # Get type hints to properly detect List[str] types
            type_hints = {}
            for base in cast(tuple[type, ...], cls.__mro__):
                if hasattr(base, '__annotations__'):
                    type_hints.update(base.__annotations__)

            for class_var in class_vars:
                # Use type hint if available, otherwise fall back to type of value
                var_type = type_hints.get(class_var, type(class_vars[class_var]))

                argdef = {
                    'argname': f'{args_prefix}-{class_var}',
                    'env_name': f'{env_var_prefix}{class_var.upper()}',
                    'type': var_type,
                    'default': class_vars[class_var],
                    'help': f'{cls._binding_name} -- ' + help.get(class_var, ''),
                }

                yield argdef

    @classmethod
    def generate_dot_env_sample(cls):
        """
        Generate a sample .env file for all LightRAG binding options.

        This method creates a .env file that includes all the binding options
        defined by the subclasses of BindingOptions. It uses the args_env_name_type_value()
        method to get the list of all options and their default values.

        Returns:
            str: A string containing the contents of the sample .env file.
        """
        from io import StringIO

        sample_top = (
            '#' * 80
            + '\n'
            + (
                '# Autogenerated .env entries list for LightRAG binding options\n'
                '#\n'
                '# To generate run:\n'
                '# $ python -m lightrag.llm.binding_options\n'
            )
            + '#' * 80
            + '\n'
        )

        sample_bottom = ('#\n# End of .env entries for LightRAG binding options\n') + '#' * 80 + '\n'

        sample_stream = StringIO()
        sample_stream.write(sample_top)
        for klass in cls.__subclasses__():
            for arg_item in klass.args_env_name_type_value():
                if arg_item['help']:
                    sample_stream.write(f'# {arg_item["help"]}\n')

                # Handle JSON formatting for list and dict types
                if arg_item['type'] is list[str] or arg_item['type'] is dict:
                    default_value = json.dumps(arg_item['default'])
                else:
                    default_value = arg_item['default']

                sample_stream.write(f'# {arg_item["env_name"]}={default_value}\n\n')

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
        prefix = cls._binding_name + '_'
        skipchars = len(prefix)
        options = {key[skipchars:]: value for key, value in vars(args).items() if key.startswith(prefix)}

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
    temperature: float = DEFAULT_TEMPERATURE  # Controls randomness (0.0-2.0)
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
    stop: list[str] = field(default_factory=list)  # Stop sequences

    # optional help strings
    _help: ClassVar[dict[str, str]] = {
        'num_ctx': 'Context window size (number of tokens)',
        'num_predict': 'Maximum number of tokens to predict',
        'num_keep': 'Number of tokens to keep from the initial prompt',
        'seed': 'Random seed for generation (-1 for random)',
        'temperature': 'Controls randomness (0.0-2.0, higher = more creative)',
        'top_k': 'Top-k sampling parameter (0 = disabled)',
        'top_p': 'Top-p (nucleus) sampling parameter (0.0-1.0)',
        'tfs_z': 'Tail free sampling parameter (1.0 = disabled)',
        'typical_p': 'Typical probability mass (1.0 = disabled)',
        'min_p': 'Minimum probability threshold (0.0 = disabled)',
        'repeat_last_n': 'Number of tokens to consider for repetition penalty',
        'repeat_penalty': 'Penalty for repetition (1.0 = no penalty)',
        'presence_penalty': 'Penalty for token presence (-2.0 to 2.0)',
        'frequency_penalty': 'Penalty for token frequency (-2.0 to 2.0)',
        'mirostat': 'Mirostat sampling algorithm (0=disabled, 1=Mirostat 1.0, 2=Mirostat 2.0)',
        'mirostat_tau': 'Mirostat target entropy',
        'mirostat_eta': 'Mirostat learning rate',
        'numa': 'Enable NUMA optimization',
        'num_batch': 'Batch size for processing',
        'num_gpu': 'Number of GPUs to use (-1 for auto)',
        'main_gpu': 'Main GPU index',
        'low_vram': 'Optimize for low VRAM',
        'num_thread': 'Number of CPU threads (0 for auto)',
        'f16_kv': 'Use half-precision for key/value cache',
        'logits_all': 'Return logits for all tokens',
        'vocab_only': 'Only load vocabulary',
        'use_mmap': 'Use memory mapping for model files',
        'use_mlock': 'Lock model in memory',
        'embedding_only': 'Only use for embeddings',
        'penalize_newline': 'Penalize newline tokens',
        'stop': 'Stop sequences (JSON array of strings, e.g., \'["</s>", "\\n\\n"]\')',
    }


@dataclass
class OllamaEmbeddingOptions(_OllamaOptionsMixin, BindingOptions):
    """Options for Ollama embeddings with specialized configuration for embedding tasks."""

    # mandatory name of binding
    _binding_name: ClassVar[str] = 'ollama_embedding'


@dataclass
class OllamaLLMOptions(_OllamaOptionsMixin, BindingOptions):
    """Options for Ollama LLM with specialized configuration for LLM tasks."""

    # mandatory name of binding
    _binding_name: ClassVar[str] = 'ollama_llm'


# =============================================================================
# Binding Options for Gemini
# =============================================================================
@dataclass
class GeminiLLMOptions(BindingOptions):
    """Options for Google Gemini models."""

    _binding_name: ClassVar[str] = 'gemini_llm'

    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int | None = None
    candidate_count: int = 1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop_sequences: list[str] = field(default_factory=list)
    seed: int | None = None
    thinking_config: dict | None = None
    safety_settings: dict | None = None

    _help: ClassVar[dict[str, str]] = {
        'temperature': 'Controls randomness (0.0-2.0, higher = more creative)',
        'top_p': 'Nucleus sampling parameter (0.0-1.0)',
        'top_k': 'Limits sampling to the top K tokens (1 disables the limit)',
        'max_output_tokens': 'Maximum tokens generated in the response',
        'candidate_count': 'Number of candidates returned per request',
        'presence_penalty': 'Penalty for token presence (-2.0 to 2.0)',
        'frequency_penalty': 'Penalty for token frequency (-2.0 to 2.0)',
        'stop_sequences': 'Stop sequences (JSON array of strings, e.g., \'["END"]\')',
        'seed': 'Random seed for reproducible generation (leave empty for random)',
        'thinking_config': 'Thinking configuration (JSON dict, e.g., \'{"thinking_budget": 1024}\' or \'{"include_thoughts": true}\')',
        'safety_settings': 'JSON object with Gemini safety settings overrides',
    }


@dataclass
class GeminiEmbeddingOptions(BindingOptions):
    """Options for Google Gemini embedding models."""

    _binding_name: ClassVar[str] = 'gemini_embedding'

    task_type: str = 'RETRIEVAL_DOCUMENT'

    _help: ClassVar[dict[str, str]] = {
        'task_type': 'Task type for embedding optimization (RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING, CODE_RETRIEVAL_QUERY, QUESTION_ANSWERING, FACT_VERIFICATION)',
    }


# =============================================================================
# Binding Options for OpenAI
# =============================================================================
#
# OpenAI binding options provide configuration for OpenAI's API and Azure OpenAI.
# These options control model behavior, sampling parameters, and generation settings.
# The parameters are based on OpenAI's API specification and provide fine-grained
# control over model inference and generation.
#
# =============================================================================
@dataclass
class OpenAILLMOptions(BindingOptions):
    """Options for OpenAI LLM with configuration for OpenAI and Azure OpenAI API calls."""

    # mandatory name of binding
    _binding_name: ClassVar[str] = 'openai_llm'

    # Sampling and generation parameters
    frequency_penalty: float = 0.0  # Penalty for token frequency (-2.0 to 2.0)
    max_completion_tokens: int | None = None  # Maximum number of tokens to generate
    presence_penalty: float = 0.0  # Penalty for token presence (-2.0 to 2.0)
    reasoning_effort: str = 'medium'  # Reasoning effort level (low, medium, high)
    safety_identifier: str = ''  # Safety identifier for content filtering
    service_tier: str = ''  # Service tier for API usage
    stop: list[str] = field(default_factory=list)  # Stop sequences
    temperature: float = DEFAULT_TEMPERATURE  # Controls randomness (0.0 to 2.0)
    top_p: float = 1.0  # Nucleus sampling parameter (0.0 to 1.0)
    max_tokens: int | None = None  # Maximum number of tokens to generate(deprecated, use max_completion_tokens instead)
    extra_body: dict[str, Any] | None = None  # Extra body parameters for OpenRouter of vLLM

    # Help descriptions
    _help: ClassVar[dict[str, str]] = {
        'frequency_penalty': 'Penalty for token frequency (-2.0 to 2.0, positive values discourage repetition)',
        'max_completion_tokens': 'Maximum number of tokens to generate (optional, leave empty for model default)',
        'presence_penalty': 'Penalty for token presence (-2.0 to 2.0, positive values encourage new topics)',
        'reasoning_effort': 'Reasoning effort level for o1 models (low, medium, high)',
        'safety_identifier': 'Safety identifier for content filtering (optional)',
        'service_tier': 'Service tier for API usage (optional)',
        'stop': 'Stop sequences (JSON array of strings, e.g., \'["</s>", "\\n\\n"]\')',
        'temperature': 'Controls randomness (0.0-2.0, higher = more creative)',
        'top_p': 'Nucleus sampling parameter (0.0-1.0, lower = more focused)',
        'max_tokens': 'Maximum number of tokens to generate (deprecated, use max_completion_tokens instead)',
        'extra_body': 'Extra body parameters for OpenRouter of vLLM (JSON dict, e.g., \'"reasoning": {"reasoning": {"enabled": false}}\')',
    }


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

if __name__ == '__main__':
    import sys

    import dotenv
    # from io import StringIO

    dotenv.load_dotenv(dotenv_path='.env', override=False)

    # env_strstream = StringIO(
    #     ("OLLAMA_LLM_TEMPERATURE=0.1\nOLLAMA_EMBEDDING_TEMPERATURE=0.2\n")
    # )
    # # Load environment variables from .env file
    # dotenv.load_dotenv(stream=env_strstream)

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Add arguments for OllamaEmbeddingOptions, OllamaLLMOptions, and OpenAILLMOptions
        parser = ArgumentParser(description='Test binding options')
        OllamaEmbeddingOptions.add_args(parser)
        OllamaLLMOptions.add_args(parser)
        OpenAILLMOptions.add_args(parser)

        # Parse arguments test
        args = parser.parse_args(
            [
                '--ollama-embedding-num_ctx',
                '1024',
                '--ollama-llm-num_ctx',
                '2048',
                '--openai-llm-temperature',
                '0.7',
                '--openai-llm-max_completion_tokens',
                '1000',
                '--openai-llm-stop',
                '["</s>", "\\n\\n"]',
                '--openai-llm-extra_body',
                '{"effort": "high", "max_tokens": 2000, "exclude": false, "enabled": true}',
            ]
        )
        print('Final args for LLM and Embedding:')
        print(f'{args}\n')

        print('Ollama LLM options:')
        print(OllamaLLMOptions.options_dict(args))

        print('\nOllama Embedding options:')
        print(OllamaEmbeddingOptions.options_dict(args))

        print('\nOpenAI LLM options:')
        print(OpenAILLMOptions.options_dict(args))

        # Test creating OpenAI options instance
        openai_options = OpenAILLMOptions(
            temperature=0.8,
            max_completion_tokens=1500,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            stop=['<|end|>', '\n\n'],
        )
        print('\nOpenAI LLM options instance:')
        print(openai_options.asdict())

        # Test creating OpenAI options instance with extra_body parameter
        openai_options_with_extra_body = OpenAILLMOptions(
            temperature=0.9,
            max_completion_tokens=2000,
            extra_body={
                'effort': 'medium',
                'max_tokens': 1500,
                'exclude': True,
                'enabled': True,
            },
        )
        print('\nOpenAI LLM options instance with extra_body:')
        print(openai_options_with_extra_body.asdict())

        # Test dict parsing functionality
        print('\n' + '=' * 50)
        print('TESTING DICT PARSING FUNCTIONALITY')
        print('=' * 50)

        # Test valid JSON dict parsing
        test_parser = ArgumentParser(description='Test dict parsing')
        OpenAILLMOptions.add_args(test_parser)

        try:
            test_args = test_parser.parse_args(['--openai-llm-extra_body', '{"effort": "low", "max_tokens": 1000}'])
            print('✓ Valid JSON dict parsing successful:')
            print(f'  Parsed extra_body: {OpenAILLMOptions.options_dict(test_args)["extra_body"]}')
        except Exception as e:
            print(f'✗ Valid JSON dict parsing failed: {e}')

        # Test invalid JSON dict parsing
        try:
            test_args = test_parser.parse_args(
                [
                    '--openai-llm-extra_body',
                    '{"effort": "low", "max_tokens": 1000',  # Missing closing brace
                ]
            )
            print("✗ Invalid JSON should have failed but didn't")
        except SystemExit:
            print('✓ Invalid JSON dict parsing correctly rejected')
        except Exception as e:
            print(f'✓ Invalid JSON dict parsing correctly rejected: {e}')

        # Test non-dict JSON parsing
        try:
            test_args = test_parser.parse_args(
                [
                    '--openai-llm-extra_body',
                    '["not", "a", "dict"]',  # Array instead of dict
                ]
            )
            print("✗ Non-dict JSON should have failed but didn't")
        except SystemExit:
            print('✓ Non-dict JSON parsing correctly rejected')
        except Exception as e:
            print(f'✓ Non-dict JSON parsing correctly rejected: {e}')

        print('\n' + '=' * 50)
        print('TESTING ENVIRONMENT VARIABLE SUPPORT')
        print('=' * 50)

        # Test environment variable support for dict
        import os

        os.environ['OPENAI_LLM_EXTRA_BODY'] = '{"effort": "high", "max_tokens": 3000, "exclude": false}'

        env_parser = ArgumentParser(description='Test env var dict parsing')
        OpenAILLMOptions.add_args(env_parser)

        try:
            env_args = env_parser.parse_args([])  # No command line args, should use env var
            extra_body_from_env = OpenAILLMOptions.options_dict(env_args).get('extra_body')
            if extra_body_from_env:
                print('✓ Environment variable dict parsing successful:')
                print(f'  Parsed extra_body from env: {extra_body_from_env}')
            else:
                print('✗ Environment variable dict parsing failed: No extra_body found')
        except Exception as e:
            print(f'✗ Environment variable dict parsing failed: {e}')
        finally:
            # Clean up environment variable
            if 'OPENAI_LLM_REASONING' in os.environ:
                del os.environ['OPENAI_LLM_REASONING']

    else:
        print(BindingOptions.generate_dot_env_sample())
