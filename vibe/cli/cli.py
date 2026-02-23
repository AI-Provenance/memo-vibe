from __future__ import annotations

import argparse
import os
import sys

from rich import print as rprint

from vibe.cli.textual_ui.app import run_textual_ui
from vibe.core.agent_loop import AgentLoop
from vibe.core.agents.models import BuiltinAgentName
from vibe.core.config import (
    MissingAPIKeyError,
    MissingPromptFileError,
    ModelConfig,
    ProviderConfig,
    VibeConfig,
    load_dotenv_values,
)
from vibe.core.paths.config_paths import CONFIG_FILE, HISTORY_FILE
from vibe.core.programmatic import run_programmatic
from vibe.core.session.session_loader import SessionLoader
from vibe.core.types import LLMMessage, OutputFormat, Role
from vibe.core.utils import ConversationLimitException, logger
from vibe.setup.onboarding import run_onboarding


def _apply_api_key_override(api_key: str, model_name: str | None) -> None:
    """Set API key in environment variable based on model or default provider."""
    if model_name:
        config = VibeConfig.load()
        model_config = _find_model_config(config, model_name)
        if model_config:
            provider = config.get_provider_for_model(model_config)
            if provider.api_key_env_var:
                os.environ[provider.api_key_env_var] = api_key
                return

    # Default: try common provider API keys
    for env_var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MISTRAL_API_KEY"]:
        os.environ[env_var] = api_key
        break


def _find_model_config(config: VibeConfig, model_name: str) -> ModelConfig | None:
    """Find model config by name or alias."""
    for model in config.models:
        if model.name == model_name or model.alias == model_name:
            return model
    return None


def _apply_model_override(config: VibeConfig, model_name: str) -> VibeConfig:
    """Override the active model in config."""
    model_config = _find_model_config(config, model_name)

    if model_config:
        # Model exists in config, set it as active
        config.active_model = model_config.alias or model_config.name
        return config

    # Model doesn't exist - we need to create a temporary config with the model
    # Try to determine the provider from the model name or use openai as default
    provider_name = _infer_provider_from_model(model_name)

    # Check if provider exists, if not create a temporary one
    provider = None
    for p in config.providers:
        if p.name == provider_name:
            provider = p
            break

    if provider is None:
        # Create a new provider config
        provider = ProviderConfig(
            name=provider_name,
            api_base=_get_api_base_for_provider(provider_name),
            api_key_env_var=f"{provider_name.upper()}_API_KEY",
        )
        config.providers.append(provider)

    # Create model config
    new_model = ModelConfig(
        name=model_name,
        provider=provider_name,
        alias=model_name.split("/")[-1],  # Use last part of model name as alias
        input_price=0.0,
        output_price=0.0,
    )
    config.models.append(new_model)
    config.active_model = new_model.alias

    return config


def _infer_provider_from_model(model_name: str) -> str:
    """Infer provider from model name."""
    model_lower = model_name.lower()
    if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
        return "openai"
    if "claude" in model_lower:
        return "anthropic"
    if "mistral" in model_lower or "codestral" in model_lower:
        return "mistral"
    if "llama" in model_lower:
        return "ollama"
    if "qwen" in model_lower:
        return "ollama"
    # Default to openai
    return "openai"


def _get_api_base_for_provider(provider: str) -> str:
    """Get API base URL for a provider."""
    bases = {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com",
        "mistral": "https://api.mistral.ai/v1",
        "ollama": "http://localhost:11434/v1",
        "fireworks": "https://api.fireworks.ai/inference/v1",
    }
    return bases.get(provider, "https://api.openai.com/v1")


def get_initial_agent_name(args: argparse.Namespace) -> str:
    if args.prompt is not None and args.agent == BuiltinAgentName.DEFAULT:
        return BuiltinAgentName.AUTO_APPROVE
    return args.agent


def get_prompt_from_stdin() -> str | None:
    if sys.stdin.isatty():
        return None
    try:
        if content := sys.stdin.read().strip():
            sys.stdin = sys.__stdin__ = open("/dev/tty")
            return content
    except KeyboardInterrupt:
        pass
    except OSError:
        return None

    return None


def load_config_or_exit() -> VibeConfig:
    try:
        return VibeConfig.load()
    except MissingAPIKeyError:
        run_onboarding()
        return VibeConfig.load()
    except MissingPromptFileError as e:
        rprint(f"[yellow]Invalid system prompt id: {e}[/]")
        sys.exit(1)
    except ValueError as e:
        rprint(f"[yellow]{e}[/]")
        sys.exit(1)


def bootstrap_config_files() -> None:
    if not CONFIG_FILE.path.exists():
        try:
            VibeConfig.save_updates(VibeConfig.create_default())
        except Exception as e:
            rprint(f"[yellow]Could not create default config file: {e}[/]")

    if not HISTORY_FILE.path.exists():
        try:
            HISTORY_FILE.path.parent.mkdir(parents=True, exist_ok=True)
            HISTORY_FILE.path.write_text("Hello Vibe!\n", "utf-8")
        except Exception as e:
            rprint(f"[yellow]Could not create history file: {e}[/]")


def load_session(
    args: argparse.Namespace, config: VibeConfig
) -> list[LLMMessage] | None:
    if not args.continue_session and not args.resume:
        return None

    if not config.session_logging.enabled:
        rprint(
            "[red]Session logging is disabled. "
            "Enable it in config to use --continue or --resume[/]"
        )
        sys.exit(1)

    session_to_load = None
    if args.continue_session:
        session_to_load = SessionLoader.find_latest_session(config.session_logging)
        if not session_to_load:
            rprint(
                f"[red]No previous sessions found in "
                f"{config.session_logging.save_dir}[/]"
            )
            sys.exit(1)
    else:
        session_to_load = SessionLoader.find_session_by_id(
            args.resume, config.session_logging
        )
        if not session_to_load:
            rprint(
                f"[red]Session '{args.resume}' not found in "
                f"{config.session_logging.save_dir}[/]"
            )
            sys.exit(1)

    try:
        loaded_messages, _ = SessionLoader.load_session(session_to_load)
        return loaded_messages
    except Exception as e:
        rprint(f"[red]Failed to load session: {e}[/]")
        sys.exit(1)


def _load_messages_from_previous_session(
    agent_loop: AgentLoop, loaded_messages: list[LLMMessage]
) -> None:
    non_system_messages = [msg for msg in loaded_messages if msg.role != Role.system]
    agent_loop.messages.extend(non_system_messages)
    logger.info("Loaded %d messages from previous session", len(non_system_messages))


def run_cli(args: argparse.Namespace) -> None:
    load_dotenv_values()
    bootstrap_config_files()

    if args.setup:
        run_onboarding()
        sys.exit(0)

    # Handle --api-key before loading config, so it's available for provider lookup
    if args.api_key:
        _apply_api_key_override(args.api_key, args.model)

    try:
        initial_agent_name = get_initial_agent_name(args)
        config = load_config_or_exit()

        # Apply CLI overrides after config is loaded
        if args.model:
            config = _apply_model_override(config, args.model)

        if args.enabled_tools:
            config.enabled_tools = args.enabled_tools

        loaded_messages = load_session(args, config)

        stdin_prompt = get_prompt_from_stdin()
        if args.prompt is not None:
            programmatic_prompt = args.prompt or stdin_prompt
            if not programmatic_prompt:
                print(
                    "Error: No prompt provided for programmatic mode", file=sys.stderr
                )
                sys.exit(1)
            output_format = OutputFormat(
                args.output if hasattr(args, "output") else "text"
            )

            try:
                final_response = run_programmatic(
                    config=config,
                    prompt=programmatic_prompt,
                    max_turns=args.max_turns,
                    max_price=args.max_price,
                    output_format=output_format,
                    previous_messages=loaded_messages,
                    agent_name=initial_agent_name,
                )
                if final_response:
                    print(final_response)
                sys.exit(0)
            except ConversationLimitException as e:
                print(e, file=sys.stderr)
                sys.exit(1)
            except RuntimeError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            agent_loop = AgentLoop(
                config, agent_name=initial_agent_name, enable_streaming=True
            )

            if loaded_messages:
                _load_messages_from_previous_session(agent_loop, loaded_messages)

            run_textual_ui(
                agent_loop=agent_loop,
                initial_prompt=args.initial_prompt or stdin_prompt,
                teleport_on_start=args.teleport,
            )

    except (KeyboardInterrupt, EOFError):
        rprint("\n[dim]Bye![/]")
        sys.exit(0)
