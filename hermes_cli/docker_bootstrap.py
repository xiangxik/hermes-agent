"""Docker-only bootstrap helpers for Hermes containers.

This module is invoked by the Docker entrypoint when the container command
contains ``--docker-init``. It consumes Docker-only initialization flags,
persists the requested settings into ``HERMES_HOME/config.yaml`` and
``HERMES_HOME/.env`` using Hermes' existing config helpers, and returns the
Hermes CLI arguments that should be executed afterward.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from hermes_cli.config import ensure_hermes_home, load_config, save_config, save_env_value


DISABLE_ENV_VAR = "HERMES_DOCKER_BOOTSTRAP_DISABLED"


@dataclass
class DockerBootstrapPayload:
    model: Optional[str] = None
    provider: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    set_values: List[Tuple[str, str]] = field(default_factory=list)
    set_env_values: List[Tuple[str, str]] = field(default_factory=list)
    append_toolsets: List[str] = field(default_factory=list)
    terminal_backend: Optional[str] = None
    terminal_cwd: Optional[str] = None
    gateway: Optional[bool] = None
    yolo: Optional[bool] = None


_PROVIDERS_THAT_OWN_BASE_URL = {
    "openai",
    "openrouter",
    "anthropic",
    "deepseek",
    "minimax",
    "minimax-cn",
    "qwen",
    "zai",
    "kimi-coding",
    "kimi-coding-cn",
    "alibaba",
    "xai",
    "arcee",
    "ai-gateway",
    "kilocode",
    "nvidia",
    "mistral",
    "gemini",
    "openai-codex",
    "copilot",
    "copilot-acp",
    "ollama-cloud",
    "qwen-oauth",
    "google-gemini-cli",
}


def is_bootstrap_disabled() -> bool:
    return os.environ.get(DISABLE_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def _coerce_scalar(value: str):
    lowered = value.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        try:
            return int(value)
        except ValueError:
            pass
    try:
        if "." in value:
            return float(value)
    except ValueError:
        pass
    return value


def _set_nested(config: Dict, dotted_key: str, value) -> None:
    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        return
    current = config
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def parse_docker_argv(argv: List[str]) -> Tuple[List[str], Optional[DockerBootstrapPayload]]:
    if "--docker-init" not in argv:
        return list(argv), None

    idx = argv.index("--docker-init")
    docker_args = argv[idx + 1 :]
    payload = DockerBootstrapPayload()
    forwarded: List[str] = []
    i = 0

    while i < len(docker_args):
        token = docker_args[i]

        if token == "--":
            forwarded = docker_args[i + 1 :]
            break

        if token == "--model" and i + 1 < len(docker_args):
            payload.model = docker_args[i + 1]
            i += 2
            continue
        if token == "--provider" and i + 1 < len(docker_args):
            payload.provider = docker_args[i + 1]
            i += 2
            continue
        if token == "--base-url" and i + 1 < len(docker_args):
            payload.base_url = docker_args[i + 1]
            i += 2
            continue
        if token == "--api-key" and i + 1 < len(docker_args):
            payload.api_key = docker_args[i + 1]
            i += 2
            continue
        if token == "--set" and i + 1 < len(docker_args):
            item = docker_args[i + 1]
            if "=" in item:
                key, value = item.split("=", 1)
                payload.set_values.append((key, value))
            i += 2
            continue
        if token == "--set-env" and i + 1 < len(docker_args):
            item = docker_args[i + 1]
            if "=" in item:
                key, value = item.split("=", 1)
                payload.set_env_values.append((key, value))
            i += 2
            continue
        if token == "--append-toolset" and i + 1 < len(docker_args):
            payload.append_toolsets.append(docker_args[i + 1])
            i += 2
            continue
        if token == "--terminal-backend" and i + 1 < len(docker_args):
            payload.terminal_backend = docker_args[i + 1]
            i += 2
            continue
        if token == "--terminal-cwd" and i + 1 < len(docker_args):
            payload.terminal_cwd = docker_args[i + 1]
            i += 2
            continue
        if token == "--gateway":
            payload.gateway = True
            i += 1
            continue
        if token == "--no-gateway":
            payload.gateway = False
            i += 1
            continue
        if token == "--yolo":
            payload.yolo = True
            i += 1
            continue
        if token == "--no-yolo":
            payload.yolo = False
            i += 1
            continue
        forwarded = docker_args[i:]
        break

    forwarded_args: List[str] = []
    if payload.yolo is True:
        forwarded_args.append("--yolo")
    if payload.gateway is True and not forwarded:
        forwarded_args.extend(["gateway", "run"])
    forwarded_args.extend(forwarded)

    return forwarded_args, payload


def apply_bootstrap_settings(payload: DockerBootstrapPayload) -> None:
    ensure_hermes_home()
    config = load_config()
    if not isinstance(config, dict):
        config = {}

    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        model_cfg = {}
        config["model"] = model_cfg

    terminal_cfg = config.get("terminal")
    if not isinstance(terminal_cfg, dict):
        terminal_cfg = {}
        config["terminal"] = terminal_cfg

    if payload.model is not None:
        model_cfg["default"] = payload.model
    if payload.provider is not None:
        normalized_provider = payload.provider.strip().lower()
        previous_provider = str(model_cfg.get("provider") or "").strip().lower()
        model_cfg["provider"] = payload.provider
        if payload.base_url is None and (
            previous_provider != normalized_provider
            or normalized_provider in _PROVIDERS_THAT_OWN_BASE_URL
        ):
            model_cfg.pop("base_url", None)
            model_cfg.pop("api_mode", None)
    if payload.base_url is not None:
        model_cfg["base_url"] = payload.base_url

    for key, value in payload.set_values:
        _set_nested(config, key, _coerce_scalar(value))

    if payload.append_toolsets:
        toolsets = config.get("toolsets")
        if not isinstance(toolsets, list):
            toolsets = []
        for toolset in payload.append_toolsets:
            if toolset not in toolsets:
                toolsets.append(toolset)
        config["toolsets"] = toolsets

    if payload.terminal_backend is not None:
        terminal_cfg["backend"] = payload.terminal_backend
    if payload.terminal_cwd is not None:
        terminal_cfg["cwd"] = payload.terminal_cwd

    save_config(config)

    if payload.api_key is not None:
        save_env_value("OPENAI_API_KEY", payload.api_key)
    for key, value in payload.set_env_values:
        save_env_value(key, value)


def bootstrap_orchestrate(argv: List[str]) -> Tuple[List[str], Optional[DockerBootstrapPayload]]:
    forwarded_args, payload = parse_docker_argv(argv)
    if payload is None:
        return forwarded_args, None
    if is_bootstrap_disabled():
        return forwarded_args, None
    apply_bootstrap_settings(payload)
    return forwarded_args, payload


def _write_forwarded_args_file(forwarded_args: List[str]) -> str:
    with tempfile.NamedTemporaryFile(
        delete=False,
        mode="w",
        encoding="utf-8",
        prefix="hermes_bootstrap_",
        suffix=".args",
    ) as handle:
        for arg in forwarded_args:
            handle.write(arg + "\n")
        return handle.name


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if "--docker-init" not in argv:
        return 0
    forwarded_args, _payload = bootstrap_orchestrate(argv)
    print(_write_forwarded_args_file(forwarded_args))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
