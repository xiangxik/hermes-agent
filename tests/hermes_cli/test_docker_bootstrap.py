import hermes_cli.docker_bootstrap as docker_bootstrap


def test_no_docker_init_leaves_argv_unchanged():
    argv = ["chat", "-m", "opus"]
    forwarded, payload = docker_bootstrap.parse_docker_argv(argv)

    assert forwarded == argv
    assert payload is None


def test_parse_docker_init_with_delimiter_forwards_tail():
    argv = [
        "--docker-init",
        "--model",
        "gpt-4.1",
        "--provider",
        "openai",
        "--base-url",
        "https://example.com/v1",
        "--api-key",
        "secret",
        "--set",
        "display.streaming=true",
        "--set-env",
        "EXAMPLE_TOKEN=abc",
        "--append-toolset",
        "docker-tools",
        "--terminal-backend",
        "docker",
        "--terminal-cwd",
        "/workspace",
        "--yolo",
        "--",
        "chat",
        "-q",
        "hello",
    ]

    forwarded, payload = docker_bootstrap.parse_docker_argv(argv)

    assert forwarded == ["--yolo", "chat", "-q", "hello"]
    assert payload is not None
    assert payload.model == "gpt-4.1"
    assert payload.provider == "openai"
    assert payload.base_url == "https://example.com/v1"
    assert payload.api_key == "secret"
    assert payload.set_values == [("display.streaming", "true")]
    assert payload.set_env_values == [("EXAMPLE_TOKEN", "abc")]
    assert payload.append_toolsets == ["docker-tools"]
    assert payload.terminal_backend == "docker"
    assert payload.terminal_cwd == "/workspace"


def test_unknown_token_starts_forwarded_hermes_args():
    argv = ["--docker-init", "--model", "gpt-4", "chat", "-q", "hi"]

    forwarded, payload = docker_bootstrap.parse_docker_argv(argv)

    assert forwarded == ["chat", "-q", "hi"]
    assert payload is not None
    assert payload.model == "gpt-4"


def test_gateway_defaults_to_gateway_run_without_explicit_args():
    argv = ["--docker-init", "--gateway"]

    forwarded, payload = docker_bootstrap.parse_docker_argv(argv)

    assert forwarded == ["gateway", "run"]
    assert payload is not None
    assert payload.gateway is True


def test_yolo_is_forwarded_not_persisted(monkeypatch):
    saved = {}

    monkeypatch.setattr(docker_bootstrap, "ensure_hermes_home", lambda: None)
    monkeypatch.setattr(docker_bootstrap, "load_config", lambda: {})
    monkeypatch.setattr(docker_bootstrap, "save_env_value", lambda key, value: None)

    def fake_save_config(config):
        saved.update(config)

    monkeypatch.setattr(docker_bootstrap, "save_config", fake_save_config)

    forwarded, payload = docker_bootstrap.bootstrap_orchestrate(
        ["--docker-init", "--yolo", "chat"]
    )

    assert forwarded == ["--yolo", "chat"]
    assert payload is not None
    assert "yolo" not in saved


def test_apply_bootstrap_settings_persists_real_schema(monkeypatch):
    saved = {}
    env_saved = {}

    monkeypatch.setattr(docker_bootstrap, "ensure_hermes_home", lambda: None)
    monkeypatch.setattr(
        docker_bootstrap,
        "load_config",
        lambda: {
            "toolsets": ["hermes-cli"],
            "terminal": {"backend": "local", "cwd": "."},
        },
    )

    def fake_save_config(config):
        saved.clear()
        saved.update(config)

    def fake_save_env_value(key, value):
        env_saved[key] = value

    monkeypatch.setattr(docker_bootstrap, "save_config", fake_save_config)
    monkeypatch.setattr(docker_bootstrap, "save_env_value", fake_save_env_value)

    payload = docker_bootstrap.DockerBootstrapPayload(
        model="gpt-4.1",
        provider="openai",
        base_url="https://example.com/v1",
        api_key="secret",
        set_values=[("display.streaming", "true"), ("approvals.mode", "off")],
        set_env_values=[("EXAMPLE_TOKEN", "abc")],
        append_toolsets=["docker-tools", "hermes-cli"],
        terminal_backend="docker",
        terminal_cwd="/workspace",
    )

    docker_bootstrap.apply_bootstrap_settings(payload)

    assert saved["model"]["default"] == "gpt-4.1"
    assert saved["model"]["provider"] == "openai"
    assert saved["model"]["base_url"] == "https://example.com/v1"
    assert saved["terminal"]["backend"] == "docker"
    assert saved["terminal"]["cwd"] == "/workspace"
    assert saved["display"]["streaming"] is True
    assert saved["approvals"]["mode"] == "off"
    assert saved["toolsets"] == ["hermes-cli", "docker-tools"]
    assert env_saved["OPENAI_API_KEY"] == "secret"
    assert env_saved["EXAMPLE_TOKEN"] == "abc"


def test_switching_provider_clears_stale_base_url_and_api_mode(monkeypatch):
    saved = {}

    monkeypatch.setattr(docker_bootstrap, "ensure_hermes_home", lambda: None)
    monkeypatch.setattr(
        docker_bootstrap,
        "load_config",
        lambda: {
            "model": {
                "provider": "openrouter",
                "default": "openai/gpt-4.1",
                "base_url": "https://openrouter.ai/api/v1",
                "api_mode": "chat_completions",
            }
        },
    )
    monkeypatch.setattr(docker_bootstrap, "save_env_value", lambda key, value: None)

    def fake_save_config(config):
        saved.clear()
        saved.update(config)

    monkeypatch.setattr(docker_bootstrap, "save_config", fake_save_config)

    payload = docker_bootstrap.DockerBootstrapPayload(
        provider="minimax",
        model="MiniMax-M2.7",
    )

    docker_bootstrap.apply_bootstrap_settings(payload)

    assert saved["model"]["provider"] == "minimax"
    assert saved["model"]["default"] == "MiniMax-M2.7"
    assert "base_url" not in saved["model"]
    assert "api_mode" not in saved["model"]


def test_explicit_base_url_is_preserved_when_switching_provider(monkeypatch):
    saved = {}

    monkeypatch.setattr(docker_bootstrap, "ensure_hermes_home", lambda: None)
    monkeypatch.setattr(
        docker_bootstrap,
        "load_config",
        lambda: {
            "model": {
                "provider": "openrouter",
                "default": "openai/gpt-4.1",
                "base_url": "https://openrouter.ai/api/v1",
                "api_mode": "chat_completions",
            }
        },
    )
    monkeypatch.setattr(docker_bootstrap, "save_env_value", lambda key, value: None)

    def fake_save_config(config):
        saved.clear()
        saved.update(config)

    monkeypatch.setattr(docker_bootstrap, "save_config", fake_save_config)

    payload = docker_bootstrap.DockerBootstrapPayload(
        provider="minimax",
        model="MiniMax-M2.7",
        base_url="https://api.minimax.io/anthropic",
    )

    docker_bootstrap.apply_bootstrap_settings(payload)

    assert saved["model"]["provider"] == "minimax"
    assert saved["model"]["base_url"] == "https://api.minimax.io/anthropic"


def test_bootstrap_disabled_skips_persistence(monkeypatch):
    called = {"apply": False}

    monkeypatch.setenv(docker_bootstrap.DISABLE_ENV_VAR, "1")

    def fake_apply(_payload):
        called["apply"] = True

    monkeypatch.setattr(docker_bootstrap, "apply_bootstrap_settings", fake_apply)

    forwarded, payload = docker_bootstrap.bootstrap_orchestrate(
        ["--docker-init", "--model", "gpt-4", "chat"]
    )

    assert forwarded == ["chat"]
    assert payload is None
    assert called["apply"] is False
