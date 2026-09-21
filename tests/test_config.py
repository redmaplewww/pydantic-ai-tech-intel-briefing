from search_assistant.config import Settings


def test_settings_loads_defaults_for_local_mode(tmp_path):
    settings = Settings.from_env({"SEARCH_ASSISTANT_DATA_DIR": str(tmp_path)})

    assert settings.data_dir == tmp_path
    assert settings.database_path == tmp_path / "assistant.sqlite3"
    assert settings.feishu_enabled is False
    assert settings.model_provider == "glm"
    assert settings.glm_model == "glm-4.7"
    assert settings.glm_timeout_seconds == 120.0
    assert settings.briefing_model_max_sources == 12
    assert settings.briefing_timezone == "Asia/Shanghai"
    assert settings.deepseek_model == "deepseek-v4-flash"
    assert settings.search_provider == "hybrid"
    assert settings.mcp_search_config_path is None
    assert settings.mcp_search_timeout_seconds == 18.0
    assert settings.agent_reach_enabled is False
    assert settings.agent_reach_command == "agent-reach"
    assert settings.agent_reach_timeout_seconds == 30.0
    assert settings.agent_reach_doctor_cache_seconds == 300.0
    assert settings.domestic_rss_base_url == "http://127.0.0.1:1200"
    assert settings.domestic_rss_config_path is None
    assert settings.domestic_rss_timeout_seconds == 8.0
    assert settings.briefing_search_budget_seconds == 90.0
    assert settings.briefing_planning_timeout_seconds == 40.0
    assert settings.briefing_max_queries == 28
    assert settings.bilibili_search_base_url == "https://api.bilibili.com/x/web-interface/search/type"
    assert settings.duckduckgo_timeout_seconds == 12.0
    assert settings.browser_search_engines == ["bing", "baidu", "google"]
    assert settings.browser_search_base_url == "https://cn.bing.com/search"
    assert settings.baidu_search_base_url == "https://www.baidu.com/baidu"


def test_settings_loads_domestic_rss_configuration(tmp_path):
    catalog_path = tmp_path / "domestic-rss.sources.json"

    settings = Settings.from_env(
        {
            "SEARCH_ASSISTANT_DATA_DIR": str(tmp_path),
            "RSSHUB_BASE_URL": "http://rsshub.test:1200",
            "SEARCH_ASSISTANT_DOMESTIC_RSS_CONFIG": str(catalog_path),
            "DOMESTIC_RSS_TIMEOUT_SECONDS": "11",
        }
    )

    assert settings.domestic_rss_base_url == "http://rsshub.test:1200"
    assert settings.domestic_rss_config_path == catalog_path
    assert settings.domestic_rss_timeout_seconds == 11.0


def test_settings_loads_agent_reach_configuration(tmp_path):
    settings = Settings.from_env(
        {
            "SEARCH_ASSISTANT_DATA_DIR": str(tmp_path),
            "SEARCH_ASSISTANT_SEARCH_PROVIDER": "agent-reach",
            "SEARCH_ASSISTANT_AGENT_REACH_ENABLED": "true",
            "AGENT_REACH_COMMAND": "agent-reach-custom",
            "AGENT_REACH_TIMEOUT_SECONDS": "17",
            "AGENT_REACH_DOCTOR_CACHE_SECONDS": "31",
        }
    )

    assert settings.search_provider == "agent-reach"
    assert settings.agent_reach_enabled is True
    assert settings.agent_reach_command == "agent-reach-custom"
    assert settings.agent_reach_timeout_seconds == 17.0
    assert settings.agent_reach_doctor_cache_seconds == 31.0


def test_settings_loads_env_local_when_no_mapping_is_provided(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("FEISHU_APP_ID", raising=False)
    (tmp_path / ".env.local").write_text(
        "\n".join(
            [
                "DEEPSEEK_API_KEY=from-file",
                "FEISHU_APP_ID=cli_from_file",
                "BROWSER_SEARCH_ENGINES=bing,baidu,google",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings.from_env()

    assert settings.deepseek_api_key == "from-file"
    assert settings.feishu_app_id == "cli_from_file"
    assert settings.browser_search_engines == ["bing", "baidu", "google"]


def test_environment_variables_override_env_local(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")
    (tmp_path / ".env.local").write_text("DEEPSEEK_API_KEY=from-file\n", encoding="utf-8")

    settings = Settings.from_env()

    assert settings.deepseek_api_key == "from-env"


def test_env_local_parser_ignores_comments_and_strips_matching_quotes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("FEISHU_APP_SECRET", raising=False)
    (tmp_path / ".env.local").write_text(
        "\ufeff# local secrets\n"
        "DEEPSEEK_API_KEY='quoted-key'\n"
        'FEISHU_APP_SECRET="quoted-secret"\n'
        "\n"
        "IGNORED_LINE_WITHOUT_EQUALS\n",
        encoding="utf-8",
    )

    settings = Settings.from_env()

    assert settings.deepseek_api_key == "quoted-key"
    assert settings.feishu_app_secret == "quoted-secret"


def test_explicit_env_mapping_does_not_read_env_local(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.local").write_text("DEEPSEEK_API_KEY=from-file\n", encoding="utf-8")

    settings = Settings.from_env({"SEARCH_ASSISTANT_DATA_DIR": str(tmp_path)})

    assert settings.deepseek_api_key is None
