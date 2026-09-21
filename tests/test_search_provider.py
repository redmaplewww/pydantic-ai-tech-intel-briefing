import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pytest

from search_assistant.config import Settings
from search_assistant.search.provider import (
    AgentReachSearchClient,
    BaiduBrowserSearchClient,
    BilibiliPublicSearchClient,
    BingBrowserSearchClient,
    BrowserSearchClient,
    BraveSearchClient,
    CompositeSearchClient,
    DuckDuckGoSearchClient,
    FakeSearchClient,
    GoogleBrowserSearchClient,
    McpSearchClient,
    SearxngSearchClient,
    SearchProviderError,
    SearchResult,
    ToutiaoPublicSearchClient,
    YouTubePublicSearchClient,
    search_client_from_settings,
    search_with_provider_events,
)


def test_brave_search_client_extracts_web_results():
    calls = []

    def transport(url, headers):
        calls.append({"url": url, "headers": headers})
        return json.dumps(
            {
                "web": {
                    "results": [
                        {
                            "title": "DeepSeek API Docs",
                            "url": "https://api-docs.deepseek.com/",
                            "description": "DeepSeek API uses OpenAI-compatible chat completions.",
                        }
                    ]
                }
            }
        )

    client = BraveSearchClient(api_key="test-search-key", transport=transport)

    results = client.search("DeepSeek API", limit=1)

    assert results[0].title == "DeepSeek API Docs"
    assert results[0].url == "https://api-docs.deepseek.com/"
    assert results[0].snippet == "DeepSeek API uses OpenAI-compatible chat completions."
    assert results[0].provider == "brave"
    assert "q=DeepSeek+API" in calls[0]["url"]
    assert calls[0]["headers"]["X-Subscription-Token"] == "test-search-key"


def test_duckduckgo_search_parser_extracts_results_from_html():
    html = """
    <div class="result">
      <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fdoc&amp;rut=abc">
        Example Docs
      </a>
      <a class="result__snippet">A concise snippet about the API.</a>
    </div>
    """

    results = DuckDuckGoSearchClient.parse_results(html, provider="duckduckgo", checked_at="2026-06-27T00:00:00Z")

    assert len(results) == 1
    assert results[0].title == "Example Docs"
    assert results[0].url == "https://example.com/doc"
    assert results[0].snippet == "A concise snippet about the API."
    assert results[0].provider == "duckduckgo"


def test_mcp_search_client_calls_only_configured_search_tool_and_preserves_urls(tmp_path):
    config_path = tmp_path / "search.mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {"public": {"url": "https://mcp.example.test/search"}},
                "searchBindings": {
                    "public": {
                        "tool": "search_public_sources",
                        "arguments": {"query": "{query}", "limit": "{limit}"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    toolset = RecordingMcpToolset(
        [
            {
                "title": "Industrial AI repository",
                "url": "https://github.com/example/industrial-ai",
                "snippet": "A public repository.",
                "provider": "github-api",
            },
            {"title": "Ignored", "url": "not-a-url"},
        ]
    )
    client = McpSearchClient(
        config_path,
        toolset_factory=lambda name, server, directory, timeout: toolset,
    )

    results = client.search("industrial AI", limit=3)

    assert [(result.title, result.url, result.provider) for result in results] == [
        (
            "Industrial AI repository",
            "https://github.com/example/industrial-ai",
            "mcp:public:github-api",
        )
    ]
    assert toolset.calls == [("search_public_sources", {"query": "industrial AI", "limit": 3})]


def test_mcp_search_client_does_not_fake_platform_coverage_for_site_queries(tmp_path):
    config_path = tmp_path / "search.mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {"public": {"url": "https://mcp.example.test/search"}},
                "searchBindings": {"public": {"tool": "search"}},
            }
        ),
        encoding="utf-8",
    )
    toolset = RecordingMcpToolset([])
    client = McpSearchClient(
        config_path,
        toolset_factory=lambda name, server, directory, timeout: toolset,
    )

    assert client.search("site:reddit.com industrial AI") == []
    assert toolset.calls == []


def test_mcp_search_client_allows_only_explicit_platform_bindings_for_site_queries(tmp_path):
    config_path = tmp_path / "search.mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {"domestic": {"url": "https://mcp.example.test/search"}},
                "searchBindings": {
                    "domestic": {
                        "tool": "search_domestic_rss",
                        "siteDomains": ["weibo.com"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    toolset = RecordingMcpToolset(
        [
            {"title": "Weibo", "url": "https://m.weibo.com/status/1", "snippet": "Industrial AI"},
            {"title": "Wrong domain", "url": "https://example.com/1", "snippet": "Industrial AI"},
        ]
    )
    client = McpSearchClient(
        config_path,
        toolset_factory=lambda name, server, directory, timeout: toolset,
    )

    results = client.search("site:weibo.com industrial AI", limit=3)

    assert [result.url for result in results] == ["https://m.weibo.com/status/1"]
    assert toolset.calls == [("search_domestic_rss", {"query": "site:weibo.com industrial AI", "limit": 3})]


def test_mcp_search_client_passes_runtime_environment_to_stdio_server(tmp_path):
    config_path = tmp_path / "search.mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {"domestic": {"command": "python", "args": ["server.py"]}},
                "searchBindings": {"domestic": {"tool": "search"}},
            }
        ),
        encoding="utf-8",
    )
    captured_servers = []

    def factory(name, server, directory, timeout):
        captured_servers.append(server)
        return RecordingMcpToolset([])

    McpSearchClient(
        config_path,
        toolset_factory=factory,
        server_environment={
            "RSSHUB_BASE_URL": "http://rsshub.test:1200",
            "SEARCH_ASSISTANT_DOMESTIC_RSS_CONFIG": "C:/catalog.json",
        },
    )

    assert captured_servers[0]["env"] == {
        "RSSHUB_BASE_URL": "http://rsshub.test:1200",
        "SEARCH_ASSISTANT_DOMESTIC_RSS_CONFIG": "C:/catalog.json",
    }


def test_mcp_search_client_serializes_threaded_calls_for_a_shared_toolset(tmp_path):
    config_path = tmp_path / "search.mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {"public": {"url": "https://mcp.example.test/search"}},
                "searchBindings": {"public": {"tool": "search"}},
            }
        ),
        encoding="utf-8",
    )
    toolset = ConcurrentMcpToolset()
    client = McpSearchClient(
        config_path,
        toolset_factory=lambda name, server, directory, timeout: toolset,
    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        result_sets = list(executor.map(lambda index: client.search(f"industrial AI {index}", 1), range(4)))

    assert [len(results) for results in result_sets] == [1, 1, 1, 1]
    assert toolset.max_active_calls == 1


def test_composite_search_client_keeps_mcp_results_and_deduplicates_fallback_urls():
    mcp_result = SearchResult(
        title="MCP result",
        url="https://example.com/mcp",
        snippet="",
        provider="mcp:public:search",
        checked_at="2026-07-25T00:00:00Z",
    )
    fallback_duplicate = mcp_result.model_copy(update={"provider": "browser-bing"})
    browser_result = mcp_result.model_copy(
        update={"title": "Browser result", "url": "https://example.com/browser", "provider": "browser-bing"}
    )
    client = CompositeSearchClient(
        [StaticSearchClient([mcp_result]), StaticSearchClient([fallback_duplicate, browser_result])]
    )

    results = client.search("industrial AI", limit=5)

    assert [result.url for result in results] == ["https://example.com/mcp", "https://example.com/browser"]


def test_composite_search_client_does_not_backfill_when_primary_has_sufficient_results():
    primary = StaticSearchClient(
        [
            SearchResult(title="A", url="https://example.com/a", provider="mcp:public:search", checked_at="2026-07-25T00:00:00Z"),
            SearchResult(title="B", url="https://example.com/b", provider="mcp:public:search", checked_at="2026-07-25T00:00:00Z"),
            SearchResult(title="C", url="https://example.com/c", provider="mcp:public:search", checked_at="2026-07-25T00:00:00Z"),
        ]
    )
    fallback = StaticSearchClient(
        [SearchResult(title="Fallback", url="https://example.com/fallback", provider="browser-bing", checked_at="2026-07-25T00:00:00Z")]
    )
    client = CompositeSearchClient([primary, fallback], primary_sufficient_results=3)

    results = client.search("industrial AI", limit=8)

    assert [result.url for result in results] == ["https://example.com/a", "https://example.com/b", "https://example.com/c"]


def test_agent_reach_search_client_runs_doctor_then_exa_route():
    calls = []

    def runner(command, timeout):
        calls.append(command)
        if command == ["agent-reach", "doctor", "--json"]:
            return json.dumps(
                {
                    "exa_search": {
                        "status": "warn",
                        "message": "Exa is configured but not live-probed.",
                        "active_backend": None,
                    }
                }
            )
        assert command == [
            "mcporter",
            "call",
            "exa.web_search_exa",
            "query=Agent Reach",
            "numResults=2",
        ]
        return json.dumps(
            {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            [
                                {
                                    "title": "Agent Reach repository",
                                    "url": "https://github.com/Panniantong/Agent-Reach",
                                    "snippet": "CLI capability router for agents.",
                                }
                            ]
                        ),
                    }
                ]
            }
        )

    client = AgentReachSearchClient(command_runner=runner)

    results = client.search("Agent Reach", limit=2)

    assert [call[0] for call in calls] == ["agent-reach", "mcporter"]
    assert results[0].title == "Agent Reach repository"
    assert results[0].url == "https://github.com/Panniantong/Agent-Reach"
    assert results[0].provider == "agent-reach:exa_search:mcporter"


def test_agent_reach_search_client_uses_bilibili_backend_for_scoped_query():
    calls = []

    def runner(command, timeout):
        calls.append(command)
        if command == ["agent-reach", "doctor", "--json"]:
            return json.dumps(
                {
                    "bilibili": {
                        "status": "ok",
                        "message": "bili-cli is available.",
                        "active_backend": "bili-cli",
                    },
                    "exa_search": {"status": "off", "message": "not configured", "active_backend": None},
                }
            )
        assert command == ["bili", "search", "AI 教程", "--type", "video", "-n", "1"]
        return "AI 教程入门 BV1test123"

    client = AgentReachSearchClient(command_runner=runner)

    results = client.search("site:bilibili.com AI 教程", limit=1)

    assert [call[0] for call in calls] == ["agent-reach", "bili"]
    assert results[0].url == "https://www.bilibili.com/video/BV1test123"
    assert results[0].provider == "agent-reach:bilibili:bili-cli"


def test_agent_reach_search_client_reports_missing_route_from_doctor():
    client = AgentReachSearchClient(
        command_runner=lambda command, timeout: json.dumps(
            {
                "exa_search": {"status": "off", "message": "mcporter missing", "active_backend": None},
            }
        )
    )

    with pytest.raises(SearchProviderError, match="Agent Reach has no usable route"):
        client.search("Agent Reach", limit=1)
def test_search_with_provider_events_records_errors_and_fallback_success():
    fallback_result = SearchResult(
        title="Fallback result",
        url="https://example.com/fallback",
        snippet="Fallback source.",
        provider="browser-bing",
        checked_at="2026-07-25T00:00:00Z",
    )
    client = CompositeSearchClient([FailingSearchClient(), StaticSearchClient([fallback_result])])

    outcome = search_with_provider_events(client, "industrial AI", limit=3)

    assert [result.url for result in outcome.results] == ["https://example.com/fallback"]
    assert [event.status for event in outcome.provider_events] == ["error", "success"]
    assert outcome.provider_events[0].provider == "FailingSearchClient"
    assert "blocked" in outcome.provider_events[0].error


def test_searxng_search_client_extracts_json_results_without_an_api_key():
    calls = []

    def transport(url, headers):
        calls.append({"url": url, "headers": headers})
        return json.dumps(
            {
                "results": [
                    {
                        "title": "Industrial AI architecture",
                        "url": "https://example.com/industrial-ai",
                        "content": "Agent orchestration connects MES and ERP workflows.",
                    }
                ]
            }
        )

    client = SearxngSearchClient(base_url="http://searxng.local/search", transport=transport)
    results = client.search("industrial AI", limit=1)

    assert results[0].title == "Industrial AI architecture"
    assert results[0].url == "https://example.com/industrial-ai"
    assert results[0].provider == "searxng"
    assert "format=json" in calls[0]["url"]
    assert calls[0]["headers"]["Accept"] == "application/json"


def test_bilibili_public_search_client_extracts_public_video_results_without_login():
    calls = []

    def transport(url, headers):
        calls.append({"url": url, "headers": headers})
        return json.dumps(
            {
                "code": 0,
                "data": {
                    "result": [
                        {
                            "title": "<em class=\"keyword\">工业 AI</em> 智能制造",
                            "arcurl": "http://www.bilibili.com/video/BV1test123",
                            "description": "MES 与设备数据闭环。",
                            "author": "工业研究院",
                        }
                    ]
                },
            }
        )

    client = BilibiliPublicSearchClient(transport=transport)

    results = client.search("site:bilibili.com AI+工业界发展", limit=1)

    assert results[0].title == "工业 AI 智能制造"
    assert results[0].url == "https://www.bilibili.com/video/BV1test123"
    assert results[0].snippet == "MES 与设备数据闭环。 作者：工业研究院"
    assert results[0].provider == "bilibili-public-api"
    assert "search_type=video" in calls[0]["url"]
    assert "site%3Abilibili.com" not in calls[0]["url"]
    assert calls[0]["headers"]["Referer"] == "https://search.bilibili.com/"


def test_bilibili_public_search_client_retries_with_browser_headers_after_anti_bot_error():
    calls = []

    def transport(url, headers):
        calls.append({"url": url, "headers": headers})
        if len(calls) == 1:
            raise SearchProviderError("Search request failed: HTTP Error 412: Precondition Failed")
        return json.dumps(
            {
                "code": 0,
                "data": {
                    "result": [
                        {
                            "title": "AI industry video",
                            "bvid": "BV1retry123",
                            "description": "Public Bilibili result.",
                        }
                    ]
                },
            }
        )

    client = BilibiliPublicSearchClient(transport=transport)

    results = client.search("site:bilibili.com AI industry", limit=1)

    assert results[0].url == "https://www.bilibili.com/video/BV1retry123"
    assert len(calls) == 2
    assert "Windows NT" in calls[1]["headers"]["User-Agent"]


def test_toutiao_public_search_client_extracts_original_group_urls_without_detail_fetch():
    calls = []

    def transport(url, headers):
        calls.append({"url": url, "headers": headers})
        return """
        <script>
        window.__DATA__={"display":[
          {"open_url":"https://toutiao.com/group/7662517064405451274/?source=search_tab",
           "title":"我国人工智能产业发展观察",
           "abstract":"公开搜索摘要提到产业链和应用落地。"},
          {"open_url":"https://article.zlink.toutiao.com/J4dQM?alert=0",
           "title":"跳转链接不应保留"},
          {"open_url":"https://example.com/off-domain",
           "title":"ignored"}
        ]};
        </script>
        """

    client = ToutiaoPublicSearchClient(transport=transport)

    results = client.search("site:toutiao.com 人工智能产业发展", limit=3)

    assert [result.url for result in results] == ["https://toutiao.com/group/7662517064405451274"]
    assert results[0].title == "我国人工智能产业发展观察"
    assert results[0].snippet == "公开搜索摘要提到产业链和应用落地。"
    assert results[0].provider == "toutiao-public-search"
    assert "keyword=%E4%BA%BA%E5%B7%A5" in calls[0]["url"]
    assert "site%3Atoutiao.com" not in calls[0]["url"]


def test_youtube_public_search_client_extracts_watch_urls_without_api_key():
    calls = []

    def transport(url, headers):
        calls.append({"url": url, "headers": headers})
        return (
            '{"videoRenderer":{"videoId":"abc123xyz00","thumbnail":{},'
            '"title":{"runs":[{"text":"AI industrial development keynote"}]}}}'
        )

    client = YouTubePublicSearchClient(transport=transport)

    results = client.search("site:youtube.com AI industrial development", limit=3)

    assert [result.url for result in results] == ["https://www.youtube.com/watch?v=abc123xyz00"]
    assert results[0].title == "AI industrial development keynote"
    assert results[0].provider == "youtube-public-search"
    assert "search_query=AI+industrial+development" in calls[0]["url"]
    assert "site%3Ayoutube.com" not in calls[0]["url"]


def test_browser_search_client_rejects_off_domain_results_for_site_queries():
    client = BrowserSearchClient(
        [
            StaticSearchClient(
                [
                    SearchResult(
                        title="Ignored",
                        url="https://example.com/unrelated",
                        snippet="The upstream engine ignored the site filter.",
                        provider="browser-bing",
                        checked_at="2026-07-24T00:00:00Z",
                    )
                ]
            )
        ]
    )

    results = client.search("site:reddit.com industrial AI", limit=5)

    assert results == []


def test_browser_search_client_uses_direct_scoped_client_before_html_engines():
    client = BrowserSearchClient(
        [
            StaticSearchClient(
                [
                    SearchResult(
                        title="Wrong upstream page",
                        url="https://example.com/unrelated",
                        snippet="",
                        provider="browser-bing",
                        checked_at="2026-07-24T00:00:00Z",
                    )
                ]
            )
        ],
        scoped_search_clients={
            "bilibili.com": StaticSearchClient(
                [
                    SearchResult(
                        title="Industrial AI forum",
                        url="https://www.bilibili.com/video/BV1valid",
                        snippet="Public API result.",
                        provider="bilibili-public-api",
                        checked_at="2026-07-24T00:00:00Z",
                    )
                ]
            )
        },
    )

    results = client.search("site:bilibili.com industrial AI", limit=5)

    assert [result.url for result in results] == ["https://www.bilibili.com/video/BV1valid"]
    assert results[0].provider == "bilibili-public-api"


def test_browser_search_client_uses_public_scoped_clients_for_toutiao_and_youtube():
    client = BrowserSearchClient(
        [StaticSearchClient([])],
        scoped_search_clients={
            "toutiao.com": StaticSearchClient(
                [
                    SearchResult(
                        title="Toutiao public result",
                        url="https://toutiao.com/group/1",
                        snippet="",
                        provider="toutiao-public-search",
                        checked_at="2026-07-24T00:00:00Z",
                    )
                ]
            ),
            "youtube.com": StaticSearchClient(
                [
                    SearchResult(
                        title="YouTube public result",
                        url="https://www.youtube.com/watch?v=1",
                        snippet="",
                        provider="youtube-public-search",
                        checked_at="2026-07-24T00:00:00Z",
                    )
                ]
            ),
        },
    )

    toutiao_results = client.search("site:toutiao.com industrial AI", limit=5)
    youtube_results = client.search("site:youtube.com industrial AI", limit=5)

    assert [result.provider for result in toutiao_results] == ["toutiao-public-search"]
    assert [result.provider for result in youtube_results] == ["youtube-public-search"]


def test_bing_browser_search_parser_extracts_results_from_html():
    html = """
    <li class="b_algo">
      <h2>
        <a href="https://www.bing.com/ck/a?!&amp;&amp;u=a1aHR0cHM6Ly9hcGktZG9jcy5kZWVwc2Vlay5jb20v&amp;ntb=1">
          DeepSeek API Docs
        </a>
      </h2>
      <div class="b_caption"><p>OpenAI-compatible chat completions documentation.</p></div>
    </li>
    """

    results = BingBrowserSearchClient.parse_results(html, checked_at="2026-06-27T00:00:00Z")

    assert len(results) == 1
    assert results[0].title == "DeepSeek API Docs"
    assert results[0].url == "https://api-docs.deepseek.com/"
    assert results[0].snippet == "OpenAI-compatible chat completions documentation."
    assert results[0].provider == "browser-bing"


def test_baidu_browser_search_parser_extracts_results_from_html():
    html = """
    <div class="result c-container">
      <h3 class="t"><a href="https://example.com/feishu">飞书开放平台文档</a></h3>
      <div class="c-abstract">事件订阅和机器人消息 API 说明。</div>
    </div>
    """

    results = BaiduBrowserSearchClient.parse_results(html, checked_at="2026-06-27T00:00:00Z")

    assert len(results) == 1
    assert results[0].title == "飞书开放平台文档"
    assert results[0].url == "https://example.com/feishu"
    assert results[0].snippet == "事件订阅和机器人消息 API 说明。"
    assert results[0].provider == "browser-baidu"


def test_baidu_browser_search_parser_prefers_original_mu_target_over_click_redirect():
    html = """
    <div class="result c-container" mu="https://mp.weixin.qq.com/s/example-original">
      <h3 class="t"><a href="https://www.baidu.com/link?url=opaque">Public account article</a></h3>
      <div class="c-abstract">Baidu result-page excerpt.</div>
    </div>
    """

    results = BaiduBrowserSearchClient.parse_results(html, checked_at="2026-07-26T00:00:00Z")

    assert len(results) == 1
    assert results[0].url == "https://mp.weixin.qq.com/s/example-original"
    assert results[0].snippet == "Baidu result-page excerpt."


def test_baidu_browser_search_parser_extracts_mobile_original_url_and_discards_baidu_urls():
    html = """
    <div data-state="&quot;dataUrl&quot;:&quot;https://m.baidu.com/link?opaque=1&quot;">
      <h3 class="cosc-title"><span>Discard this search redirect</span></h3>
    </div>
    <div data-state="&quot;dataUrl&quot;:&quot;https://web.toutiao.com/article/123?foo=1&amp;bar=2&quot;">
      <h3 class="cosc-title"><span>Industrial <em>AI</em> article</span></h3>
      <!--s-data:{"text":"Result-page summary for the article."}-->
    </div>
    """

    results = BaiduBrowserSearchClient.parse_results(html, checked_at="2026-07-26T00:00:00Z")

    assert len(results) == 1
    assert results[0].url == "https://web.toutiao.com/article/123?foo=1&bar=2"
    assert results[0].title == "Industrial AI article"
    assert results[0].snippet == "Result-page summary for the article."


def test_baidu_browser_search_uses_available_public_results_endpoint_by_default():
    calls = []

    def transport(url, headers):
        calls.append({"url": url, "headers": headers})
        return '<div class="result c-container"><h3 class="t"><a href="https://example.com">Example</a></h3></div>'

    client = BaiduBrowserSearchClient(transport=transport)
    client.search("industrial AI", limit=1)

    assert calls[0]["url"].startswith("https://www.baidu.com/baidu?wd=industrial+AI")
    assert "Windows NT" in calls[0]["headers"]["User-Agent"]


def test_baidu_browser_search_falls_back_when_primary_endpoint_is_challenged():
    calls = []

    def transport(url, headers):
        calls.append(url)
        if url.startswith("https://m.baidu.com/s?"):
            return "<html><title>百度安全验证</title></html>"
        return """
        <div class="result c-container" mu="https://mp.weixin.qq.com/s/fallback-original">
          <h3 class="t"><a href="https://www.baidu.com/link?url=opaque">Fallback public account article</a></h3>
          <div class="c-abstract">Baidu desktop result-page excerpt.</div>
        </div>
        """

    client = BaiduBrowserSearchClient(
        base_url="https://m.baidu.com/s",
        transport=transport,
        fallback_base_urls=("https://www.baidu.com/baidu",),
    )

    results = client.search("site:mp.weixin.qq.com industrial AI", limit=1)

    assert [result.url for result in results] == ["https://mp.weixin.qq.com/s/fallback-original"]
    assert calls[0].startswith("https://m.baidu.com/s?word=site%3Amp.weixin.qq.com")
    assert calls[1].startswith("https://www.baidu.com/baidu?wd=site%3Amp.weixin.qq.com")


def test_browser_search_client_uses_baidu_only_and_skips_platform_enrichment():
    baidu_calls = []
    page_calls = []
    bing = RecordingSearchClient("bing")
    google = RecordingSearchClient("google")

    def baidu_transport(url, headers):
        baidu_calls.append(url)
        return """
        <div class="result c-container" mu="https://mp.weixin.qq.com/s/example-original">
          <h3 class="t"><a href="https://www.baidu.com/link?url=opaque">Public account article</a></h3>
          <div class="c-abstract">Result-page excerpt only.</div>
        </div>
        """

    def content_transport(url, headers):
        page_calls.append(url)
        raise AssertionError("Baidu-only discovery must not fetch the platform detail page")

    client = BrowserSearchClient(
        [bing, BaiduBrowserSearchClient(transport=baidu_transport), google],
        enrich_content=True,
        content_transport=content_transport,
        baidu_only_domains={"mp.weixin.qq.com", "toutiao.com", "xiaohongshu.com", "xhslink.com"},
    )

    results = client.search("site:mp.weixin.qq.com DGX Spark", limit=3)

    assert [result.url for result in results] == ["https://mp.weixin.qq.com/s/example-original"]
    assert len(baidu_calls) == 1
    assert bing.calls == []
    assert google.calls == []
    assert page_calls == []


def test_browser_search_client_falls_back_to_other_engines_when_baidu_public_index_is_blocked():
    page_calls = []

    def baidu_transport(url, headers):
        return "<html><title>百度安全验证</title></html>"

    def content_transport(url, headers):
        page_calls.append(url)
        raise AssertionError("Platform public-index discovery must not fetch the detail page")

    fallback = StaticSearchClient(
        [
            SearchResult(
                title="Off-domain result",
                url="https://example.com/not-wechat",
                snippet="Should be filtered out.",
                provider="browser-bing",
                checked_at="2026-07-26T00:00:00Z",
            ),
            SearchResult(
                title="Public account fallback article",
                url="https://mp.weixin.qq.com/s/fallback-original",
                snippet="Result-page excerpt from another public search engine.",
                provider="browser-google",
                checked_at="2026-07-26T00:00:00Z",
            ),
        ]
    )
    client = BrowserSearchClient(
        [BaiduBrowserSearchClient(transport=baidu_transport), fallback],
        enrich_content=True,
        content_transport=content_transport,
        baidu_only_domains={"mp.weixin.qq.com", "toutiao.com", "xiaohongshu.com", "xhslink.com"},
    )

    results = client.search("site:mp.weixin.qq.com DGX Spark", limit=3)

    assert [result.url for result in results] == ["https://mp.weixin.qq.com/s/fallback-original"]
    assert results[0].provider == "browser-google"
    assert page_calls == []


def test_browser_search_client_retries_site_queries_with_domain_without_faking_coverage():
    class QuerySensitiveSearchClient:
        def __init__(self):
            self.calls = []

        def search(self, query, limit=5):
            self.calls.append(query)
            if query == "industrial AI reddit.com":
                return [
                    SearchResult(
                        title="Industrial AI discussion",
                        url="https://www.reddit.com/r/MachineLearning/comments/example",
                        snippet="A public Reddit discussion about industrial AI.",
                        provider="browser-bing",
                        checked_at="2026-07-26T00:00:00Z",
                    )
                ]
            return [
                SearchResult(
                    title="Off-domain result",
                    url="https://example.com/not-reddit",
                    snippet="The engine ignored the site filter.",
                    provider="browser-bing",
                    checked_at="2026-07-26T00:00:00Z",
                )
            ]

    engine = QuerySensitiveSearchClient()
    client = BrowserSearchClient([engine])

    results = client.search("site:reddit.com industrial AI", limit=3)

    assert [result.url for result in results] == ["https://www.reddit.com/r/MachineLearning/comments/example"]
    assert engine.calls[:2] == ["site:reddit.com industrial AI", "industrial AI reddit.com"]


def test_baidu_browser_search_rejects_verification_pages():
    client = BaiduBrowserSearchClient(
        transport=lambda url, headers: "<html><title>百度安全验证</title></html>"
    )

    with pytest.raises(SearchProviderError, match="verification page"):
        client.search("site:mp.weixin.qq.com industrial AI")


def test_baidu_browser_search_cools_down_after_verification_page():
    calls = []

    def transport(url, headers):
        calls.append(url)
        return "<html><title>百度安全验证</title></html>"

    client = BaiduBrowserSearchClient(
        transport=transport,
        fallback_base_urls=(),
        challenge_cooldown_seconds=30.0,
    )

    with pytest.raises(SearchProviderError, match="verification page"):
        client.search("site:mp.weixin.qq.com industrial AI")
    with pytest.raises(SearchProviderError, match="cooling down"):
        client.search("site:mp.weixin.qq.com industrial AI")

    assert len(calls) == 1


def test_google_browser_search_parser_extracts_results_from_html():
    html = """
    <div class="g">
      <a href="/url?q=https%3A%2F%2Fexample.com%2Fagent&amp;sa=U">
        <h3>Microsoft Agent Framework Docs</h3>
      </a>
      <div class="VwiC3b">Workflow and agent documentation.</div>
    </div>
    """

    results = GoogleBrowserSearchClient.parse_results(html, checked_at="2026-06-27T00:00:00Z")

    assert len(results) == 1
    assert results[0].title == "Microsoft Agent Framework Docs"
    assert results[0].url == "https://example.com/agent"
    assert results[0].snippet == "Workflow and agent documentation."
    assert results[0].provider == "browser-google"


def test_browser_search_client_merges_engines_dedupes_and_tolerates_failures():
    client = BrowserSearchClient(
        [
            StaticSearchClient(
                [
                    SearchResult(
                        title="A",
                        url="https://example.com/a",
                        snippet="from first engine",
                        provider="browser-bing",
                        checked_at="2026-06-27T00:00:00Z",
                    )
                ]
            ),
            StaticSearchClient(
                [
                    SearchResult(
                        title="A duplicate",
                        url="https://example.com/a",
                        snippet="duplicate",
                        provider="browser-baidu",
                        checked_at="2026-06-27T00:00:00Z",
                    ),
                    SearchResult(
                        title="B",
                        url="https://example.com/b",
                        snippet="from second engine",
                        provider="browser-baidu",
                        checked_at="2026-06-27T00:00:00Z",
                    ),
                ]
            ),
            FailingSearchClient(),
        ]
    )

    results = client.search("query", limit=5)

    assert [result.url for result in results] == ["https://example.com/a", "https://example.com/b"]
    assert [result.provider for result in results] == ["browser-bing", "browser-baidu"]


def test_browser_search_client_interleaves_engines_before_truncating():
    client = BrowserSearchClient(
        [
            StaticSearchClient(
                [
                    SearchResult(
                        title="Bing 1",
                        url="https://example.com/bing-1",
                        snippet="",
                        provider="browser-bing",
                        checked_at="2026-06-27T00:00:00Z",
                    ),
                    SearchResult(
                        title="Bing 2",
                        url="https://example.com/bing-2",
                        snippet="",
                        provider="browser-bing",
                        checked_at="2026-06-27T00:00:00Z",
                    ),
                ]
            ),
            StaticSearchClient(
                [
                    SearchResult(
                        title="Baidu 1",
                        url="https://example.com/baidu-1",
                        snippet="",
                        provider="browser-baidu",
                        checked_at="2026-06-27T00:00:00Z",
                    )
                ]
            ),
            StaticSearchClient(
                [
                    SearchResult(
                        title="Google 1",
                        url="https://example.com/google-1",
                        snippet="",
                        provider="browser-google",
                        checked_at="2026-06-27T00:00:00Z",
                    )
                ]
            ),
        ]
    )

    results = client.search("query", limit=3)

    assert [result.provider for result in results] == ["browser-bing", "browser-baidu", "browser-google"]


def test_browser_search_client_skips_slow_engines_after_timeout():
    client = BrowserSearchClient(
        [
            SlowSearchClient(delay_seconds=0.25),
            StaticSearchClient(
                [
                    SearchResult(
                        title="Fast result",
                        url="https://example.com/fast",
                        snippet="returned by the fast engine",
                        provider="browser-bing",
                        checked_at="2026-06-28T00:00:00Z",
                    )
                ]
            ),
        ],
        engine_timeout_seconds=0.05,
    )

    started_at = time.monotonic()
    results = client.search("query", limit=3)
    elapsed = time.monotonic() - started_at

    assert [result.url for result in results] == ["https://example.com/fast"]
    assert elapsed < 0.2


def test_browser_search_client_can_enrich_snippets_from_result_pages():
    def content_transport(url, headers):
        assert url == "https://example.com/dgx-spark"
        return """
        <html>
          <body>
            <nav>DGX Spark overview DGX Spark buy now DGX Spark support</nav>
            <main>
              <section>Other product text that should not be selected.</section>
              <section>
                NVIDIA DGX Spark combines the GB10 Grace Blackwell Superchip
                with 128 GB of unified memory for local AI development.
              </section>
            </main>
          </body>
        </html>
        """

    client = BrowserSearchClient(
        [
            StaticSearchClient(
                [
                    SearchResult(
                        title="NVIDIA DGX Spark",
                        url="https://example.com/dgx-spark",
                        snippet="Short search snippet.",
                        provider="unit",
                        checked_at="2026-06-27T00:00:00Z",
                    )
                ]
            )
        ],
        enrich_content=True,
        content_transport=content_transport,
    )

    results = client.search("DGX Spark", limit=1)

    assert "Short search snippet." in results[0].snippet
    assert "128 GB of unified memory" in results[0].snippet


def test_browser_search_client_skips_slow_content_enrichment_after_timeout():
    def slow_content_transport(url, headers):
        time.sleep(0.25)
        return "<html><body>slow page body</body></html>"

    client = BrowserSearchClient(
        [
            StaticSearchClient(
                [
                    SearchResult(
                        title="Slow page",
                        url="https://example.com/slow",
                        snippet="Original snippet.",
                        provider="unit",
                        checked_at="2026-06-28T00:00:00Z",
                    )
                ]
            )
        ],
        enrich_content=True,
        content_transport=slow_content_transport,
        content_timeout_seconds=0.05,
        max_enriched_results=1,
    )

    started_at = time.monotonic()
    results = client.search("query", limit=1)
    elapsed = time.monotonic() - started_at

    assert results[0].snippet == "Original snippet."
    assert elapsed < 0.2


def test_browser_search_client_supplements_deepseek_model_hub_sources():
    def content_transport(url, headers):
        assert url in {
            "https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3",
            "https://huggingface.co/deepseek-ai/DeepSeek-V3",
        }
        return """
        <html><body>
          DeepSeek-V3 is a Mixture-of-Experts model with 671B total parameters
          and 37B activated for each token.
        </body></html>
        """

    client = BrowserSearchClient(
        [StaticSearchClient([])],
        content_transport=content_transport,
        max_supplemental_results=2,
    )

    results = client.search("ModelScope DeepSeek-V3 671B 37B", limit=2)

    assert [result.url for result in results] == [
        "https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3",
        "https://huggingface.co/deepseek-ai/DeepSeek-V3",
    ]
    assert "671B total parameters" in results[0].snippet
    assert results[0].provider == "direct-model-hub"


def test_browser_search_client_keeps_direct_source_when_content_fetch_fails():
    def content_transport(url, headers):
        if "modelscope" in url:
            return """
            <html><body>
              DeepSeek-V3 is a Mixture-of-Experts model with 671B total parameters
              and 37B activated for each token.
            </body></html>
            """
        raise SearchProviderError("source blocked direct fetch")

    client = BrowserSearchClient(
        [StaticSearchClient([])],
        content_transport=content_transport,
        max_supplemental_results=3,
    )

    results = client.search("DeepSeek-V3 model card parameters GitHub Hugging Face", limit=3)

    by_url = {result.url: result for result in results}
    assert set(by_url) == {
        "https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3",
        "https://huggingface.co/deepseek-ai/DeepSeek-V3",
        "https://github.com/deepseek-ai/DeepSeek-V3",
    }
    assert "671B total parameters" in by_url["https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3"].snippet
    assert "Direct source selected" in by_url["https://huggingface.co/deepseek-ai/DeepSeek-V3"].snippet
    assert by_url["https://huggingface.co/deepseek-ai/DeepSeek-V3"].provider == "direct-model-hub"


def test_browser_search_client_supplements_cxl_official_source():
    def content_transport(url, headers):
        assert url == "https://www.computeexpresslink.org/about-cxl/"
        return """
        <html><body>
          Compute Express Link (CXL) is an open industry-standard interconnect
          for high-speed CPU-to-device and CPU-to-memory connectivity.
          CXL supports coherent memory access for accelerators and emerging
          Artificial Intelligence workloads.
        </body></html>
        """

    client = BrowserSearchClient(
        [StaticSearchClient([])],
        content_transport=content_transport,
    )

    results = client.search("CXL Compute Express Link memory pooling", limit=1)

    assert results[0].url == "https://www.computeexpresslink.org/about-cxl/"
    assert results[0].provider == "direct-official"
    assert "Compute Express Link" in results[0].snippet
    assert "Artificial Intelligence workloads" in results[0].snippet


def test_browser_search_client_supplements_h100_official_source_for_inference_bandwidth():
    def content_transport(url, headers):
        assert url == "https://www.nvidia.com/en-us/data-center/h100/"
        return """
        <html><body>
          NVIDIA H100 Tensor Core GPUs include HBM3 memory and accelerate
          large language model inference with Transformer Engine.
        </body></html>
        """

    client = BrowserSearchClient(
        [StaticSearchClient([])],
        content_transport=content_transport,
    )

    results = client.search(
        "H100 tok/s LLM inference decode throughput HBM3 memory bandwidth KV cache Tensor Core",
        limit=1,
    )

    assert results[0].url == "https://www.nvidia.com/en-us/data-center/h100/"
    assert results[0].provider == "direct-official"
    assert "HBM3 memory" in results[0].snippet


def test_browser_search_client_supplements_generic_deepseek_source_families():
    def content_transport(url, headers):
        if "modelscope" in url:
            return "<html><body>DeepSeek organization and model cards on ModelScope.</body></html>"
        if "huggingface" in url:
            return "<html><body>deepseek-ai models on Hugging Face.</body></html>"
        if "github" in url:
            return "<html><body>deepseek-ai source repositories on GitHub.</body></html>"
        raise AssertionError(url)

    client = BrowserSearchClient(
        [StaticSearchClient([])],
        content_transport=content_transport,
        max_supplemental_results=3,
    )

    results = client.search("DeepSeek deployment ModelScope Hugging Face GitHub NVIDIA official pages", limit=3)

    assert [result.url for result in results] == [
        "https://www.modelscope.cn/organization/deepseek-ai",
        "https://huggingface.co/deepseek-ai",
        "https://github.com/deepseek-ai",
    ]


def test_browser_search_client_supplements_deepseek_parameter_queries_without_named_hubs():
    def content_transport(url, headers):
        if "modelscope" in url:
            return "<html><body>DeepSeek model cards and parameter metadata on ModelScope.</body></html>"
        if "huggingface" in url:
            return "<html><body>deepseek-ai model cards on Hugging Face.</body></html>"
        if "github" in url:
            return "<html><body>deepseek-ai repositories on GitHub.</body></html>"
        raise AssertionError(url)

    client = BrowserSearchClient(
        [StaticSearchClient([])],
        content_transport=content_transport,
        max_supplemental_results=3,
    )

    results = client.search("DeepSeek 模型参数", limit=3)

    assert [result.url for result in results] == [
        "https://www.modelscope.cn/organization/deepseek-ai",
        "https://huggingface.co/deepseek-ai",
        "https://github.com/deepseek-ai",
    ]


def test_browser_search_client_treats_mota_as_modelscope_alias():
    def content_transport(url, headers):
        if "modelscope" in url:
            return "<html><body>DeepSeek model card on ModelScope also called 魔搭 by users.</body></html>"
        if "huggingface" in url or "github" in url:
            return "<html><body>DeepSeek source family.</body></html>"
        raise AssertionError(url)

    client = BrowserSearchClient(
        [StaticSearchClient([])],
        content_transport=content_transport,
        max_supplemental_results=1,
    )

    results = client.search("魔塔 DeepSeek 模型参数", limit=1)

    assert results[0].url == "https://www.modelscope.cn/organization/deepseek-ai"


def test_browser_search_client_supplements_dgx_spark_official_source_for_connectx():
    def content_transport(url, headers):
        assert url == "https://www.nvidia.com/en-us/products/workstations/dgx-spark/"
        return """
        <html><body>
          NVIDIA DGX Spark uses the GB10 Grace Blackwell Superchip with 128 GB
          of unified memory. NIC: ConnectX-7 NIC @ 200 Gbps.
        </body></html>
        """

    client = BrowserSearchClient(
        [StaticSearchClient([])],
        content_transport=content_transport,
    )

    results = client.search("DGX Spark ConnectX-7 CX7", limit=1)

    assert results[0].url == "https://www.nvidia.com/en-us/products/workstations/dgx-spark/"
    assert "ConnectX-7 NIC @ 200 Gbps" in results[0].snippet
    assert results[0].provider == "direct-official"


def test_browser_search_client_supplements_connectx7_official_source():
    def content_transport(url, headers):
        assert url == "https://www.nvidia.com/en-us/networking/ethernet/connectx-7/"
        return """
        <html><body>
          NVIDIA ConnectX-7 is a network adapter for high performance
          Ethernet and InfiniBand networking in AI data centers.
        </body></html>
        """

    client = BrowserSearchClient(
        [StaticSearchClient([])],
        content_transport=content_transport,
    )

    results = client.search("ConnectX-7 CX7 AI cluster networking", limit=1)

    assert results[0].url == "https://www.nvidia.com/en-us/networking/ethernet/connectx-7/"
    assert "ConnectX-7" in results[0].snippet


def test_browser_search_client_supplements_physical_ai_source_families():
    def content_transport(url, headers):
        if "cosmos" in url:
            return "<html><body>NVIDIA Cosmos world foundation models for physical AI.</body></html>"
        if "genie-2" in url:
            return "<html><body>Genie 2 is a large-scale foundation world model.</body></html>"
        if "arxiv" in url:
            return "<html><body>arXiv search results for world model embodied AI papers.</body></html>"
        raise AssertionError(url)

    client = BrowserSearchClient(
        [StaticSearchClient([])],
        content_transport=content_transport,
        max_supplemental_results=3,
    )

    results = client.search("world model embodied physical AI progress 2026", limit=3)

    assert [result.provider for result in results] == [
        "direct-official",
        "direct-official",
        "direct-literature",
    ]
    assert results[0].url == "https://www.nvidia.com/en-us/ai-data-science/cosmos/"
    assert results[1].url == "https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/"
    assert results[2].url.startswith("https://arxiv.org/search/")


def test_browser_search_client_supplements_distributed_inference_source_families():
    def content_transport(url, headers):
        if "vllm" in url:
            return "<html><body>vLLM documents distributed serving with tensor parallelism and pipeline parallelism.</body></html>"
        if "nemo" in url:
            return "<html><body>Megatron Bridge documents tensor parallelism, pipeline parallelism, and expert parallelism.</body></html>"
        if "deepspeed" in url:
            return "<html><body>DeepSpeed inference documents tensor parallelism for large model inference.</body></html>"
        raise AssertionError(url)

    client = BrowserSearchClient(
        [StaticSearchClient([])],
        content_transport=content_transport,
        max_supplemental_results=3,
    )

    results = client.search("分布式大模型推理 架构 节点 互联互通 模型并行 张量并行", limit=3)

    assert [result.provider for result in results] == [
        "direct-official",
        "direct-official",
        "direct-official",
    ]
    assert results[0].url == "https://docs.vllm.ai/en/stable/serving/parallelism_scaling/"
    assert results[1].url == "https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html"
    assert results[2].url == "https://deepspeed.readthedocs.io/en/latest/inference-init.html"
    assert "tensor parallelism" in results[0].snippet


def test_supplemental_excerpt_uses_chinese_distributed_inference_aliases():
    def content_transport(url, headers):
        if "vllm" in url:
            return (
                "<html><body>"
                + ("Navigation Home Deployment API Reference " * 120)
                + "Distributed inference strategies for a single-model replica include "
                + "tensor parallelism and pipeline parallelism. Expert parallelism uses "
                + "all-to-all communication across nodes for mixture-of-experts models."
                + "</body></html>"
            )
        if "nemo" in url:
            return "<html><body>Megatron Bridge documents expert parallelism.</body></html>"
        if "deepspeed" in url:
            return "<html><body>DeepSpeed inference documents tensor parallelism.</body></html>"
        raise AssertionError(url)

    client = BrowserSearchClient(
        [StaticSearchClient([])],
        content_transport=content_transport,
        max_supplemental_results=1,
    )

    results = client.search("分布式大模型是否可以理解为多个节点负责一部分推理并互联互通", limit=1)

    assert "Distributed inference strategies" in results[0].snippet
    assert "tensor parallelism" in results[0].snippet
    assert "all-to-all communication" in results[0].snippet
    assert "Navigation Home" not in results[0].snippet[:120]


def test_browser_search_client_supplemental_excerpt_prefers_parameter_and_nic_facts():
    def content_transport(url, headers):
        if "modelscope" in url:
            return (
                "<html><body>"
                + ("ModelScope home navigation " * 120)
                + "DeepSeek-V3 is a strong MoE model with 671B total parameters "
                + "and 37B activated for each token."
                + "</body></html>"
            )
        return (
            "<html><body>"
            + ("ConnectX Networking teaser " * 80)
            + "NVIDIA DGX Spark has 128 GB unified memory. "
            + "NIC: ConnectX-7 NIC @ 200 Gbps."
            + "</body></html>"
        )

    model_client = BrowserSearchClient(
        [StaticSearchClient([])],
        content_transport=content_transport,
        max_supplemental_results=1,
    )
    dgx_client = BrowserSearchClient([StaticSearchClient([])], content_transport=content_transport)

    model_results = model_client.search("ModelScope DeepSeek-V3 671B 37B", limit=1)
    dgx_results = dgx_client.search("DGX Spark ConnectX-7 CX7", limit=1)

    assert "671B total parameters" in model_results[0].snippet
    assert "37B activated" in model_results[0].snippet
    assert "ConnectX-7 NIC @ 200 Gbps" in dgx_results[0].snippet


def test_browser_search_client_supplemental_excerpt_decodes_modelscope_escaped_text():
    def content_transport(url, headers):
        return (
            '<html><body><script>window.__DATA__="DeepSeek-V3\\n'
            'We present DeepSeek-V3, a strong Mixture-of-Experts model with '
            '671B total parameters with 37B activated for each token."</script></body></html>'
        )

    client = BrowserSearchClient(
        [StaticSearchClient([])],
        content_transport=content_transport,
        max_supplemental_results=1,
    )

    results = client.search("ModelScope DeepSeek-V3 671B 37B", limit=1)

    assert "671B total parameters" in results[0].snippet
    assert "37B activated" in results[0].snippet


def test_bing_browser_search_uses_configured_market_without_api_key():
    calls = []

    def transport(url, headers):
        calls.append({"url": url, "headers": headers})
        return '<li class="b_algo"><h2><a href="https://example.com">Example</a></h2></li>'

    client = BingBrowserSearchClient(market="zh-CN", transport=transport)

    client.search("飞书 bot", limit=1)

    assert "q=%E9%A3%9E%E4%B9%A6+bot" in calls[0]["url"]
    assert "mkt=zh-CN" in calls[0]["url"]
    assert calls[0]["headers"]["Accept-Language"] == "zh-CN,zh;q=0.9,en;q=0.6"


def test_bing_browser_search_uses_english_market_for_ascii_query():
    calls = []

    def transport(url, headers):
        calls.append({"url": url, "headers": headers})
        return '<li class="b_algo"><h2><a href="https://example.com">Example</a></h2></li>'

    client = BingBrowserSearchClient(market="zh-CN", transport=transport)

    client.search("DGX Spark", limit=1)

    assert "q=DGX+Spark" in calls[0]["url"]
    assert "mkt=en-US" in calls[0]["url"]
    assert calls[0]["headers"]["Accept-Language"] == "en-US,en;q=0.9"


def test_search_client_from_settings_uses_hybrid_search_by_default_without_api_key(tmp_path):
    settings = Settings.from_env({"SEARCH_ASSISTANT_DATA_DIR": str(tmp_path)})

    client = search_client_from_settings(settings)

    assert isinstance(client, CompositeSearchClient)
    assert isinstance(client.clients[0], McpSearchClient)
    assert isinstance(client.clients[1], BrowserSearchClient)
    assert client.primary_sufficient_results is None
    assert [type(engine).__name__ for engine in client.clients[1].engines] == [
        "BingBrowserSearchClient",
        "BaiduBrowserSearchClient",
        "GoogleBrowserSearchClient",
    ]
    assert client.clients[1].baidu_only_domains == {
        "mp.weixin.qq.com",
        "weixin.qq.com",
        "toutiao.com",
        "xiaohongshu.com",
        "xhslink.com",
    }
    baidu = next(engine for engine in client.clients[1].engines if isinstance(engine, BaiduBrowserSearchClient))
    assert baidu.base_url == "https://www.baidu.com/baidu"


def test_search_client_from_settings_allows_agent_reach_provider(tmp_path):
    settings = Settings.from_env(
        {
            "SEARCH_ASSISTANT_DATA_DIR": str(tmp_path),
            "SEARCH_ASSISTANT_SEARCH_PROVIDER": "agent-reach",
            "AGENT_REACH_COMMAND": "agent-reach-custom",
            "AGENT_REACH_TIMEOUT_SECONDS": "11",
            "AGENT_REACH_DOCTOR_CACHE_SECONDS": "22",
        }
    )

    client = search_client_from_settings(settings)

    assert isinstance(client, AgentReachSearchClient)
    assert client.command == "agent-reach-custom"
    assert client.timeout_seconds == 11.0
    assert client.doctor_cache_seconds == 22.0


def test_search_client_from_settings_appends_agent_reach_to_hybrid(tmp_path):
    settings = Settings.from_env(
        {
            "SEARCH_ASSISTANT_DATA_DIR": str(tmp_path),
            "SEARCH_ASSISTANT_AGENT_REACH_ENABLED": "true",
        }
    )

    client = search_client_from_settings(settings)

    assert isinstance(client, CompositeSearchClient)
    assert isinstance(client.clients[0], McpSearchClient)
    assert isinstance(client.clients[1], BrowserSearchClient)
    # Agent Reach runs last as a supplementary pass so its slower subprocess
    # routes cannot starve the MCP/browser engines of the search budget.
    assert isinstance(client.clients[2], AgentReachSearchClient)
    assert client.clients[2].require_available is False


def test_search_client_from_settings_allows_browser_engine_selection(tmp_path):
    settings = Settings.from_env(
        {
            "SEARCH_ASSISTANT_DATA_DIR": str(tmp_path),
            "SEARCH_ASSISTANT_SEARCH_PROVIDER": "browser",
            "BROWSER_SEARCH_ENGINES": "baidu,google",
        }
    )

    client = search_client_from_settings(settings)

    assert isinstance(client, BrowserSearchClient)
    assert [type(engine).__name__ for engine in client.engines] == [
        "BaiduBrowserSearchClient",
        "GoogleBrowserSearchClient",
    ]


def test_search_client_from_settings_allows_duckduckgo_in_browser_engine_pool(tmp_path):
    settings = Settings.from_env(
        {
            "SEARCH_ASSISTANT_DATA_DIR": str(tmp_path),
            "SEARCH_ASSISTANT_SEARCH_PROVIDER": "browser",
            "BROWSER_SEARCH_ENGINES": "bing,duckduckgo",
        }
    )

    client = search_client_from_settings(settings)

    assert isinstance(client, BrowserSearchClient)
    assert [type(engine).__name__ for engine in client.engines] == [
        "BingBrowserSearchClient",
        "DuckDuckGoSearchClient",
    ]


def test_search_client_from_settings_passes_browser_latency_settings(tmp_path):
    settings = Settings.from_env(
        {
            "SEARCH_ASSISTANT_DATA_DIR": str(tmp_path),
            "SEARCH_ASSISTANT_SEARCH_PROVIDER": "browser",
            "BROWSER_SEARCH_TIMEOUT_SECONDS": "3.5",
            "BROWSER_CONTENT_TIMEOUT_SECONDS": "1.25",
            "BROWSER_ENRICH_MAX_RESULTS": "2",
        }
    )

    client = search_client_from_settings(settings)

    assert isinstance(client, BrowserSearchClient)
    assert client.engine_timeout_seconds == 3.5
    assert client.content_timeout_seconds == 1.25
    assert client.max_enriched_results == 2


def test_search_client_from_settings_uses_brave_by_default_when_key_is_set(tmp_path):
    settings = Settings.from_env(
        {
            "SEARCH_ASSISTANT_DATA_DIR": str(tmp_path),
            "SEARCH_ASSISTANT_SEARCH_PROVIDER": "brave",
            "BRAVE_SEARCH_API_KEY": "test-search-key",
        }
    )

    client = search_client_from_settings(settings)

    assert isinstance(client, BraveSearchClient)


def test_search_client_from_settings_uses_self_hosted_searxng_without_a_key(tmp_path):
    settings = Settings.from_env(
        {
            "SEARCH_ASSISTANT_DATA_DIR": str(tmp_path),
            "SEARCH_ASSISTANT_SEARCH_PROVIDER": "searxng",
            "SEARXNG_BASE_URL": "http://localhost:8080/search",
            "SEARXNG_TIMEOUT_SECONDS": "7",
        }
    )

    client = search_client_from_settings(settings)

    assert isinstance(client, SearxngSearchClient)
    assert client.base_url == "http://localhost:8080/search"


def test_search_client_from_settings_requires_brave_key_when_brave_is_selected(tmp_path):
    settings = Settings.from_env(
        {
            "SEARCH_ASSISTANT_DATA_DIR": str(tmp_path),
            "SEARCH_ASSISTANT_SEARCH_PROVIDER": "brave",
        }
    )

    with pytest.raises(RuntimeError, match="BRAVE_SEARCH_API_KEY"):
        search_client_from_settings(settings)


def test_search_client_from_settings_allows_fake_only_for_tests(tmp_path):
    settings = Settings.from_env(
        {
            "SEARCH_ASSISTANT_DATA_DIR": str(tmp_path),
            "SEARCH_ASSISTANT_SEARCH_PROVIDER": "fake",
            "SEARCH_ASSISTANT_ALLOW_FAKE_RUNTIME": "true",
        }
    )

    client = search_client_from_settings(settings)

    assert isinstance(client, FakeSearchClient)
    assert client.search("hello") == []


def test_search_client_from_settings_rejects_fake_without_test_flag(tmp_path):
    settings = Settings.from_env(
        {
            "SEARCH_ASSISTANT_DATA_DIR": str(tmp_path),
            "SEARCH_ASSISTANT_SEARCH_PROVIDER": "fake",
        }
    )

    with pytest.raises(RuntimeError, match="Fake search provider is disabled"):
        search_client_from_settings(settings)


class StaticSearchClient:
    def __init__(self, results):
        self.results = results

    def search(self, query, limit=5):
        return self.results[:limit]


class RecordingSearchClient:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def search(self, query, limit=5):
        self.calls.append((query, limit))
        return []


class RecordingMcpToolset:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def direct_call_tool(self, name, args):
        self.calls.append((name, args))
        return self.result


class ConcurrentMcpToolset:
    def __init__(self):
        self._lock = Lock()
        self.active_calls = 0
        self.max_active_calls = 0

    async def direct_call_tool(self, name, args):
        with self._lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            await asyncio.sleep(0.02)
            return [
                {
                    "title": "Industrial AI",
                    "url": "https://example.com/industrial-ai",
                    "provider": "test",
                }
            ]
        finally:
            with self._lock:
                self.active_calls -= 1


class FailingSearchClient:
    def search(self, query, limit=5):
        raise SearchProviderError("blocked")


class SlowSearchClient:
    def __init__(self, delay_seconds):
        self.delay_seconds = delay_seconds

    def search(self, query, limit=5):
        time.sleep(self.delay_seconds)
        return [
            SearchResult(
                title="Slow result",
                url="https://example.com/slow-engine",
                snippet="too slow",
                provider="browser-slow",
                checked_at="2026-06-28T00:00:00Z",
            )
        ]
