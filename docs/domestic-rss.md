# Domestic RSSHub Integration

## Purpose and boundary

This integration turns selected public domestic-platform feeds into an MCP search source. It is not a login automation system and it is not a platform-internal search API.

- RSSHub is self-hosted at `RSSHUB_BASE_URL`, defaulting to `http://127.0.0.1:1200`.
- The search assistant starts `scripts/domestic_rss_mcp.py` through Pydantic AI MCP.
- The MCP service parses RSS, Atom, and JSON Feed outputs, keeps only original platform URLs, and returns the source platform in the provider field.
- The service does not read, inject, store, or report platform cookies.
- A source marked `requires_login: true` is always skipped.
- A `site:` query can invoke this MCP source only for the domains explicitly declared by the catalog entry. Returned URLs are checked against the same domain list.

## Local deployment

Use the checked-in local-only compose file:

```powershell
Set-Location infra\rsshub
docker compose up -d
Invoke-WebRequest "http://127.0.0.1:1200/healthz"
```

The Chromium-enabled image is intentional: an operator-enabled RSSHub route for Douyin may need browser automation even where no cookie is supplied. Keep port `1200` bound to loopback. This deployment does not make strict anti-crawling routes reliable and must not be used to bypass a platform access control.

## Catalog

`configs/domestic-rss.sources.json` is the default catalog. An operator can select another catalog with:

```powershell
$env:RSSHUB_BASE_URL = "http://127.0.0.1:1200"
$env:SEARCH_ASSISTANT_DOMESTIC_RSS_CONFIG = "C:\path\to\domestic-rss.sources.json"
$env:DOMESTIC_RSS_TIMEOUT_SECONDS = "8"
```

Each source requires an `id`, `platform`, `domains`, and either `route_template` or `feed_url`.

```json
{
  "id": "douyin-public-user",
  "platform": "douyin",
  "kind": "subscription",
  "route_template": "/douyin/user/{uid}",
  "domains": ["douyin.com"],
  "parameters": {"uid": "PUBLIC_UID"},
  "enabled": true,
  "requires_login": false,
  "requires_browser": true,
  "requires_external_subscription": false
}
```

Template values are URL encoded. Missing values leave the source inactive. `feed_url` is useful for a platform's official feed or a separately hosted RSS output, but item links still must resolve to the source's declared platform domain.

## Platform matrix

| Platform | Route shape | Default | Constraint |
| --- | --- | --- | --- |
| Weibo | `/weibo/keyword/{query}` | Enabled | Keyword discovery only; no cookie is sent. |
| Zhihu | `/zhihu/hot` | Enabled | Hot-topic signal; public pages may not expose full text. |
| Bilibili | Direct public video search API | Enabled | Query-aware public video search is used instead of RSSHub. |
| 36Kr | `/36kr/newsflashes` | Enabled | Public news-flash signal, filtered locally by the retrieval topic. |
| Juejin | `/juejin/aicoding` | Enabled | Public AI coding channel, filtered locally by the retrieval topic. |
| Douyin | `/douyin/user/{uid}` | Disabled | Needs a public user id and RSSHub browser support; anti-crawling can make it unavailable. |
| Kuaishou | `/kuaishou/profile/{principal_id}` | Disabled | Needs a public profile id and RSSHub browser support. |

## Baidu-first public-index discovery

WeChat public accounts, Toutiao, and Xiaohongshu are intentionally outside the RSSHub catalog and the domestic MCP binding. Their daily-brief coverage uses Baidu's public result pages first, preferring `https://www.baidu.com/baidu` and falling back to other public Baidu endpoints when one is challenged:

| Platform | Public-index query | Detail-page fetch | Retained URL |
| --- | --- | --- | --- |
| WeChat public accounts | `site:mp.weixin.qq.com <topic>` | Disabled | The Baidu result's original `mp.weixin.qq.com` target. |
| Toutiao | `site:toutiao.com <topic>` plus the public Toutiao search page | Disabled | The public search result's original `toutiao.com` target. |
| Xiaohongshu | `site:xiaohongshu.com <topic>` and `site:xhslink.com <topic>` | Disabled | Each Baidu result's original `xiaohongshu.com` or `xhslink.com` target. |

The assistant does not call login-gated platform APIs, RSSHub routes, or article/note detail pages for these three channels. Baidu remains the preferred public-index source for WeChat and Xiaohongshu; when a Baidu endpoint is blocked, another public Baidu endpoint is tried before the browser search client falls back to other configured public search engines and safer site-query variants. Toutiao also has a no-login public search-page parser because that page exposes original `toutiao.com` result URLs in its markup. Every fallback still enforces the original platform domain. Search-engine click-tracking links are discarded unless an original target URL is exposed in the public result markup. This remains public-index discovery, not exhaustive platform coverage.

The upstream RSSHub project documents that these route capabilities can change independently. Treat a route as active only after its local health check and a real source response succeed. Do not represent a configured but failing route as coverage in a briefing.

## Diagnostics

The MCP server exposes `list_domestic_rss_sources` for source status. The regular retrieval binding invokes only `search_domestic_rss`. A source can be present in the catalog yet show one of these non-active states:

- `disabled`: intentionally not searched.
- `requires_login`: rejected by policy.
- `missing_required_parameters`: a public account id, profile token, or external subscription id has not been supplied.

For a controlled local verification, run the focused tests:

```powershell
$env:PYTHONUTF8 = "1"
py -3.12 -m pytest -q tests/test_mcp_domestic_rss.py tests/test_search_provider.py
```
