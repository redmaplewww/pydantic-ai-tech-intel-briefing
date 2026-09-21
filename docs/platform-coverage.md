# Platform Coverage And Boundaries

The assistant retrieves public material. A configured route is not equivalent
to a guaranteed search result, and no route bypasses platform access controls.

## Supported Public Routes

| Platform or source | Route | Login state | Result boundary |
| --- | --- | --- | --- |
| GitHub | Local read-only MCP API client | None | Public repositories and metadata. |
| arXiv | Local read-only MCP API client | None | Public papers and metadata. |
| Hacker News | Local read-only MCP API client | None | Public stories and discussions. |
| Stack Exchange | Local read-only MCP API client | None | Public questions and answers. |
| Bing, Baidu, Google | Public result-page parsing | None | Search engine output can change or block automation. |
| Bilibili | Public video-search API | None | Public video search; original video URL retained. |
| Weibo | Optional RSSHub keyword feed | None | Public keyword discovery. |
| Zhihu | Optional RSSHub public hot feed | None | Topic signal, not exhaustive full text. |
| 36Kr, Juejin | Optional RSSHub feeds | None | Public channel and news-flash signals. |
| WeChat public accounts | Baidu public-index query | None | Original target URL only, no article fetch. |
| Toutiao | Baidu public-index query | None | Original target URL only, no article fetch. |
| Xiaohongshu | Baidu public-index query | None | Original target URL only, no note fetch. |
| LinkedIn, X, Reddit, YouTube | Public result-page discovery | None | Public pages only; no internal platform search. |

## Source Capability Contracts

The code keeps a local registry for supported source families.  Each contract
records aliases, hostnames, provider families, allowed read-only actions,
unsupported actions such as login/write/private-feed access, and a default
search-budget weight.  This makes source coverage auditable without adding any
new private automation route.

Useful local checks:

- `search-assistant source-contracts`
- `search-assistant topic-recipe-show "<topic>"`
- `search-assistant topic-recipe-set "<topic>" '{"general-web": 3, "bilibili": 1}'`

## Disabled Or Conditional Routes

| Platform | Status | Reason |
| --- | --- | --- |
| Douyin | Disabled RSSHub subscription slot | Needs a public account identifier and route validation; anti-crawling may still block it. |
| Kuaishou | Disabled RSSHub subscription slot | Same constraints as Douyin. |
| Private accounts, comments, recommendation feeds | Unsupported | Require login state, private APIs, or non-repeatable ranking. |

## Source Quality Rules

Before a result can become report evidence, the briefing service checks:

- It has a valid original `http` or `https` URL.
- It is not a known generic reference, search-result wrapper, login page, or
  malformed HTML/JSON page dump.
- A `site:` query result is actually inside the requested platform domain.
- It has enough topic or technical signal overlap to be relevant.
- CAD queries reject common medical uses of the acronym.
- Title and snippet are compacted before persistence and model synthesis.

The report keeps all retained original URLs. It does not imply that retained
sources are exhaustive, independently verified, or representative of all
platform discussions.

## RSSHub

The optional RSSHub setup is local-only by default. See
[domestic-rss.md](domestic-rss.md) for its catalog, health checks, and safety
constraints. Do not expose the local RSSHub port publicly without an access
control layer.
