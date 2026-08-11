# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Search backends for the Anthropic ``web_search`` server tool.

``web_search_20250305`` is a *server* tool: the client sends the tool
definition, and the server is expected to run the query itself and feed the
results back to the model within a single ``/v1/messages`` exchange. That
means this server needs an actual search provider.

No provider is enabled by default. Deployments select one through
``TRTLLM_ANTHROPIC_WEB_SEARCH``; until they do, ``web_search_20250305``
keeps returning the same "not supported by this server" error as before, so
enabling the feature is always an explicit, auditable decision.

Providers
---------

=========== ============================ ==========================
provider    credentials                  notes
=========== ============================ ==========================
``off``       -                          default; feature disabled
``wikipedia`` none                       official MediaWiki search API;
                                         encyclopaedic scope only, but
                                         reliable and rate-limit friendly
``mojeek``    none                       general web, scraped from HTML.
                                         Unauthenticated use is rate
                                         limited and starts returning 403
                                         under sustained load - fine for
                                         a trial, not for a shared server
``brave``     ``BRAVE_SEARCH_API_KEY``   general web, JSON API
``tavily``    ``TAVILY_API_KEY``         general web, JSON API, LLM-oriented
``searxng``   ``SEARXNG_URL``            self-hosted JSON endpoint
=========== ============================ ==========================

Environment
-----------

``TRTLLM_ANTHROPIC_WEB_SEARCH``            provider name (default ``off``)
``TRTLLM_ANTHROPIC_WEB_SEARCH_MAX_RESULTS``  results per query (default 5)
``TRTLLM_ANTHROPIC_WEB_SEARCH_TIMEOUT_S``    per-query timeout (default 15)
``TRTLLM_ANTHROPIC_WEB_SEARCH_MAX_USES``     hard cap on searches per
                                             request (default 5); a client
                                             asking for more is clamped, so a
                                             prompt cannot make the server
                                             issue unbounded outbound traffic
"""

import asyncio
import html
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote_plus, urlparse

import aiohttp

from tensorrt_llm.logger import logger

DEFAULT_MAX_RESULTS = 5
DEFAULT_TIMEOUT_S = 15.0
DEFAULT_MAX_USES = 5
DEFAULT_RETRIES = 2
DEFAULT_RETRY_BACKOFF_S = 0.5

# Mojeek returns plain HTML; results are <a class="title" href="..."> anchors
# followed by a <p class="s"> snippet. Parsed with regexes rather than an HTML
# parser to avoid adding a dependency for one provider.
_MOJEEK_RESULT_RE = re.compile(
    r'<a class="title"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.DOTALL,
)
_MOJEEK_SNIPPET_RE = re.compile(r'<p class="s">(?P<snippet>.*?)</p>', re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) TensorRT-LLM/anthropic-web-search"
)


class WebSearchError(RuntimeError):
    """A search query could not be completed."""


@dataclass
class WebSearchResult:
    url: str
    title: str
    snippet: str = ""
    page_age: Optional[str] = None


@dataclass
class WebSearchConfig:
    provider: str = "off"
    max_results: int = DEFAULT_MAX_RESULTS
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_uses: int = DEFAULT_MAX_USES
    retries: int = DEFAULT_RETRIES
    retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    # Populated from the request's tool definition, not the environment.
    allowed_domains: Sequence[str] = field(default_factory=tuple)
    blocked_domains: Sequence[str] = field(default_factory=tuple)

    @property
    def enabled(self) -> bool:
        return self.provider != "off"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %s", name, raw, default)
        return default


def load_web_search_config() -> WebSearchConfig:
    """Read the web-search configuration from the environment.

    Called per request so a provider can be switched without a restart; the
    work is a handful of ``os.environ`` lookups.
    """
    provider = (os.environ.get("TRTLLM_ANTHROPIC_WEB_SEARCH") or "off").strip().lower()
    config = WebSearchConfig(
        provider=provider,
        max_results=_env_int(
            "TRTLLM_ANTHROPIC_WEB_SEARCH_MAX_RESULTS", DEFAULT_MAX_RESULTS
        ),
        timeout_s=_env_float(
            "TRTLLM_ANTHROPIC_WEB_SEARCH_TIMEOUT_S", DEFAULT_TIMEOUT_S
        ),
        max_uses=_env_int("TRTLLM_ANTHROPIC_WEB_SEARCH_MAX_USES", DEFAULT_MAX_USES),
        retries=_env_int("TRTLLM_ANTHROPIC_WEB_SEARCH_RETRIES", DEFAULT_RETRIES),
    )
    if provider == "brave":
        config.api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
        config.endpoint = "https://api.search.brave.com/res/v1/web/search"
    elif provider == "tavily":
        config.api_key = os.environ.get("TAVILY_API_KEY")
        config.endpoint = "https://api.tavily.com/search"
    elif provider == "searxng":
        config.endpoint = os.environ.get("SEARXNG_URL")
    return config


def validate_web_search_config(config: WebSearchConfig) -> Optional[str]:
    """Return an error string if the selected provider cannot run, else None."""
    if not config.enabled:
        return None
    if config.provider not in _PROVIDERS:
        return (
            f"unknown web search provider {config.provider!r}; expected one of "
            "off, " + ", ".join(sorted(_PROVIDERS))
        )
    if config.provider == "brave" and not config.api_key:
        return "web search provider 'brave' requires BRAVE_SEARCH_API_KEY"
    if config.provider == "tavily" and not config.api_key:
        return "web search provider 'tavily' requires TAVILY_API_KEY"
    if config.provider == "searxng" and not config.endpoint:
        return "web search provider 'searxng' requires SEARXNG_URL"
    return None


def _strip_html(raw: str) -> str:
    return html.unescape(_TAG_RE.sub("", raw)).strip()


def _domain_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def _domain_matches(domain: str, pattern: str) -> bool:
    pattern = pattern.strip().lower().lstrip(".")
    if not pattern:
        return False
    return domain == pattern or domain.endswith("." + pattern)


def filter_results(
    results: Sequence[WebSearchResult], config: WebSearchConfig
) -> List[WebSearchResult]:
    """Apply the request's allowed/blocked domain lists.

    Anthropic treats these as mutually exclusive; if a caller sends both,
    ``allowed_domains`` wins because it is the more restrictive intent.
    """
    filtered: List[WebSearchResult] = []
    for result in results:
        domain = _domain_of(result.url)
        if not domain:
            continue
        if config.allowed_domains:
            if not any(_domain_matches(domain, p) for p in config.allowed_domains):
                continue
        elif config.blocked_domains:
            if any(_domain_matches(domain, p) for p in config.blocked_domains):
                continue
        filtered.append(result)
    return filtered


async def _fetch(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    **kwargs: Any,
) -> str:
    async with session.request(method, url, **kwargs) as response:
        body = await response.text()
        if response.status >= 400:
            raise WebSearchError(
                f"search backend returned HTTP {response.status}"
            )
        return body


async def _search_mojeek(
    session: aiohttp.ClientSession, query: str, config: WebSearchConfig
) -> List[WebSearchResult]:
    url = f"https://www.mojeek.com/search?q={quote_plus(query)}"
    body = await _fetch(session, "GET", url)
    titles = list(_MOJEEK_RESULT_RE.finditer(body))
    snippets = [m.group("snippet") for m in _MOJEEK_SNIPPET_RE.finditer(body)]
    results: List[WebSearchResult] = []
    for index, match in enumerate(titles):
        snippet = _strip_html(snippets[index]) if index < len(snippets) else ""
        results.append(
            WebSearchResult(
                url=html.unescape(match.group("url")),
                title=_strip_html(match.group("title")),
                snippet=snippet,
            )
        )
    return results


async def _search_brave(
    session: aiohttp.ClientSession, query: str, config: WebSearchConfig
) -> List[WebSearchResult]:
    body = await _fetch(
        session,
        "GET",
        config.endpoint,
        params={"q": query, "count": config.max_results},
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": config.api_key or "",
        },
    )
    payload = json.loads(body)
    return [
        WebSearchResult(
            url=item.get("url", ""),
            title=_strip_html(item.get("title", "")),
            snippet=_strip_html(item.get("description", "")),
            page_age=item.get("page_age"),
        )
        for item in (payload.get("web", {}).get("results") or [])
    ]


async def _search_tavily(
    session: aiohttp.ClientSession, query: str, config: WebSearchConfig
) -> List[WebSearchResult]:
    body = await _fetch(
        session,
        "POST",
        config.endpoint,
        json={
            "api_key": config.api_key,
            "query": query,
            "max_results": config.max_results,
        },
        headers={"Content-Type": "application/json"},
    )
    payload = json.loads(body)
    return [
        WebSearchResult(
            url=item.get("url", ""),
            title=_strip_html(item.get("title", "")),
            snippet=_strip_html(item.get("content", "")),
        )
        for item in (payload.get("results") or [])
    ]


async def _search_searxng(
    session: aiohttp.ClientSession, query: str, config: WebSearchConfig
) -> List[WebSearchResult]:
    body = await _fetch(
        session,
        "GET",
        config.endpoint.rstrip("/") + "/search",
        params={"q": query, "format": "json"},
    )
    payload = json.loads(body)
    return [
        WebSearchResult(
            url=item.get("url", ""),
            title=_strip_html(item.get("title", "")),
            snippet=_strip_html(item.get("content", "")),
        )
        for item in (payload.get("results") or [])
    ]


async def _search_wikipedia(
    session: aiohttp.ClientSession, query: str, config: WebSearchConfig
) -> List[WebSearchResult]:
    body = await _fetch(
        session,
        "GET",
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": config.max_results,
            "format": "json",
        },
    )
    payload = json.loads(body)
    results = []
    for item in payload.get("query", {}).get("search", []) or []:
        title = item.get("title", "")
        results.append(
            WebSearchResult(
                url="https://en.wikipedia.org/wiki/" + quote_plus(
                    title.replace(" ", "_")
                ),
                title=title,
                snippet=_strip_html(item.get("snippet", "")),
                page_age=item.get("timestamp"),
            )
        )
    return results


_PROVIDERS = {
    "mojeek": _search_mojeek,
    "wikipedia": _search_wikipedia,
    "brave": _search_brave,
    "tavily": _search_tavily,
    "searxng": _search_searxng,
}


async def run_web_search(
    query: str, config: WebSearchConfig
) -> List[WebSearchResult]:
    """Run one search query and return filtered, truncated results.

    Raises ``WebSearchError`` on any transport or backend failure; the caller
    turns that into a ``web_search_tool_result`` error block so the model can
    carry on rather than the whole request failing.
    """
    provider = _PROVIDERS.get(config.provider)
    if provider is None:
        raise WebSearchError(f"web search provider {config.provider!r} is not available")
    if not query or not query.strip():
        raise WebSearchError("web search query is empty")

    timeout = aiohttp.ClientTimeout(total=config.timeout_s)
    # Keyless providers drop connections intermittently - measured around half
    # of requests on one cluster - and a dropped connection is indistinguishable
    # from "no results" once it reaches the model. Retry transport failures so a
    # flaky hop does not silently become a wrong answer.
    last_error: Optional[Exception] = None
    for attempt in range(config.retries + 1):
        if attempt:
            await asyncio.sleep(config.retry_backoff_s * attempt)
        try:
            async with aiohttp.ClientSession(
                timeout=timeout, headers={"User-Agent": _USER_AGENT}
            ) as session:
                results = await provider(session, query, config)
            break
        except (aiohttp.ClientError, WebSearchError) as e:
            last_error = e
        except json.JSONDecodeError as e:
            raise WebSearchError(f"search backend returned invalid JSON: {e}") from e
        except (TimeoutError, asyncio.TimeoutError) as e:
            last_error = e
        logger.warning(
            "web search attempt %d/%d failed: %s",
            attempt + 1,
            config.retries + 1,
            last_error,
        )
    else:
        raise WebSearchError(
            f"search backend failed after {config.retries + 1} attempts: {last_error}"
        )

    results = [r for r in results if r.url]
    results = filter_results(results, config)
    return results[: config.max_results]


def results_as_tool_content(results: Sequence[WebSearchResult]) -> List[Dict[str, Any]]:
    """Shape results as Anthropic ``web_search_result`` blocks."""
    return [
        {
            "type": "web_search_result",
            "url": result.url,
            "title": result.title,
            "page_age": result.page_age,
            # Anthropic returns an opaque blob here that can be replayed in a
            # later turn. There is nothing to encrypt server-side, so the
            # snippet is passed through in the clear; clients treat the field
            # as opaque either way.
            "encrypted_content": result.snippet,
        }
        for result in results
    ]


def results_as_model_text(query: str, results: Sequence[WebSearchResult]) -> str:
    """Render results as the tool-result text handed back to the model."""
    if not results:
        return f'No results found for "{query}".'
    lines = [f'Search results for "{query}":', ""]
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result.title}")
        lines.append(f"   URL: {result.url}")
        if result.snippet:
            lines.append(f"   {result.snippet}")
        lines.append("")
    return "\n".join(lines).strip()
