# Anthropic Compatibility Fixtures

This directory stores sanitized wire fixtures captured from real clients and
route tests. Fixtures are compatibility evidence, not example secrets or raw
production transcripts.

## 1. Planned Fixture Sets

```text
claude_code_basic_chat.request.json
claude_code_basic_chat.response.json
claude_code_basic_stream.sse

claude_code_tool_loop.turn_1.request.json
claude_code_tool_loop.turn_1.response.json
claude_code_tool_loop.turn_2.request.json
claude_code_tool_loop.turn_2.response.json

claude_code_glean_mcp.turn_1.request.json
claude_code_glean_mcp.turn_1.response.json
claude_code_glean_mcp.turn_2.request.json
claude_code_glean_mcp.turn_2.response.json
```

Add a fixture only when it is used by a test or referenced as manual validation
evidence. Do not commit unused packet captures.

## 2. Required Metadata

Each fixture set must have a companion metadata file:

```text
<fixture-set>.metadata.json
```

The metadata should record:

```json
{
  "captured_at": "YYYY-MM-DD",
  "claude_code_version": "<version>",
  "tensorrt_llm_commit": "<commit>",
  "working_tree_changes": true,
  "model": "<checkpoint>",
  "serve_mode": "standard|disaggregated",
  "serve_options": ["<non-secret options>"],
  "related_trackers": ["P0-01"],
  "related_gaps": ["GAP-02"]
}
```

## 3. Sanitization Rules

Before committing a fixture:

- remove API keys, authorization headers, cookies, OAuth tokens, and session
  identifiers;
- replace user names, home directories, cluster names, and internal host names;
- remove or replace proprietary source code and document contents;
- replace internal URLs and Glean result links with stable placeholders;
- preserve message roles, content-block types, tool names, schemas, event order,
  stop reasons, and usage shapes needed by the test;
- preserve boundary conditions such as fragmented JSON or SSE bytes without
  preserving sensitive payload content;
- inspect both request and response fields, including metadata and nested tool
  results.

Use explicit placeholders such as:

```text
<REDACTED_TOKEN>
<INTERNAL_URL>
<PROJECT_PATH>
<DOCUMENT_CONTENT>
```

Do not use realistic-looking replacement secrets.

## 4. Capture Boundaries

For a client-side MCP flow, capture only the material needed to prove the
compatibility boundary:

```text
Claude Code -> /v1/messages request with ordinary MCP tool schemas
/v1/messages -> Claude Code tool_use response
Claude Code -> /v1/messages request with tool_result
/v1/messages -> Claude Code final response
```

Do not commit MCP authorization traffic or raw enterprise search results. A
separate local debug trace may confirm MCP `initialize`, `tools/list`, and
`tools/call`, but the committed fixture should contain sanitized summaries only.

## 5. Review Checklist

- [ ] Fixture is linked to a tracker and gap ID.
- [ ] Fixture is consumed by a test or documented manual validation.
- [ ] All credentials and identifiers are removed.
- [ ] Proprietary content is removed.
- [ ] Protocol structure needed by the test is preserved.
- [ ] Metadata records the exact validation environment.
- [ ] A second reviewer checks sanitization before commit.
