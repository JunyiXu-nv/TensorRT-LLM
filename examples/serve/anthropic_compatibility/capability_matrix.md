# Claude Messages API Capability Tracking

Status: **Draft v0.8; updated with the manual interactive CT-PROMPT-15 pass**

This document separates four concerns that should not be collapsed into one
table:

1. **Catalog:** what public Claude Platform capabilities exist.
2. **Tracker:** which capabilities are active TensorRT-LLM compatibility work.
3. **Acceptance:** what observable behavior proves each active capability.
4. **Evidence:** which tests and sanitized real-client fixtures support a claim.

The target compatibility profile is Claude Code using `POST /v1/messages` with
DeepSeek V4 served by `trtllm-serve`. This is not a commitment to reproduce the
entire Claude Platform.

Authoritative catalog sources:

- [Claude Platform introduction](https://platform.claude.com/docs/en/intro)
- [Features overview](https://platform.claude.com/docs/en/build-with-claude/overview)
- [Using the Messages API](https://platform.claude.com/docs/en/build-with-claude/working-with-messages)
- [Tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
- [Tool reference](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference)
- [Claude Code tools reference](https://code.claude.com/docs/en/tools-reference)

## 1. Tracking Model

### 1.1 Scope

| Scope | Meaning |
| --- | --- |
| `P0` | Required for the initial Claude Code plus DeepSeek V4 workflow. |
| `P1` | Broader Messages API compatibility after the P0 loop is reliable. |
| `P2` | Requires a new executor, endpoint family, persistent service, or agent loop. |
| `Out` | Explicitly outside the current TensorRT-LLM compatibility target. |
| `Review` | Scope has not been decided. |

### 1.2 Compatibility Disposition

Disposition records the intended product behavior, independently from delivery
progress.

| Disposition | Meaning |
| --- | --- |
| `target_supported` | Intended to provide equivalent observable behavior. |
| `target_best_effort` | Intended to run with a documented semantic difference. |
| `explicitly_rejected` | Intended to fail clearly and consistently. |
| `out_of_scope` | No implementation is planned for the current target. |
| `undecided` | Product behavior has not been selected. |

An explicitly rejected capability can be delivery-complete. Conversely, code
existing for a target-supported capability does not make it supported.

### 1.3 Delivery Stage

```text
inventory
contract_defined
mapping_implemented
unit_validated
route_validated
real_model_validated
claude_code_e2e
done
```

Stages only advance when the corresponding evidence has executed successfully.
A test file that exists but has not run is not `unit_validated`.

### 1.4 Overall Compatibility Result

| Result | Meaning |
| --- | --- |
| `supported` | Required behavior passed end-to-end validation. |
| `lossless_mapping` | Representation differs but observable behavior is equivalent. |
| `best_effort` | Request runs, but Claude-equivalent behavior is not guaranteed. |
| `adapter_unsupported` | The request is valid but the adapter cannot translate it. |
| `model_unsupported` | The adapter can translate it but the target model cannot honor it. |
| `configuration_error` | Required serving configuration is missing. |
| `pending_validation` | Implementation exists but required evidence is incomplete. |

## 2. Official Feature Catalog

This is the inventory layer. Public features remain listed even when they are
not current implementation targets. Availability reflects the Claude API entry
in the official overview at the time this draft was prepared; it is not a
TensorRT-LLM support claim.

### 2.1 Model Capabilities

| Catalog ID | Official feature | Claude API availability | Scope | Scope rationale |
| --- | --- | --- | --- | --- |
| CAT-MOD-01 | Context windows | GA | P0 | Claude Code sessions depend on predictable context limits. |
| CAT-MOD-02 | Adaptive thinking | GA | P1 | Current DeepSeek mapping is only an approximation. |
| CAT-MOD-03 | Batch processing | GA | Out | Separate asynchronous endpoint; not required by Claude Code. |
| CAT-MOD-04 | Citations | GA | P1 | Requires citation-aware input and output blocks. |
| CAT-MOD-05 | Data residency | GA | Out | Anthropic infrastructure routing has no local TRT equivalent. |
| CAT-MOD-06 | Effort | GA | P1 | Field can be forwarded, but model behavior needs evaluation. |
| CAT-MOD-07 | Extended thinking | GA | P0 | Claude Code sends thinking controls and history. |
| CAT-MOD-08 | Fallback credit | Beta | Out | Anthropic billing and cache-credit mechanism. |
| CAT-MOD-09 | PDF support | GA | P1 | Requires document parsing or a compatible multimodal path. |
| CAT-MOD-10 | Search results | GA | P1 | Requires search-result content and citation semantics. |
| CAT-MOD-11 | Server-side fallback | Beta | P2 | Requires model routing and refusal-aware retries. |
| CAT-MOD-12 | Structured outputs | GA | P1 | Mapping exists; real JSON Schema enforcement is unverified. |

### 2.2 Server-Side Tools

| Catalog ID | Official feature | Claude API availability | Scope | Scope rationale |
| --- | --- | --- | --- | --- |
| CAT-STOOL-01 | Advisor tool | Beta | P2 | Requires a server executor and model orchestration. |
| CAT-STOOL-02 | Code execution | GA | P2 | Requires a secured execution environment and agent loop. |
| CAT-STOOL-03 | Web fetch | GA | P2 | Requires server-side network execution and content blocks. |
| CAT-STOOL-04 | Web search | GA | P2 | Requires search execution, result blocks, and citations. |

### 2.3 Client-Side Tools

| Catalog ID | Official feature | Claude API availability | Scope | Scope rationale |
| --- | --- | --- | --- | --- |
| CAT-CTOOL-01 | Bash | GA | P0 | Claude Code may send the Anthropic versioned Bash schema. |
| CAT-CTOOL-02 | Computer use | Beta | P1 | Not required for the initial terminal-only profile. |
| CAT-CTOOL-03 | Memory | GA | P1 | Requires authoritative schema and real-client traffic. |
| CAT-CTOOL-04 | Text editor | GA | P0 | Claude Code may send the Anthropic versioned editor schema. |

### 2.4 Tool Infrastructure

| Catalog ID | Official feature | Claude API availability | Scope | Scope rationale |
| --- | --- | --- | --- | --- |
| CAT-INF-01 | Agent Skills | Beta | P2 | Anthropic-hosted Skills are not Claude Code plugin skills. |
| CAT-INF-02 | Fine-grained tool streaming | GA | P1 | Current fragments need validation against the exact contract. |
| CAT-INF-03 | MCP Connector | Beta | P2 | Server-side MCP requires an MCP client and executor loop. |
| CAT-INF-04 | Programmatic tool calling | GA | P2 | Requires server-side code execution and orchestration. |
| CAT-INF-05 | Tool search | GA | P1 | Deferred tool references are not currently supported. |

### 2.4.1 Client-Tool and Skill Execution Boundaries

Catalog availability describes the public Claude API, not current
TensorRT-LLM support. The following distinctions are required when interpreting
the catalog and designing benchmarks:

| Capability | What it does | Standard wire flow | Current TensorRT-LLM compatibility status |
| --- | --- | --- | --- |
| `CAT-CTOOL-02` Computer use | Lets the model inspect screenshots and request mouse movement, clicks, keyboard input, and other desktop actions. The caller owns a sandboxed desktop and performs every action. Bash and text editor tools can be supplied alongside it but are separate tools. | The request supplies a supported versioned `computer_*` definition. The model emits `tool_use`; the client performs the action and commonly returns screenshot/image content in `tool_result`; the loop repeats. | **Unsupported.** The type is recognized but its authoritative built-in schema is not implemented, so the standard schema-less request is rejected. In addition, image content inside `tool_result` is currently rejected, and no desktop executor is provided. Scope remains P1. |
| `CAT-CTOOL-03` Memory | Gives the model persistent, application-owned storage under `/memories`. It can view, create, replace, insert, delete, and rename memory files so selected facts or progress survive across conversations. This is not the same feature as Claude Code's local auto-memory. | The request supplies exactly `{"type":"memory_20250818","name":"memory"}`. The model emits file-operation `tool_use` blocks; the client validates paths, executes them against its persistent store, and returns `tool_result`. | **Unsupported.** The type is recognized but the built-in schema is not implemented, and this server provides no `/memories` persistence or executor. An explicit caller-supplied `input_schema` can pass through the generic mapper, but that does not establish compatibility with the official schema or persistence behavior. Scope remains P1. |
| `CAT-CTOOL-04` Text editor | Lets the model view and modify text files through the Anthropic-defined commands, including viewing, creating, inserting, and exact string replacement. The client owns filesystem access and execution. | The request supplies `text_editor_20250728` or `text_editor_20250124`, normally named `str_replace_based_edit_tool`, without redefining Anthropic's built-in schema. The client executes each requested file operation and returns text results. | **Official versioned flow is unsupported.** The type is recognized, but a normal schema-less request is rejected because the built-in schema registry is pending. **The separate Claude Code `Read`, `Edit`, and `Write` generic-tool flow is mapping-implemented**, because Claude Code supplies complete schemas; it is not evidence for `CAT-CTOOL-04` and still needs named, sanitized E2E fixtures. |
| Claude Code filesystem Skill | Packages reusable instructions in `SKILL.md`, with optional references, scripts, and assets. Claude Code discovers it locally and invokes it through its ordinary `Skill` client tool when relevant or explicitly requested. | Claude Code loads the skill from its filesystem and may advertise a normal user-defined `Skill` tool in `tools[]`; subsequent file, shell, or other operations use Claude Code's ordinary client tools. | **Generic mapping is expected but unvalidated.** A complete client-supplied `Skill` definition should travel through `EXT-TOOL-01`; `CT-PROMPT-17` defines the required named fixture, but no captured request, unit result, or real E2E currently proves it. This is distinct from `CAT-INF-01`. |
| `CAT-INF-01` Anthropic API Agent Skills | Supplies Anthropic-hosted or uploaded skill packages to an Anthropic code-execution container, for example the `pptx`, `xlsx`, `docx`, and `pdf` skills. | The request uses `container.skills`, a Skills beta header, and an Anthropic `code_execution_*` server tool; custom skills also require Skills/Files lifecycle APIs. | **Unsupported and P2/outside the current target.** `container.skills` is not translated, `code_execution_*` is explicitly rejected as a server tool, and this server has no Skills API, Files API, container, or execution loop. |

The `nemo_skills` evaluation package elsewhere in this repository is unrelated
to Anthropic Agent Skills and must not be used as compatibility evidence.

### 2.5 Context Management

| Catalog ID | Official feature | Claude API availability | Scope | Scope rationale |
| --- | --- | --- | --- | --- |
| CAT-CTX-01 | Compaction | Beta | P2 | Anthropic server-side summarization is not implemented. |
| CAT-CTX-02 | Context editing | Beta | P2 | Requires server-side context transformation policy. |
| CAT-CTX-03 | Automatic prompt caching | GA | P1 | Backend caching is not equivalent to Anthropic semantics. |
| CAT-CTX-04 | Prompt caching, 5 minutes | GA | P1 | Cache-control placement and usage need explicit support. |
| CAT-CTX-05 | Prompt caching, 1 hour | GA | P1 | Requires duration-aware cache behavior. |
| CAT-CTX-06 | Token counting | GA | P1 | Protocol models exist, but the route is not registered. |

### 2.6 Files and Assets

| Catalog ID | Official feature | Claude API availability | Scope | Scope rationale |
| --- | --- | --- | --- | --- |
| CAT-FILE-01 | Files API | Beta | P2 | Requires upload, storage, authorization, and lifecycle APIs. |

### 2.7 Required Compatibility Extensions

These capabilities are required for the target integration but are not separate
rows in the official feature overview table.

| Extension ID | Capability | Scope | Reason |
| --- | --- | --- | --- |
| EXT-MSG-01 | Core Messages request and response contract | P0 | Base protocol used by every other feature. |
| EXT-MSG-02 | Streaming Messages | P0 | Claude Code relies on Anthropic SSE event semantics. |
| EXT-TOOL-01 | User-defined client tools | P0 | Foundation of the client-executed tool loop. |
| EXT-MCP-01 | Claude Code client-side MCP | P0 | Claude Code discovers and executes Glean as ordinary tools. |
| EXT-ROUTE-01 | Standard and disaggregated route parity | Out | Both routes share the adapter, but real parity is not required by the current disaggregated acceptance scope. |

## 3. Active Capability Tracker

Only active P0/P1 delivery work belongs in this table. The catalog above remains
the source for deferred and out-of-scope features.

| Tracker ID | Capability | Priority | Disposition | Delivery stage | Overall result | Current evidence | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0-01 | Core Messages | P0 | `target_supported` | `claude_code_e2e` | `pending_validation` | Current adapter/route batches pass; real disaggregated chat returned HTTP 200 through Claude Code | Add sanitized wire fixtures and broader request validation. |
| P0-02 | Conversation and system semantics | P0 | `target_best_effort` | `mapping_implemented` | `best_effort` | Current conversion tests pass; real sessions exercised multi-turn history | Capture sanitized Claude Code system/history traffic and validate ordering. |
| P0-03 | Streaming Messages | P0 | `target_supported` | `claude_code_e2e` | `pending_validation` | GAP-10 is fixed: parser smoke cases plus OpenAI, Anthropic, and Claude Code streaming E2E preserve the complete EOS-adjacent tail; normal completion still logs noisy internal `GeneratorExit` tracebacks | Add split UTF-8/line, disconnect, post-HTTP-200 fault, and normal-generator-close cases. |
| P0-04 | Client tool use | P0 | `target_supported` | `claude_code_e2e` | `pending_validation` | With corrected CT-PROMPT-10 semantics and the manual interactive CT-PROMPT-15 pass, the post-fix Pro suite has 9 strict passes, 6 partials, 1 tool-selection failure, and 1 environment-unsupported result | Resolve the residual strict formatting, ordering, and file-side-effect cases without conflating them with the closed stream-tail defect. |
| P0-05 | Claude Code client-side MCP | P0 | `target_supported` | `claude_code_e2e` | `pending_validation` | An earlier exact Slurm MCP loop passed, but the post-fix full rerun called `Bash` instead while the MCP server was connected and its tool advertised | Run a focused repeated MCP selection test, capture sanitized traffic, and run the intended NVIDIA MaaS Glean loop. |
| P0-06 | Extended thinking | P0 | `target_best_effort` | `mapping_implemented` | `best_effort` | Current adapter tests pass and thinking appeared in real traces | Evaluate enabled/disabled history, budgets, and effort as model behaviors. |
| P0-07 | Stop, usage, and error semantics | P0 | `target_supported` | `route_validated` | `pending_validation` | Current adapter/route error batches pass | Complete usage, request IDs, auth/rate/timeout mapping, and SSE failure tests. |
| P1-01 | Structured outputs | P1 | `target_supported` | `claude_code_e2e` | `pending_validation` | Claude Code `--json-schema` returned the constrained object through guided decoding | Validate complex schemas and combinations with tools/reasoning. |
| P1-02 | Adaptive thinking and effort | P1 | `target_best_effort` | `mapping_implemented` | `best_effort` | Prompt behavior is partially documented | Run fixed-seed model comparisons. |
| P1-03 | Token counting | P1 | `target_supported` | `contract_defined` | `adapter_unsupported` | Request/response models exist; no route | Implement the endpoint with the generation template path. |
| P1-04 | Tool search and deferred loading | P1 | `undecided` | `inventory` | `adapter_unsupported` | Unsupported server-tool classification exists | Define `tool_reference` contract and model strategy. |

## 4. P0 Acceptance Checklists

Checklists track detailed protocol and edge-case work beneath each public
capability. An implementation checkbox means code is present; a validation
checkbox is checked only after the command or end-to-end scenario passes.

### P0-01 Core Messages

Implementation:

- [x] Parse the base Anthropic Messages request.
- [x] Convert text messages into the shared chat-completions pipeline.
- [x] Convert non-streaming chat responses into Anthropic messages.
- [x] Expose `/v1/messages` on standard and disaggregated servers.

Validation:

- [x] Current adapter unit batch passes in the TensorRT-LLM development image.
- [x] Current standard route test passes.
- [x] Current disaggregated route test passes.
- [x] Real DeepSeek V4 text generation passes.
- [x] Claude Code ordinary chat passes.

### P0-02 Conversation and System Semantics

Implementation:

- [x] Preserve top-level text system content.
- [x] Preserve multi-turn user and assistant history.
- [x] Merge accepted inline system messages into the top-level system content.
- [x] Carry historical assistant reasoning into the model representation.

Validation:

- [ ] Capture real Claude Code system and history requests.
- [ ] Validate legal role/content combinations.
- [ ] Validate ordering around thinking, text, tools, and results.
- [ ] Document where inline-system merging differs from Claude behavior.

### P0-03 Streaming Messages

Implementation:

- [x] Emit Anthropic message and content-block SSE events.
- [x] Reframe text, thinking, and tool-call deltas.
- [x] Emit final stop reason and usage.
- [x] Buffer fragmented upstream SSE records.

Validation:

- [x] Current SSE unit batch passes.
- [ ] Split UTF-8 and split-line transport tests pass.
- [ ] Tool JSON fragments concatenate into the intended object.
- [ ] Errors after HTTP 200 produce an Anthropic SSE error event.
- [ ] Client disconnect and upstream termination behavior pass.
- [x] Tool-enabled DeepSeek V4 streams flush EOS-bearing final normal text.
- [x] Claude Code consumes real streaming responses without recovery errors.

### P0-04 Client Tool Use

Implementation:

- [x] Preserve ordinary tool names, descriptions, and complete input schemas.
- [x] Convert generated tool calls into Anthropic `tool_use` blocks.
- [x] Convert text `tool_result` history for the model.
- [x] Make client-reported tool errors visible to the model.
- [x] Reject malformed generated tool arguments instead of replacing them.
- [x] Reject unsupported result media instead of silently flattening it.
- [ ] Implement authoritative schemas for required versioned Bash and Text Editor tools.

Validation:

- [x] Current generic tool unit batch passes.
- [ ] Named, automatic, disabled, strict, and parallel behavior is classified.
- [ ] Non-streaming DeepSeek V4 tool selection passes.
- [x] Streaming DeepSeek V4 tool selection passes for the Bash E2E case.
- [x] Claude Code executes the selected tool and submits its result.
- [x] Final generation uses the submitted result.

#### P0-04 Client-Tool Prompt Benchmarks

In this section, a client tool is any tool that Claude Code advertises in the
request and executes locally after receiving an assistant `tool_use` block.
This includes Claude Code built-ins such as `Read` and namespaced MCP tools. It
is distinct from Anthropic server-executed tools, and from the versioned
Anthropic-schema tools in Section 2.3. For example, Claude Code's ordinary
`Bash` tool and the Anthropic-schema `bash_20250124` tool must be recorded as
different wire contracts even though both run shell commands on the client.

Catalog mapping is determined by the request's wire definition, not by a
similar display name or capability:

| Request tool definition | Catalog or extension | Prompt cases | Classification rule |
| --- | --- | --- | --- |
| `{"type":"bash_20250124","name":"bash"}` | `CAT-CTOOL-01` Bash | `CT-CATALOG-01` | Anthropic supplies the versioned schema; the client supplies a persistent shell executor. |
| `{"type":"memory_20250818","name":"memory"}` | `CAT-CTOOL-03` Memory | `CT-CATALOG-02` | Anthropic supplies the versioned schema; the client supplies persistent storage rooted at `/memories`. |
| `{"type":"text_editor_20250728","name":"str_replace_based_edit_tool"}` or the earlier `text_editor_20250124` type | `CAT-CTOOL-04` Text editor | Not yet defined | This is not the same contract as Claude Code's separate `Read`, `Edit`, and `Write` tools. |
| A supported versioned `computer_*` definition named `computer` | `CAT-CTOOL-02` Computer use | Not yet defined | This requires screenshot, mouse, and keyboard execution by the client. |
| User-defined `{"name":"Bash","description":...,"input_schema":...}` | `EXT-TOOL-01` User-defined client tools | `CT-PROMPT-01`, `07` | This is the ordinary Claude Code built-in tested by the existing Bash E2E; capitalization is significant. It does not prove `CAT-CTOOL-01`. |
| User-defined `Read`, `Glob`, `Grep`, `Write`, `Edit`, `NotebookEdit`, `Agent`/`Task`, task-list, or interactive definitions | `EXT-TOOL-01` User-defined client tools | `CT-PROMPT-02` through `15`, except `16` | Preserve each client-supplied name and complete schema; do not translate them into versioned Anthropic-schema tools. |
| User-defined Claude Code `WebFetch` or `WebSearch` definitions | `EXT-TOOL-01` User-defined client tools | `CT-PROMPT-11`, `12` | These are client-side only when Claude Code advertises them as ordinary definitions. A versioned `web_fetch_*` or `web_search_*` type is instead the corresponding server-side catalog feature. |
| A namespaced `mcp__<server>__<tool>` definition supplied by Claude Code | `EXT-MCP-01` Claude Code client-side MCP | `CT-PROMPT-16` | Claude Code owns the MCP connection and execution; this is not the server-side `mcp_toolset` connector. |

Consequently, none of `CT-PROMPT-01` through `16` validates catalog Memory,
and the existing `Bash` cases validate `EXT-TOOL-01`, not catalog Bash. The two
catalog-specific cases below require an API harness that sends the exact
versioned tool definition and supplies the corresponding client executor.

The exact built-in set varies with the Claude Code version, platform, enabled
features, and connected MCP servers. Before running these cases, capture the
actual `tools[]` array from the client under test. The initial evidence used
Claude Code 2.1.145; a later client must be treated as a new matrix dimension.

Run every file-mutating case in a fresh temporary workspace. The benchmark
runner substitutes `<RUN_DIR>` with that workspace's absolute path and creates
the stated prerequisites. A final textual answer is not sufficient evidence:
the pass criteria must be checked against the tool trace and filesystem or
network result.

| Case | Client tool(s) | Prerequisite | Test prompt | Deterministic pass signal |
| --- | --- | --- | --- | --- |
| CT-PROMPT-01 | `Bash` | Start Claude Code in the repository root. | `Use the Bash tool exactly once to run pwd. Do not infer the path. Return only BASH_OK:<the exact stdout without the trailing newline>.` | A `Bash` `tool_use` runs `pwd`; its `tool_result` is returned on the next request; the final path equals the runner's working directory. **Passed streaming E2E on Claude Code 2.1.145.** |
| CT-PROMPT-02 | `Read` | The current repository contains this matrix. | `Use Read, not Bash, to read examples/serve/anthropic_compatibility/capability_matrix.md. Return only its exact first Markdown heading.` | The trace contains `Read` and no `Bash`; the answer is `# Claude Messages API Capability Tracking`. |
| CT-PROMPT-03 | `Glob` | The current repository contains `examples/serve/anthropic_compatibility/`. | `Use Glob, not Bash, with the pattern examples/serve/anthropic_compatibility/*.md. Return the matching basenames in lexicographic order, one per line.` | The trace contains `Glob`; every returned path is an actual match; no path is omitted or invented. |
| CT-PROMPT-04 | `Grep` | The current matrix contains `EXT-TOOL-01`. | `Use Grep, not Bash or Read, to search for the literal EXT-TOOL-01 under examples/serve/anthropic_compatibility. Return only the relative file path and matching line.` | The trace contains `Grep`; the result identifies this matrix and the exact matching line. |
| CT-PROMPT-05 | `Write` | `<RUN_DIR>/write.txt` does not exist. | `Use Write, not Bash, to create <RUN_DIR>/write.txt with exactly WRITE_OK_7F3A followed by one newline. Do not modify any other file. Then reply only done.` | The trace contains one `Write`; the file has exactly 14 bytes and the expected content; no other file changes. |
| CT-PROMPT-06 | `Edit` | Create `<RUN_DIR>/edit.txt` with `mode=before` and `keep=unchanged` on separate newline-terminated lines. | `Use Edit, not Bash or Write, to change only mode=before to mode=after in <RUN_DIR>/edit.txt. Then reply only done.` | The trace contains one `Edit`; the first line changes and the second line and final newline remain byte-for-byte unchanged. |
| CT-PROMPT-07 | `Read`, `Edit`, `Bash` | Copy a tiny tested project into `<RUN_DIR>` and inject one known one-line defect. | `Inspect <RUN_DIR> with Read, fix the known defect with Edit, and run the project's single test with Bash. Do not use Write. Report only TEST_OK if the real test passes.` | The trace shows the ordered read-edit-test loop; the Bash `tool_result` has exit code zero; the resulting diff contains only the expected edit. |
| CT-PROMPT-08 | parallel `Read` | Create `<RUN_DIR>/a.txt` containing `ALPHA_17` and `<RUN_DIR>/b.txt` containing `BETA_29`. | `Read <RUN_DIR>/a.txt and <RUN_DIR>/b.txt in parallel using two Read calls in the same assistant turn. Return only ALPHA_17\|BETA_29.` | Two `tool_use` blocks are emitted before their matching `tool_result` blocks; IDs are paired correctly; the final value is exact. |
| CT-PROMPT-09 | error recovery with `Read` and `Glob` | Create only `<RUN_DIR>/recovery.txt` containing `RECOVERED_43`. | `First use Read on <RUN_DIR>/missing.txt. After observing the real error, use Glob to find <RUN_DIR>/*.txt, then Read the existing file and return only its content.` | The missing-file result is visible to the model as an error; it then calls `Glob` and `Read`; the final answer is `RECOVERED_43`. |
| CT-PROMPT-10 | `Read`, `NotebookEdit` | Create `<RUN_DIR>/bench.ipynb` with a code cell whose stable cell ID is `bench-cell` and source is `value = 1`. | `First use Read to read <RUN_DIR>/bench.ipynb. After observing the Read result, use NotebookEdit exactly once to replace the source of cell bench-cell with value = 2. Do not use Bash, Edit, or Write. Then reply only done.` | The `Read` result is returned before exactly one successful `NotebookEdit`; the final answer is `done`. After parsing both notebooks as JSON and normalizing the target cell's permitted string-or-string-array `source` representation, only `bench-cell.source` changes and its normalized value is `value = 2`. JSON serialization details such as terminal whitespace are not adapter acceptance conditions. |
| CT-PROMPT-11 | `WebFetch` | Client web access is enabled. | `Use WebFetch exactly once to fetch https://example.com/. Return only the page title.` | The trace contains a real `WebFetch`; the returned content, not model memory, supports `Example Domain`. A policy or network denial is an environment result, not an adapter pass. |
| CT-PROMPT-12 | `WebSearch` | Client web search is enabled. | `Use WebSearch exactly once to search for the official NVIDIA TensorRT-LLM GitHub repository. Return only the repository URL.` | The trace contains a real `WebSearch` and a result containing `github.com/NVIDIA/TensorRT-LLM`. The earlier no-web SWE-bench runs do not satisfy this case. |
| CT-PROMPT-13 | `Agent` or legacy `Task` | Subagents are enabled; record which name appears in the captured `tools[]`. | `Use the available subagent tool exactly once. Ask the subagent to return only SUBAGENT_OK_314159, then return that token unchanged. Do not solve the task yourself.` | The advertised subagent tool is called, its result is submitted, and the final token is exact. Record `Agent` and legacy `Task` separately rather than treating them as aliases in the adapter. |
| CT-PROMPT-14 | `TaskCreate`, `TaskList`, `TaskUpdate`, `TaskGet` | The task-list feature is enabled. | `Create one task named verify-client-tool, list the tasks, mark that task completed, then get it by ID. Return only TASK_OK if its final status is completed.` | All four named calls occur with one consistent task ID and the final client state is `completed`. Record legacy `TodoWrite` as a separate versioned-client case if it is advertised. |
| CT-PROMPT-15 | `AskUserQuestion` | Run interactively, not with unattended `claude -p`. | `Use AskUserQuestion to ask me to choose exactly one value: red or blue. After I answer, return only CHOICE:<selected value>.` | A real client question is shown, the selected value returns through the tool result, and the final answer matches it. This is a manual interactive case. |
| CT-PROMPT-16 | namespaced MCP | Connect and allowlist one read-only MCP tool; the current fixture uses `mcp__slurm-broker__slurm_my_jobs`. | `Use mcp__slurm-broker__slurm_my_jobs exactly once, then return only MCP_OK:<number of jobs in the real result>.` | The exact namespaced name survives conversion; the client executes it; the matching result is used in the second request. **The generic Slurm MCP loop passed; the intended MaaS Glean tool remains pending.** |
| CT-PROMPT-17 | Claude Code `Skill` | Before starting Claude Code, install the `client-tool-sentinel` fixture shown below at `.claude/skills/client-tool-sentinel/SKILL.md`. Ensure `Skill` is not denied by client policy. | `Use the Skill tool exactly once to invoke client-tool-sentinel. Do not answer from its description and do not use any other tool. After the skill finishes, return only the exact token required by the skill.` | The captured request advertises `Skill` with a complete client-supplied schema; the assistant emits exactly one `Skill` `tool_use` selecting `client-tool-sentinel`; the client loads the skill body; the final response is exactly `SKILL_OK_83D1`. The case fails if the token is produced without a `Skill` call. |
| CT-CATALOG-01 | `CAT-CTOOL-01` versioned Bash | The API harness sends exactly `{"type":"bash_20250124","name":"bash"}` and owns a persistent shell rooted at `<RUN_DIR>`. | `Call bash exactly twice. In the first call, change directory to <RUN_DIR> and export CT_BASH_TOKEN=BASH_STATE_29. In the second call, print the token and current directory as <token>:<directory>. Return only the second command's stdout.` | Both responses use tool name `bash`; the two calls share one shell session; the final value is exactly `BASH_STATE_29:<RUN_DIR>`. No client-supplied `input_schema` is added or substituted. |
| CT-CATALOG-02 | `CAT-CTOOL-03` Memory | The API harness sends exactly `{"type":"memory_20250818","name":"memory"}` and maps a fresh persistent store to `/memories`. Run phase B as a new conversation against the same store. | **Phase A:** `Remember the exact benchmark value MEMORY_OK_5C91 in /memories/client-tool-benchmark.txt. Use the memory tool, then return only SAVED.` **Phase B:** `Retrieve the benchmark value saved in memory. Use the memory tool and return only that value.` | Phase A uses `memory` to inspect and create/update the file; phase B performs a real memory view in a fresh conversation and returns exactly `MEMORY_OK_5C91`. Every path stays under `/memories`; storage persists between phases. |

#### 2026-07-19 DeepSeek-V4-Pro pre-fix execution status

The following run used Claude Code 2.1.145, the aggregated TP8
DeepSeek-V4-Pro server in Slurm job `5492420`, and the prompts above without
relaxing their deterministic pass criteria. Full local evidence and exact
observations are in the
[run result](../../../../../runs/anthropic_capability_prompt_bench_20260719/RESULTS.md).
The raw traces are intentionally outside the repository because they have not
been sanitized.

`PARTIAL` means that a real client tool was selected and executed, but the
complete case failed an ordering, side-effect, or exact-final-answer criterion.
It is neither an adapter pass nor evidence that the tool path is absent.

| Case | Pre-fix result | Key observation |
| --- | --- | --- |
| CT-PROMPT-01 | `PARTIAL` | One real `Bash pwd` completed; final path omitted `/TensorRT-LLM`. |
| CT-PROMPT-02 | `PARTIAL` | `Read` returned the heading; final text was only `` `# ``. |
| CT-PROMPT-03 | `PARTIAL` | `Glob` executed and retried after timeout; the final top-level list omitted `usage.md`. |
| CT-PROMPT-04 | `PARTIAL` | `Grep` returned real matches, but the answer did not obey the single-line output contract. |
| CT-PROMPT-05 | `PARTIAL` | `Write` created the file without the required final newline (13 rather than 14 bytes). |
| CT-PROMPT-06 | `PARTIAL` | The one-line edit was byte-correct; final text was `Done.` rather than `done`. |
| CT-PROMPT-07 | `PARTIAL` | The read-edit-test loop fixed the one-line defect and the real test passed; final `TEST_OK` was absent. |
| CT-PROMPT-08 | `PARTIAL` | Two `Read` calls were parallel and paired correctly; final text omitted `BETA_29`. |
| CT-PROMPT-09 | `PARTIAL` | Error propagation and recovery tools worked, but `Glob` was issued before observing the missing-file result. |
| CT-PROMPT-10 | `PARTIAL` | `NotebookEdit` changed the target cell and left valid JSON; final text was empty. |
| CT-PROMPT-11 | `PARTIAL` | A real `WebFetch` returned `Example Domain`; no final title was emitted. |
| CT-PROMPT-12 | `ENVIRONMENT_UNSUPPORTED` | Claude Code's `WebSearch` execution invoked unsupported server tool `web_search_20250305`; no real search occurred. |
| CT-PROMPT-13 | `PARTIAL` | One real `Agent` ran; nested and parent exact sentinel outputs were incomplete. |
| CT-PROMPT-14 | `PARTIAL` | All four task calls used one ID and reached `completed`; final `TASK_OK` was absent. |
| CT-PROMPT-15 | `PARTIAL` | The real question UI returned `Red`; final text was only `CHOICE:`. |
| CT-PROMPT-16 | `PARTIAL` | The exact MCP tool returned one real RUNNING job; final text was only `M`. |
| CT-PROMPT-17 | `PARTIAL` | `Skill` selected and launched the installed fixture; final text was only `SKILL`. |
| CT-CATALOG-01 | `EXPECTED_REJECTION` | Exact `bash_20250124` request returned HTTP 400 because the built-in schema is not implemented. |
| CT-CATALOG-02 | `EXPECTED_REJECTION` | Exact `memory_20250818` request returned HTTP 400 because the built-in schema is not implemented. |

Strict prompt aggregate: 0 pass, 16 partial, and 1 environment-unsupported.
A fixed-output control without tools returned complete text in both modes. A
second identical tool-history comparison returned complete non-streaming text
but truncated streaming text with full output-token usage. Equivalent OpenAI
streaming truncated before Anthropic reframing. Code inspection matches the
observed boundary: `DeepSeekV4Parser` inherits a streaming path that buffers a
final normal-text chunk containing raw DeepSeek EOS and never flushes that
buffer. Therefore the repeated incomplete finals are a confirmed streaming
tool-parser defect, not current evidence of shortened model generation.

The `CT-PROMPT-17` fixture is intentionally minimal. Its description does not
contain the expected token, so the model cannot pass from discovery metadata
alone:

```markdown
---
name: client-tool-sentinel
description: A deterministic compatibility benchmark skill. Use only when explicitly asked to run the client-tool-sentinel benchmark.
---

Return exactly `SKILL_OK_83D1`. Do not call any other tool and do not add any
other text.
```

Run this case by sending the table's natural-language prompt after Claude Code
starts and discovers the fixture. Invoking `/client-tool-sentinel` directly is
a useful client-discovery smoke test, but it may expand the skill without a
model-generated `Skill` call and therefore does not satisfy `CT-PROMPT-17`.

#### 2026-07-19 GAP-10 post-fix validation

Persistent server attempt 5 loaded the editable-source fix and passed direct
non-stream/stream comparisons through both `/v1/chat/completions` and
`/v1/messages`. All four tool-enabled responses ended in the complete value
`BASH_OK:.../TensorRT-LLM`; both streaming routes carried `/TensorRT-LLM` in
their final content delta and removed the raw EOS token.

Claude Code 2.1.145 then reran CT-PROMPT-01, selected `Bash` exactly once,
executed real `pwd`, submitted the tool result, and returned the complete exact
path with exit code 0. Full local evidence is in
[the GAP-10 E2E run](../../../../../runs/dsml_streaming_fix_e2e_20260719/RESULTS.md).

#### 2026-07-19 full post-fix prompt rerun

The fixed persistent server then reran every non-interactive prompt case, and
the user later completed CT-PROMPT-15 in a real interactive Claude Code
terminal. Full evidence is in the
[post-fix run result](../../../../../runs/anthropic_capability_prompt_bench_postfix_20260719/RESULTS.md).

| Case | Post-fix result | Key observation |
| --- | --- | --- |
| CT-PROMPT-01 | `PASS` | One real `Bash pwd`; exact complete path including `/TensorRT-LLM`. |
| CT-PROMPT-02 | `PARTIAL` | Correct `Read` result, but the exact heading was wrapped in a code fence. |
| CT-PROMPT-03 | `PARTIAL` | `Glob` timed out and the model recovered with forbidden `Bash`; complete list was fenced. |
| CT-PROMPT-04 | `PARTIAL` | Real `Grep`, but a formatted multi-match answer violated the requested output shape. |
| CT-PROMPT-05 | `PARTIAL` | One `Write` and exact `done`; required final newline was still absent. |
| CT-PROMPT-06 | `PARTIAL` | Edit bytes were correct; extra `Read` and final `Done.` violated the strict contract. |
| CT-PROMPT-07 | `PASS` | Exact one-line fix, real passing `unittest` fallback, and exact `TEST_OK`. |
| CT-PROMPT-08 | `PASS` | Two parallel `Read` calls with correct pairing and exact combined output. |
| CT-PROMPT-09 | `PARTIAL` | Recovery and final value were correct, but `Glob` was issued before observing the `Read` error. |
| CT-PROMPT-10 | `PASS` | Corrected targeted rerun observed `Read` before one `NotebookEdit`, exact `done`, exact normalized target source, and equality of every other parsed JSON value. |
| CT-PROMPT-11 | `PASS` | One real client-side `WebFetch`; exact `Example Domain`. |
| CT-PROMPT-12 | `ENVIRONMENT_UNSUPPORTED` | Claude Code still attempted unsupported `web_search_20250305`; the URL was not supported by a real search result. |
| CT-PROMPT-13 | `PASS` | One real `Agent`; nested and parent sentinel outputs were complete and exact. |
| CT-PROMPT-14 | `PASS` | All four task calls shared ID 1, reached `completed`, and returned exact `TASK_OK`. |
| CT-PROMPT-15 | `PASS` | Real Red/Blue UI, `Red` returned through the matching `tool_result`, and exact `CHOICE:Red`. |
| CT-PROMPT-16 | `FAIL` | MCP was connected and the tool advertised, but the model invoked `Bash` instead of the MCP tool. |
| CT-PROMPT-17 | `PASS` | One real `Skill`; fixture loaded and exact `SKILL_OK_83D1` returned. |

Strict aggregate across all 17 prompt cases: 9 pass, 6 partial, 1 fail, and 1
environment-unsupported. No post-fix partial or failure showed the former
stream-tail truncation signature: complete final strings, including long
sentinels and `/TensorRT-LLM`, survived Claude Code streaming. The remaining
results are instruction-following, side-effect/ordering, executor, or tool
selection issues rather than GAP-10 regressions.

The corrected CT-PROMPT-10 evidence is in the
[semantic NotebookEdit rerun](../../../../../runs/anthropic_ct10_semantic_rerun_20260719/RESULTS.md).

`ToolSearch`, `ListMcpResourcesTool`, `ReadMcpResourceTool`, and platform- or
feature-gated tools such as `PowerShell` and plan-mode controls need dedicated
fixtures before they can be prompt-benchmarked. Add a row when the required
deferred tool, MCP resource, platform, or interactive approval path is fixed
and reproducible; merely asking the model to call an unadvertised tool is not a
valid test.

The following protocol cases cannot be established by prompt wording alone and
must be separate API-harness tests:

- `tool_choice` with automatic, disabled, required-any, and named-tool modes;
- strict versus non-strict input-schema enforcement;
- malformed, truncated, or non-object generated arguments;
- unknown tool names and unknown versioned Anthropic-schema tool types;
- mixed text and tool blocks, multiple/parallel result ordering, and duplicate
  or missing `tool_use_id` values;
- text, image, document, empty, and `is_error` tool-result content where each
  form is in scope.

### P0-05 Claude Code Client-Side MCP

Implementation:

- [x] Treat namespaced MCP tools as ordinary client tools.
- [x] Keep MCP connection and execution outside `trtllm-serve`.
- [x] Reject the unrelated server-side `mcp_toolset` architecture.
- [x] Document the Glean setup and ownership boundary.
- [ ] Align the TensorRT-LLM Agent Toolkit server name and tool allowlist.

Validation:

- [x] A real namespaced client-side MCP tool survives adapter and template conversion.
- [x] DeepSeek V4 selects the Slurm MCP tool and emits valid arguments.
- [x] Claude Code executes the Slurm MCP `tools/call`.
- [x] The MCP result returns as `tool_result` and final generation uses it.
- [ ] `/mcp` reports the NVIDIA MaaS Glean server and expected tools.
- [ ] Captured request contains complete schemas and no `mcp_toolset`.
- [ ] A namespaced Glean tool survives adapter and template conversion unchanged.
- [ ] DeepSeek V4 selects the tool and emits valid arguments.
- [ ] Claude Code executes MCP `tools/call`.
- [ ] Glean text results return as `tool_result`.
- [ ] Final response uses the Glean result.

### P0-06 Extended Thinking

Implementation:

- [x] Map enabled and disabled thinking controls.
- [x] Validate the currently accepted manual budget shape.
- [x] Map assistant thinking history into reasoning history.
- [x] Emit thinking content before text and tool-use content.
- [x] Explicitly reject redacted thinking history.

Validation:

- [x] Current thinking adapter unit batch passes.
- [ ] Thinking prompt snapshot tests pass.
- [ ] Real DeepSeek V4 enabled and disabled behavior passes.
- [ ] Budget enforcement is measured rather than inferred from field mapping.
- [ ] Thinking plus tool-use history passes.
- [ ] Adaptive and effort differences are measured and documented.

### P0-07 Stop, Usage, and Error Semantics

Implementation:

- [x] Map natural stop, token limit, stop sequence, and tool-use reasons.
- [x] Report basic input, output, and cache-read usage.
- [x] Convert known request failures into Anthropic errors.
- [x] Fail generated malformed tool JSON safely.

Validation:

- [ ] Every supported stop reason passes in JSON and streaming responses.
- [ ] Disaggregated usage accounting passes for all supported result types.
- [ ] Authentication, permission, rate limit, timeout, and overload errors are classified.
- [ ] Request IDs are returned consistently.
- [ ] Internal tracebacks never reach clients.

### Excluded: Standard/Disaggregated Parity

Both servers expose the Anthropic route and share request/response adapters,
but real standard/disaggregated parity is not part of the current acceptance
scope. The disaggregated deployment is the validation target. Re-open
`EXT-ROUTE-01` only if product scope changes.

## 5. Milestones

| Milestone | Outcome | Included trackers | State | Exit criterion |
| --- | --- | --- | --- | --- |
| M0 Basic Messages | Ordinary Claude Code chat works | P0-01, P0-02, P0-07 | Basic E2E validated | Disaggregated basic chat passes; sanitized fixtures remain. |
| M1 Streaming | Claude Code consumes stable SSE | P0-03, P0-07 | In progress | Fragmentation, errors, and real-client streaming pass. |
| M2 Client Tools | Full client tool loop works | P0-04, P0-07 | In progress | Tool loop passes and required authoritative client schemas are implemented. |
| M3 Glean MCP | Real client-side MCP loop works | P0-05 | Planned | Glean discovery through final answer passes. |
| M4 Thinking | Thinking behavior and limits are classified | P0-06, P1-02 | Planned | Real-model behavior is measured and documented. |
| M5 Broader Messages API | Selected P1 capabilities expand | P1 trackers | Backlog | Each selected feature has contract and exit criteria. |

Progress is reported by achieved validation gates, not by percentages. A small
server-executor feature can be more expensive than several field mappings.

## 6. Gap Register

| Gap ID | Parent tracker | Owner layer | Priority | Required work | Exit criterion |
| --- | --- | --- | --- | --- | --- |
| GAP-01 | P0-01 through P0-07 | Validation environment | P0 | Preserve the passing 76 adapter/client and 27 route/client tests, then add remaining fault cases | All selected P0 cases pass in the development image. |
| GAP-02 | P0-02, P0-04 | Client traffic | P0 | Capture sanitized real Claude Code requests | Fixtures cover basic chat and a full tool loop. |
| GAP-03 | P0-05 | Client plus end-to-end | P0 | Run NVIDIA MaaS Glean through Claude Code | Discovery, execution, result, and final response pass. |
| GAP-04 | P0-04 | Adapter | P0 | Add required versioned schema-client-tool definitions | Supported versions expand losslessly; unknown versions fail closed. |
| GAP-05 | P0-06, P1-02 | Model/template | P0/P1 | Evaluate thinking, budget, and effort | Each behavior is supported, best effort, or rejected with evidence. |
| GAP-07 | P1-03 | Route/tokenizer | P1 | Implement Anthropic token counting | Count uses the same template, tools, and system content as generation. |
| GAP-08 | CAT-MOD-09 | Model/adapter | P1 | Define document and PDF behavior | Supported input forms and target-model limitations are explicit. |
| GAP-09 | CAT-MOD-01 | Client/model/server | P0 | Reconcile Claude Code's advertised 1M model label with the server's actual 128K limit | Client behavior at and beyond 128K is explicit and tested. |
| GAP-10 | P0-03, P0-04, P0-05 | DeepSeek-V4 streaming tool parser | P0 | **Closed 2026-07-19:** flush ordinary final text retained with `<｜end▁of▁sentence｜>` and fail closed on incomplete DSML | Parser smoke cases plus OpenAI, Anthropic, and Claude Code CT-PROMPT-01 preserve the complete tail and remove EOS; see the linked GAP-10 E2E run. |
| GAP-11 | P0-03, P0-07 | Serving lifecycle/logging | P1 | Stop treating normal closure of the internal OpenAI stream at `[DONE]` as an error-level `GeneratorExit` traceback | Normal Anthropic streaming completion produces no error traceback while preserving all final events. |

Use the stable Gap ID in commits, issues, and review notes. Closing a gap
requires updating the parent tracker, its checklist, and its evidence link.

## 7. Evidence and Fixture Policy

Every validation-stage promotion must link to executable evidence:

| Promotion | Required evidence |
| --- | --- |
| `contract_defined` | Official contract link plus representative wire example. |
| `mapping_implemented` | Code link plus explicit unsupported/error behavior. |
| `unit_validated` | Passing test command and environment. |
| `route_validated` | Passing HTTP or SSE route fixture. |
| `real_model_validated` | Checkpoint, serve command, prompt, and result. |
| `claude_code_e2e` | Sanitized Claude Code request/response and tool trace. |
| `done` | All tracker exit criteria met and documented. |

Fixture naming and sanitization rules are defined in
[fixtures/README.md](fixtures/README.md).

## 8. Update Rules

- Update the Catalog when the official overview adds, removes, renames, or
  changes availability for a public feature.
- Add work to the Tracker only after selecting a scope and disposition.
- Keep protocol mechanics and edge cases in the parent acceptance checklist.
- Never mark a capability supported solely because serialization succeeds.
- Keep adapter, model/template, serving configuration, client, and executor gaps
  distinct.
- Record explicit failure behavior for every rejected capability.
- Do not change a delivery stage without linking the required evidence.
- Review the matrix after model, tokenizer, parser, or Claude Code upgrades.
