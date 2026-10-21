<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# S1: adapt trtllm-serve to a model's chat format

**Inputs:**
- the harness contracts of the cells in play;
- the checkpoint, with its own encoder or chat template.

**Outputs:**
- rendering, parsers and stop rules that pass the fault matrix;
- regression tests that failed on the old code;
- `profiles/model/<id>.yaml`: format card, markers, degenerate ids, parsers, renderer source.

One adaptation serves every harness. Both output paths (Responses, and Chat with Messages on top)
use the same parsers.

## 0. Invariants (each one was a production bug)

1. **One generation, one story.** The streaming path (`parse_streaming_increment` per chunk, then
   the end-of-stream drain) and the whole-text path (`detect_and_parse`: non-streaming replies,
   and any snapshot rebuild) yield the same calls and the same text.
2. **Chunking never changes the result.** Real streams split text anywhere between tokens.
3. **Nothing generated is silently dropped.** Unterminated markup reaches the client as text;
   whitespace is text.
4. **Names resolve the same way on both paths:** bare and qualified spellings of a declared tool,
   and never a name recovered from prose.
5. **A single-call request yields at most one call.** That covers `parallel_tool_calls=false`
   and `disable_parallel_tool_use`. The parser exposes a `single_call_stop`.
6. **Degenerate ids stop the request,** which then fails without output.

## 1. Inventory what exists (do not rebuild it)

```bash
grep -n "<model_type>" tensorrt_llm/serve/tool_parser/tool_parser_factory.py \
    tensorrt_llm/llmapi/reasoning_parser.py tensorrt_llm/tokenizer/tokenizer.py
ls tensorrt_llm/serve/tool_parser/ tensorrt_llm/tokenizer/ tensorrt_llm/serve/extensions/
git log --oneline origin/main -- tensorrt_llm/serve/tool_parser/<model>*
```

State on main, 2026-10-21:

| Model | Renderer | Tool / reasoning parser | Note |
|---|---|---|---|
| DeepSeek V4.1 | `--custom_tokenizer deepseek_v41` (never auto-selected) | `deepseek_v41` / `deepseek_v41` | Main only (#19727). Thinking mode needs per-request `chat_template_kwargs` |
| Kimi-K3 | Checkpoint encoder, plus `serve/extensions/kimi_k3.py` (thinking, effort, tool_choice into template kwargs) | `kimi_k3` / `kimi_k3` | The K3 serving branch rewrote both parsers and adds the `[PAD]` stop |
| GLM-5.2/5.3 (`glm_moe_dsa`) | HF Jinja template | `glm47` (auto) / `glm47`, `glm_moe_dsa` | Main's parsers were rewritten in #19935; later campaign fixes are on the branch |
| GLM-5.3-Flash (`glm5_next`) | HF Jinja template | no automatic tool-parser mapping | Different template hash from GLM-5.2 |

## 2. Take the format from the model, not from memory

- **Source of truth.** Use the checkpoint's own encoder (DeepSeek V4.1 `encoding/encoding.py`
  with `test_encoding.py`; K3 `encoding_k3.py`) or its `chat_template.jinja` (GLM).
- **Golden conversations.** Render them with the reference AND with trtllm-serve's renderer, and
  require byte equality. At minimum:
  - developer and system messages, including a mid-conversation system message;
  - a tool call with its result;
  - two calls with results given in a different order;
  - a tool result as content parts, and an image;
  - reasoning from a previous turn: does the model keep, drop or re-render it? Record the answer
    as `reasoning.history_policy`;
  - a multi-turn history in each harness's real shape.
- **The format card.** Write it into the model profile (`format_card`). Today's cards:

  | Field | DeepSeek V4.1 | GLM-5.x | Kimi-K3 |
  |---|---|---|---|
  | `block_open` / `block_close` | `<｜DSML｜ calls>` / `</｜DSML｜ calls>` (leading-space tag names; V4 had none) | none | `<\|open\|>tools<\|sep\|>` / `<\|close\|>tools<\|sep\|>` |
  | `call_open` | `<｜DSML｜ invoke name="` | `<tool_call>` | `<\|open\|>call` plus attributes |
  | `name_end` | `">` | `<arg_key>` (no newline) | n/a: the name is an attribute |
  | `call_close` | `</｜DSML｜ invoke>` | `</tool_call>` | `<\|close\|>call<\|sep\|>` |
  | `value_open` / `value_close` | `string="true">` / `</｜DSML｜ parameter>` | `<arg_value>` / `</arg_value>` | ... / `<\|close\|>argument<\|sep\|>` |

- **Special and degenerate ids.** List the stop tokens, the think open/close tokens, and the
  degenerate ids (pad, last vocab id). The validator's leak and degeneration screens read them
  from the profile.

## 3. Parser checklist (`tensorrt_llm/serve/tool_parser/base_tool_parser.py`)

- **`bot_token` / `eot_token`, `has_tool_call`, `needs_raw_special_tokens`.** Set the last one to
  True when the markup consists of special tokens; otherwise detokenization strips them.
- **`detect_and_parse` and `parse_streaming_increment` share one grammar.** Never a regex on one
  side and a state machine on the other without the fault matrix proving them equal.
- **`finish()`** emits, or releases as text, whatever the stream left buffered. A no-op `finish`
  is how a dangling close tag reaches the client.
- **`single_call_stop`:** the text that completes exactly one call.
- **`markup_tokens`:** delivered arguments containing them are flagged in traces.
- **`structure_info` / structural tags:** for strict or forced tool choice.
- **`resolve_tool_name`** on both paths.
- **No text-rewriting idioms:** no `.strip()`, no `replace(token, "")` on generated text.

Some of these hooks (`single_call_stop`, `markup_tokens`, the stream-assembly helpers) exist only
on the campaign branches. Port them with the fixes.

## 4. Run the fault matrix

```bash
python scripts/fault_mutations.py --card card.json --samples canonical.jsonl --out mutated.jsonl
python scripts/chunk_equivalence.py --parser <name> --samples mutated.jsonl --tools tools.json \
    --marker '<block open>' --marker '<call close>' --ignorable '<eos token>' --out report.json
```

- **The card** is the profile's `format_card` written out as JSON.
- **Canonical samples** are rendered with the model's own encoder (section 2), never by hand.
- **What `chunk_equivalence.py` compares per sample:**
  - the whole-text parse;
  - streams at every split point;
  - per-character streams;
  - random chunkings.
- **Split at token boundaries for special-token formats.** K3's `<|...|>` control tokens never
  arrive split per character. Character-level splits produced 23 false findings in 28 samples.
- **Target:**
  - zero `calls_differ`, `streams_disagree` and `text_lost`;
  - `normal_text_differs` only where documented;
  - whitespace-only differences understood.
- **Run it on the whole parser family.** Siblings inherit defects: glm4/glm47, and DeepSeek
  V3.2 → V4 → V4.1.
- **Without the C++ build:** run inside the TRT-LLM container. The campaign's stub loader does
  not work on main (it crashes in `responses_utils.py`).

## 5. Fault catalog: what production models emitted, and what serving must do

All of these were seen under Codex in the campaign (GLM-5.3, Kimi-K3, about 24M requests).
Commits are on the campaign branches.

| Shape | Mutation | Evidence | Serving must | Fix |
|---|---|---|---|---|
| Stream ends inside a call | `truncate@N` | 1 in 3,437 responses; the stream showed nothing, the snapshot 429 chars of markup | Release unterminated markup as text on BOTH views; never deliver a half call | `_flush_tool_parser`, `_assembled_tool_calls` |
| A call's close tag is missing | `drop_call_close` | K3; under `parallel_tool_calls=false` two calls were delivered (B009-01, 1 in 187,895) | Recover identically on both paths, or treat as text on both | `63bc387690`; B009-01 policy open |
| The wrapper is never closed | `drop_block_close` | DeepSeek V3.2/V4 whole-text parse returns no call; streaming emits them | Same calls on both paths | Open for DeepSeek |
| Empty name | `empty_name` | B008-05: nameless call; `output_item.done` had 1,395 chars, `completed` 1,371 | Never a nameless call; identical text on both views | `006272aa23` |
| Bare vs qualified name | `qualified_name`, `bare_name` | 247 bare `exec` against 34 qualified in one session; 32.1% aborted | Resolve unambiguous spellings on BOTH paths; custom tool as `custom_tool_call` | `b52237f18c` |
| Mangled names | (manual) | About 1 in 800 GLM calls | Strip balanced markup from names; never recover a name from prose | `e1b236f72e`, `364065809b` |
| Extra calls where one is allowed | `duplicate_call` | K3-01: 6,628 replies | Stop at the end of the first call (`single_call_stop`) | `a3e176fef0`, `221949e395` |
| A new opener inside an unclosed call | `nested_open` | GLM: the model abandons a call and starts over | Do not deliver the abandoned half call; both views agree | `af44f3a18d` |
| Close markup inside a value | `markup_in_value` | GLM's own; also GLM markup reaching K3 through cross-model history | Deliver the value byte-exact or reject the call, the same on both views; flag it | `cae822bab1`, `1317e90bf4` |
| Prose before or after the calls | `leading_text`, `trailing_text` | DeepSeek whole-text parse drops text after the block | Deliver all prose, identically | Open for DeepSeek |
| Whitespace | `blank_lines` | `.strip()` deleted bytes (622 deleted close tags in an audit) | Whitespace is generated text | `be7746ef53` |
| Non-ASCII, quotes, backslashes | `unicode_value` | Escaping errors | Arguments round-trip exactly; types follow the schema | `8290b93f8d`, `d5c7446321` |
| Degenerate runaway (`[PAD]` repeated) | (separate) | 66 K3 loops held about 76 h of slots (root cause: a missing fence in the Helix all-to-all) | Stop and fail without output | `a13ecfbb50`; root cause `c47b06d6e7` |
| `</think>` twice; leading `</think>` residue | (separate) | GLM; 32–35% of turns before the fix | A dedicated reasoning parser; drop the residue, not the content | `6c9834e713`, `951f41ed54` |

**Reading the results:**
- `calls_differ`, `streams_disagree` and `text_lost` must be zero.
- `normal_text_differs` on `trailing_text` or `markup_in_value` means generated text was
  dropped or rewritten.
- Every finding is parser-level until L2 confirms it with real token boundaries.

## 6. Tests and registration

1. **Turn every failing sample into a test.**
   - Inherit `BaseToolParserTestClass` (`tests/unittest/llmapi/apps/test_tool_parsers.py`), with
     the catalog cases parametrized.
   - Each new test must FAIL on the old parser and PASS on the new one. Keep the node id and
     both results.
2. **Register the parsers.**
   - Map them in `tool_parser_factory.py` (and `MODEL_TYPE_TO_TOOL_PARSER`) and in
     `reasoning_parser.py`.
   - Document the flags: `--tool_parser`, `--reasoning_parser`, `--custom_tokenizer`.
3. **Map the harness's reasoning controls into the template where needed.** For example, Codex
   sends only `reasoning.effort`, Claude Code sends `thinking` and `output_config.effort`, and pi
   sends whatever its `thinkingFormat` dictates. Compare `serve/extensions/kimi_k3.py`.
4. **Move on to S2** (`replay.md`).

## 7. Traps

- **Recovering a tool name from prose fabricates calls.** Only bare dotted identifiers, and only
  unambiguous tails.
- **Reasoning history from another model** (cross-pool routing) puts foreign markup into the
  prompt, and the model copies it into its own calls. Keep a conversation on one model.
- **A model writing its own markup inside code is not a leak.** Quoted hits are usually the model
  explaining syntax. Report screens; do not change serving to hide model behavior.
