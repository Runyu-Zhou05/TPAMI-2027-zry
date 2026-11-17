# DeepSeek MathVista judge smoke

Date: 2026-11-17

The official DeepSeek chat endpoint was tested without printing or storing the
credential in the repository. The API health request returned HTTP 200.

A checksum-pinned `MathVista_MINI.tsv` was then downloaded and verified:

```text
MD5: f199b98e178e5a2a20e7048f5dcb0464
Rows: 1000
```

The first real MathVista item was passed through the pinned VLMEvalKit
few-shot answer-extraction prompt. A synthetic model response retained a
`<think>...</think>` trace and ended with `<answer>1.2</answer>`.
`deepseek-chat` returned exactly `1.2`, which passed MathVista's official typed
float comparison against the expected answer.

```text
HTTP status: 200
MathVista index: 1
Question type: free_form
Answer type: float
Expected: 1.2
Extracted: 1.2
Typed check: PASS
Prompt tokens: 744
Completion tokens: 3
```

The pinned VLMEvalKit patch routes the explicit name `deepseek-chat` through
its OpenAI-compatible wrapper at
`https://api.deepseek.com/chat/completions`. The pre-existing name `deepseek`
continues to mean the SiliconFlow-hosted model and is not used by this workflow.
The wrapper's API-key log line is patched to print `***REDACTED***`.
