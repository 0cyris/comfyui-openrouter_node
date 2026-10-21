---
status: accepted
---

# Model lists load in the background; `model` is not validated against them

`INPUT_TYPES` never waits on the network. One background worker refreshes all model lists, and it also starts at import. Until it finishes, the nodes show fallback lists. Fetching synchronously blocked building the node list for up to 120 s per list. So that saved workflows can still queue with a model the list doesn't have yet, each node defines `VALIDATE_INPUTS(cls, model): return True`, which turns off ComfyUI's combo check for `model`.

## Consequences

A typo in a model id reaches OpenRouter and fails there, not in ComfyUI validation. Right after a restart, the lists can show fallbacks. Code that reads per-model parameters must allow for them being empty early.

On `staging` (issue #2). `main` still fetches synchronously until `staging` is merged.
