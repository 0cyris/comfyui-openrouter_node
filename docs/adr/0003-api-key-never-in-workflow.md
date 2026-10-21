---
status: accepted
---

# The API key never travels through a workflow or a queued prompt

The key comes only from the `LLM_KEY` environment variable, then `openrouter_api_key.json` (`get_api_key()` takes no arguments). A typed key would otherwise end up in saved workflows and in the prompt that is embedded in every output image. The `api_key` widget stays in each node's schema as a hidden, ignored placeholder (`widget.hidden = true`, `serializeValue = () => ""`, cleared on load). Removing it would shift every saved `widgets_values` (ADR-0002). A server prompt handler (`blank_prompt_api_keys`) also empties the input for API clients and old workflows.

## Considered Options

- Blank the field on save, as upstream does: rejected; the key still reaches the queued prompt and the image metadata.
- Delete the widget: rejected; it breaks saved workflows, including the character-sheet workflows.
- `options.serialize = false`: rejected; it drops a required input and fails server validation.

On `staging` (issue #4). `main` still reads the widget first until `staging` is merged.
