# Break from upstream; develop the fork as its own node pack

This fork (`0cyris/comfyui-openrouter_node`) no longer merges from gabe-init/ComfyUI-Openrouter_node. Upstream moved to one `OpenRouterNode` with a `request_type` switch, plus `openrouter_workflow.js`, which rewrites the settings of every saved `OpenRouterNode`. Merging that would scramble saved workflows and undo this fork's split into dedicated nodes per modality (chat, image, Image API, speech, transcription, rerank, video, decision). Upstream also refuses to retry paid calls and drops OpenRouter's error text, which this fork does on purpose (see ADR-0005). Upstream stays as the read-only `upstream` remote. Ideas from it are re-implemented by hand through issues, not merged.

## Consequences

- `gh` can resolve the fork to its upstream, so every `gh issue`/`gh pr`/`gh api` call passes `-R 0cyris/comfyui-openrouter_node`.
- Re-joining upstream later means reconciling two incompatible widget layouts in saved workflows.
