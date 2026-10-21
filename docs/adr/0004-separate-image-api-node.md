# Two image nodes: keep the chat-completions node, add a separate Image API node

`OpenRouterImageGenNode` makes images through chat completions. That endpoint cannot give `n` images, real resolution and aspect fields, or several reference frames. The new `OpenRouterImageAPINode` uses `POST /api/v1/images` and filters settings per model through `/images/models`. The old node got only a minimal fix (`image_config` instead of the rejected `provider.data.size`) and otherwise stays as it was, because saved workflows use it.

## Consequences

The two nodes differ in endpoint, size fields and failure behaviour. The old node still returns a black 64x64 placeholder on error, while the new one follows ADR-0005. Removing either one breaks the workflows that use it.
