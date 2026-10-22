# New widgets and outputs are appended, never inserted or reordered

ComfyUI saves a node's widget values by position (`widgets_values`) and its links by output slot. So the order of `required` inputs and of `RETURN_TYPES` is a contract with every saved workflow and every image that embeds one. New widgets go at the end of `required` (for example `retries`, `service_tier`, `video_job_id`, `on_error`), and new outputs go after the existing ones (for example `cost`). That is why some inputs sit in an order that looks arbitrary. Do not "tidy" it.

## Considered Options

- Insert in logical order: rejected; it silently shifts values in saved workflows.
- Put `service_tier` in `optional`: considered, because API-format prompts saved before a new required input fail validation. It was kept required to match how `reasoning_effort` was added.
