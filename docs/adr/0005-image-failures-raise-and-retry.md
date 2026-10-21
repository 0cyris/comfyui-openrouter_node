# Failed paid requests raise or block, and retry inside the node

A failure must never look like a result. New nodes do not return a placeholder image, which ComfyUI would cache and the user would pay for without knowing. Instead they put OpenRouter's message, provider and reasons in the error. Transient failures (408, 429, 5xx, timeouts and moderation blocks) are retried inside the node (`retries`, default 1, with backoff; `post_with_retries` in `openrouter_shared.py`). Each retry is billed. This was a deliberate choice of money over reliability, after one moderation failure stopped a run and threw away the five other views, which had already been billed. `on_error` defaults to `block`: one failure blocks only what depends on that node, the rest of the run finishes and stays cached, and `raise` stops the whole run.

## Considered Options

- Black placeholder (the old behaviour): rejected; it hides failures and gets cached.
- Never retry paid calls (upstream's policy): rejected for the reason above.

## Consequences

`IS_CHANGED` returns a per-node failure counter. A blocked result is cached like any other, and the counter makes the next queue run that node again.
