# A failed Decision check blocks only `best_image`

With `stop_on_false`, a best candidate that does not pass now blocks the `best_image` output (`on_false` = `block`, the default), in the same way as `on_error` on the image nodes (ADR-0005). Only what uses `best_image`, such as a save or the assembled sheet, is skipped. The rest of the run finishes, and `answer`, `value`, `passed`, `details` and `cost` still report the result, so previews and checks downstream still show it. Before this, a failed check raised and stopped the whole run, so the other panels were not saved and nothing showed the failed image. `on_false` = `raise` keeps that behaviour.

## Consequences

- This changed existing workflows: a check with `stop_on_false` that used to stop the run now blocks only `best_image`.
- A blocked result is cached like any other. Queueing the same image again does not pay for the check again; to get a new result, change the image.
- `on_false` was appended after `stop_on_false` (ADR-0002).
