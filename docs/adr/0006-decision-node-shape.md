# Decision node: one question, plain typed outputs, value scaled to 0-1

`OpenRouterDecisionNode` sends one noul, choice or score question to the alpha endpoint `POST /api/alpha/decisions`. It returns plain typed values (`answer`, `value`, `passed`, `details`, `cost`, `best_index`, `best_image`) that a workflow can wire straight into switches and math nodes. `value` is always 0-1: the probability for noul, the confidence for choice, and score / (levels - 1) for score. So `threshold` means the same for every type. Images are resized to fit the strictest listed model limits (256-1568 px per side), so no model rejects them by size.

## Considered Options

- Several questions per node: deferred; the outputs could not stay plain typed values. Open design in issue #15.
- Fold it into the chat node: rejected; the endpoint and the probability output are different.

## Consequences

For `choice`, `passed` means the model is confident in whatever option it picked, not that it picked the expected option.
