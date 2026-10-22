# PRs land on `staging`, which is merged into `main` later

PRs target `staging`, a permanent integration branch, not `main`. This makes it easy to test a batch of changes together in ComfyUI before committing to them on `main`. `main` stays the known-good line.

## Consequences

- GitHub fires `Closes #N` only on merges into the default branch (`main`), so after a PR lands on `staging`, close its issue by hand (see `docs/agents/issue-tracker.md`).
- `main` can lag behind `staging` by several decisions. An ADR whose code exists only on `staging` says so.
