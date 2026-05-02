# `.github/` Workflow Patches

GitHub Apps cannot push changes under `.github/workflows/`. Patches that need
to land there are staged here so a human can apply and push them.

## How to apply

```bash
git checkout turboquant/runtime
git pull
git apply .github-patches/claude-code-review-allow-bot.patch
git add .github/workflows/claude-code-review.yml
git commit -m "ci(claude-review): allow claude[bot] as triggering actor"
git push
```

Then delete the patch file:

```bash
git rm .github-patches/claude-code-review-allow-bot.patch
git commit -m "chore: drop applied claude-review patch"
git push
```

## Open patches

| Patch | Target branch | Why |
|-------|---------------|-----|
| `claude-code-review-allow-bot.patch` | `turboquant/runtime` | Add `allowed_bots: 'claude[bot]'` so PR re-runs caused by Claude's own commits don't fail. |
