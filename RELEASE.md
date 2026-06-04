# Release Workflow

This repo uses `pyproject.toml` and `src/codex_telegram/__init__.py` as the
package version sources. Keep them in sync. Tags use the same semantic version
with a `v` prefix, for example `v0.2.0`.

## Gate

Before merging a feature branch to `main`:

1. Confirm the worktree is clean except for the intended release changes.
2. Confirm the relevant Linear child issues are Done and have evidence.
3. Bump both version files to the next version.
4. Run the full unit suite:

   ```bash
   PYTHONPATH=src python3 -m unittest discover -s tests
   ```

5. Run `git diff --check`.
6. Run the nested Codex review-agent loop until it returns clean.
7. Run the Telegram manual E2E for the release candidate. Prefer the committed
   preflight harness first:

   ```bash
   release_branch="$(git branch --show-current)"
   candidate_sha="$(git rev-parse --short HEAD)"
   /tmp/codex-telegram-e2e-venv/bin/python scripts/telegram-e2e-preflight.py \
     --env-file $HOME/.local/share/codex-telegram/.env \
     --credentials telegram-cred.md \
     --session .codex-telegram/e2e/admin-account \
     --group "Codex Telegram E2E" \
     --expected-service-workdir $HOME/.local/share/codex-telegram \
     --expected-service-branch "$release_branch" \
     --expected-service-commit "$candidate_sha" \
     --marker
   ```

   If the release candidate is running as a foreground bridge from this checkout
   instead of the installed service, pass `--skip-service-check` and record that
   reason in Linear.

8. For v0.2 and later forum-session releases, also run a real General-chat
   create-session E2E against the release candidate, then delete the temporary
   topic and verify local topic-session state is clean.
9. Confirm local-only files are not staged: `.env`, `telegram-cred.md`,
   `.codex-telegram/`, Telethon sessions, caches, build output, and service
   runtime state.

## Merge And Tag

After the gate passes:

```bash
release_branch="$(git branch --show-current)"
release_version="$(PYTHONPATH=src python3 -c 'import codex_telegram; print(codex_telegram.__version__)')"
release_tag="v${release_version}"
git checkout main
git merge --ff-only "$release_branch"
git tag -a "$release_tag" -m "Release $release_tag"
git checkout "$release_branch"
```

If a fast-forward merge is not possible, inspect why before merging. Do not
force a merge with a dirty worktree or after a failed review/E2E gate.

## Installed Service Update

After tagging, update the installed checkout deliberately:

```bash
cd $HOME/.local/share/codex-telegram
git fetch --all --tags
git checkout main
git pull --ff-only
bash scripts/install.sh
systemctl --user status codex-telegram.service --no-pager
```

If the installed checkout intentionally remains on a feature branch for testing,
record that in Linear and pass matching `--expected-service-*` arguments to the
preflight harness.

## Rollback

To roll back the installed service, choose the previous known-good tag or commit,
then reinstall/restart from that checkout:

```bash
cd $HOME/.local/share/codex-telegram
git checkout <previous-tag-or-commit>
bash scripts/install.sh
systemctl --user restart codex-telegram.service
systemctl --user status codex-telegram.service --no-pager
```

Record the rollback commit/tag and service status in Linear.
