# uv.lock Update Guide

## What is uv.lock?

`uv.lock` is uv's lock file. It captures the exact version of every dependency, including transitive ones, much like:
- Node.js `package-lock.json`
- Rust `Cargo.lock`
- Python Poetry `poetry.lock`

Keeping `uv.lock` in version control guarantees that everyone installs the same dependency set.

## When does uv.lock change?

### Situations where it does *not* change automatically

- Running `uv sync --frozen`
- Building Docker images that call `uv sync --frozen`
- Editing source code without touching dependency metadata

### Situations where it will change

1. **`uv lock` or `uv lock --upgrade`**

   ```bash
   uv lock                # Resolve according to current constraints
   uv lock --upgrade      # Re-resolve and upgrade to the newest compatible releases
   ```

   Use these commands after modifying `pyproject.toml`, when you want fresh dependency versions, or if the lock file was deleted or corrupted.

2. **`uv add`**

   ```bash
    uv add requests           # Adds the dependency and updates both files
    uv add --dev pytest       # Adds a dev dependency
   ```

   `uv add` edits `pyproject.toml` and refreshes `uv.lock` in one step.

3. **`uv remove`**

   ```bash
   uv remove requests
   ```

   This removes the dependency from `pyproject.toml` and rewrites `uv.lock`.

4. **`uv sync` without `--frozen`**

   ```bash
   uv sync
   ```

   Normally this only installs what is already locked. However, if `pyproject.toml` and `uv.lock` disagree or the lock file is missing, uv will regenerate and update `uv.lock`. In CI and production builds you should prefer `uv sync --frozen` to prevent unintended updates.

## Example workflows

### Scenario 1: Add a new dependency

```bash
# Recommended: let uv handle both files
uv add fastapi
git add pyproject.toml uv.lock
git commit -m "Add fastapi dependency"

# Manual alternative
# 1. Edit pyproject.toml
# 2. Regenerate the lock file
uv lock
git add pyproject.toml uv.lock
git commit -m "Add fastapi dependency"
```

### Scenario 2: Relax or tighten a version constraint

```bash
# 1. Edit the requirement in pyproject.toml,
#    e.g. openai>=1.0.0,<2.0.0 -> openai>=1.5.0,<2.0.0

# 2. Re-resolve the lock file
uv lock

# 3. Commit both files
git add pyproject.toml uv.lock
git commit -m "Update openai to >=1.5.0"
```

### Scenario 3: Upgrade everything to the newest compatible versions

```bash
uv lock --upgrade
git diff uv.lock
git add uv.lock
git commit -m "Upgrade dependencies to latest compatible versions"
```

### Scenario 4: Teammate syncing the project

```bash
git pull               # Fetch latest code and lock file
uv sync --frozen       # Install exactly what uv.lock specifies
```

## Using uv.lock in Docker

```dockerfile
RUN uv sync --frozen --no-dev --extra api
```

`--frozen` guarantees reproducible builds because uv will refuse to deviate from the locked versions.
`--extra api` install API server

## Generating a lock file that includes offline dependencies

If you need `uv.lock` to capture the optional offline stacks, regenerate it with the relevant extras enabled:

```bash
uv lock --extra api --extra offline
```

This command resolves the base project requirements plus both the `api` and `offline` optional dependency sets, ensuring downstream `uv sync --frozen --extra api --extra offline` installs work without further resolution.

## Frequently asked questions

- **`uv.lock` is almost 1 MB. Does that matter?**
  No. The file is read only during dependency resolution.

- **Should we commit `uv.lock`?**
  Yes. Commit it so collaborators and CI jobs share the same dependency graph.

- **Deleted the lock file by accident?**
  Run `uv lock` to regenerate it from `pyproject.toml`.

- **Can `uv.lock` and `requirements.txt` coexist?**
  They can, but maintaining both is redundant. Prefer relying on `uv.lock` alone whenever possible.

- **How do I inspect locked versions?**
  ```bash
  uv tree
  grep -A5 'name = "openai"' uv.lock
  ```

## Best practices

### Recommended

1. Commit `uv.lock` alongside `pyproject.toml`.
2. Use `uv sync --frozen` in CI, Docker, and other reproducible environments.
3. Use plain `uv sync` during local development if you want uv to reconcile the lock for you.
4. Run `uv lock --upgrade` periodically to pick up the latest compatible releases.
5. Regenerate the lock file immediately after changing dependency constraints.

### Avoid

1. Running `uv sync` without `--frozen` in CI or production pipelines.
2. Editing `uv.lock` by hand—uv will overwrite manual edits.
3. Ignoring lock file diffs in code reviews—unexpected dependency changes can break builds.

## Summary

| Command               | Updates `uv.lock` | Typical use                               |
|-----------------------|-------------------|-------------------------------------------|
| `uv lock`             | ✅ Yes            | After editing constraints                 |
| `uv lock --upgrade`   | ✅ Yes            | Upgrade to the newest compatible versions |
| `uv add <pkg>`        | ✅ Yes            | Add a dependency                          |
| `uv remove <pkg>`     | ✅ Yes            | Remove a dependency                       |
| `uv sync`             | ⚠️ Maybe          | Local development; can regenerate the lock |
| `uv sync --frozen`    | ❌ No             | CI/CD, Docker, reproducible builds        |

Remember: `uv.lock` only changes when you run a command that tells it to. Keep it in sync with your project and commit it whenever it changes.
