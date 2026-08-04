# Maintainer notes

## Required status checks (branch protection)

In GitHub → **Settings → Branches → Branch protection rules → `main`**:

1. Enable **Require a pull request before merging**
2. Enable **Require status checks to pass before merging**
3. Require these checks (names must match workflow job names):
   - `CI passed` (from workflow **CI**)
   - `Scan commit trailers` (from workflow **No AI co-author**)
   - `PR title` (from workflow **PR checks**)
   - `Diff sanity` (from workflow **PR checks**)
4. Enable **Do not allow bypassing the above settings** (recommended)
5. Optionally: **Require conversation resolution before merging**

This enforces tests + attribution rules before anything lands on `main`.

## Labels

Create labels from [labels.yml](labels.yml):

```bash
# requires gh + yq, or create manually from the YAML
while read -r name color desc; do
  gh label create "$name" --color "$color" --description "$desc" --force
done < <(python3 - <<'PY'
import yaml, pathlib
for item in yaml.safe_load(pathlib.Path(".github/labels.yml").read_text()):
    print(item["name"], item["color"], item["description"])
PY
)
```

Or add them in the GitHub UI.

## Dependabot

[dependabot.yml](dependabot.yml) opens weekly PRs for Cargo, npm, and Actions.
Review like any other PR; CI must still pass; no AI co-author trailers.
