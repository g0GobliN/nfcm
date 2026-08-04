#!/usr/bin/env bash
# Create / update GitHub labels from .github/labels.yml
set -euo pipefail
cd "$(dirname "$0")/../.."

if ! command -v gh >/dev/null; then
  echo "gh CLI required" >&2
  exit 1
fi

python3 <<'PY'
from pathlib import Path

text = Path(".github/labels.yml").read_text()
items = []
cur = None
for line in text.splitlines():
    if line.startswith("- name:"):
        if cur:
            items.append(cur)
        cur = {"name": line.split(":", 1)[1].strip()}
    elif cur is not None and line.strip().startswith("color:"):
        cur["color"] = line.split(":", 1)[1].strip()
    elif cur is not None and line.strip().startswith("description:"):
        cur["description"] = line.split(":", 1)[1].strip()
if cur:
    items.append(cur)

out = Path("/tmp/nfcm-labels.tsv")
with out.open("w") as f:
    for item in items:
        f.write(f"{item['name']}\t{item['color']}\t{item['description']}\n")
print(f"wrote {len(items)} labels to {out}")
PY

while IFS=$'\t' read -r name color desc; do
  echo "label: $name"
  gh label create "$name" --color "$color" --description "$desc" --force
done < /tmp/nfcm-labels.tsv

echo "Done."
