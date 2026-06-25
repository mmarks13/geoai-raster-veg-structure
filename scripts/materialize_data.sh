#!/usr/bin/env bash
# Materialize symlinks under data/processed/ and data/output/ into real files
# in place. Idempotent: regular files are skipped, broken symlinks are reported.
# Run before tarring/rsyncing the repo to a new host.
set -euo pipefail

cd "$(dirname "$0")/.."

# Roots to walk. Override by passing one or more paths as positional args:
#   bash scripts/materialize_data.sh data/processed/forest_plot_data
ROOTS=("${@:-data/processed data/output}")
# When no args were given, ROOTS is a single string "data/processed data/output";
# split on whitespace into a real array.
if [[ ${#ROOTS[@]} -eq 1 && "${ROOTS[0]}" == *" "* ]]; then
    read -ra ROOTS <<< "${ROOTS[0]}"
fi

# Filter to roots that actually exist (so a fresh checkout doesn't error).
existing_roots=()
for r in "${ROOTS[@]}"; do
    if [[ -d "$r" ]]; then existing_roots+=("$r"); fi
done

if [[ ${#existing_roots[@]} -eq 0 ]]; then
    echo "No matching roots exist under $(pwd). Nothing to do."
    exit 0
fi

shopt -s globstar nullglob
mapfile -t links < <(find "${existing_roots[@]}" -type l)

if [[ ${#links[@]} -eq 0 ]]; then
    echo "No symlinks under: ${existing_roots[*]}. Nothing to do."
    exit 0
fi

echo "Materializing ${#links[@]} symlink(s) under: ${existing_roots[*]}"

for link in "${links[@]}"; do
    target="$(readlink -f "$link" || true)"
    if [[ -z "$target" || ! -e "$target" ]]; then
        echo "SKIP (broken):       $link -> ${target:-<unreadable>}"
        continue
    fi
    tmp="${link}.materializing.$$"
    rsync -a --copy-links "$target" "$tmp"
    rm "$link"
    mv "$tmp" "$link"
    echo "MATERIALIZED:        $link  (from $target)"
done
