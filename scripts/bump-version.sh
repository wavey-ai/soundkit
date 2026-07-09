#!/usr/bin/env bash
# Bump every workspace crate to the same next version.
#
# Keeps all Bitneedle crates in lockstep: finds the current maximum
# `version` across every workspace member's Cargo.toml, bumps the patch
# segment by one (or an explicit --set version), then rewrites:
#
#   - each member's own `[package] version = "..."` field, and
#   - every path-dependency's companion `version = "..."` field that points
#     at another workspace crate, anywhere in the repo (including
#     non-member crates such as player-wasm's dev-dependency on record-cut).
#
# It never touches registry version *constraints* on crates that are not a
# local path dependency (e.g. player-wasm's `record-core = "0.1.3"` pulled
# from crates.io) — only `{ path = "...", version = "..." }` entries.
#
# Written for bash 3.2 (macOS's default /bin/bash) — no mapfile, no
# associative arrays.
#
# Usage:
#   scripts/bump-version.sh                # bump patch, apply, then `cargo check`
#   scripts/bump-version.sh --dry-run       # show the plan, change nothing
#   scripts/bump-version.sh --set 0.2.0     # use an explicit version
#   scripts/bump-version.sh --no-check      # skip the trailing `cargo check`

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DRY_RUN=0
NO_CHECK=0
EXPLICIT_VERSION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --no-check) NO_CHECK=1; shift ;;
    --set) EXPLICIT_VERSION="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Workspace member directory names, parsed from the root Cargo.toml's
# `members = [...]` list, de-duplicated, in file order.
MEMBERS_RAW="$(
  awk '
    /members[ \t]*=[ \t]*\[/ { in_members = 1 }
    in_members {
      while (match($0, /"[^"]+"/)) {
        print substr($0, RSTART + 1, RLENGTH - 2)
        $0 = substr($0, RSTART + RLENGTH)
      }
    }
    in_members && /\]/ { in_members = 0 }
  ' Cargo.toml | awk '!seen[$0]++'
)"

MEMBERS=()
while IFS= read -r line; do
  [[ -n "$line" ]] && MEMBERS+=("$line")
done <<< "$MEMBERS_RAW"

if [[ ${#MEMBERS[@]} -eq 0 ]]; then
  echo "error: could not parse [workspace] members from Cargo.toml" >&2
  exit 1
fi

package_version() {
  # First `^version = "X.Y.Z"` line in a crate's Cargo.toml is always the
  # [package] version — dependency version fields are never written at the
  # start of a line (they're nested inside `name = { ... }` entries).
  grep -m1 -E '^version[[:space:]]*=[[:space:]]*"[0-9]+\.[0-9]+\.[0-9]+"' "$1" \
    | sed -E 's/^version[[:space:]]*=[[:space:]]*"([0-9]+\.[0-9]+\.[0-9]+)".*/\1/'
}

# --- Determine the new unified version ---------------------------------

MAX_VERSION="0.0.0"
CURRENT_VERSIONS=()  # parallel array to MEMBERS
for member in "${MEMBERS[@]}"; do
  toml="$member/Cargo.toml"
  if [[ ! -f "$toml" ]]; then
    echo "error: $toml not found (workspace member missing?)" >&2
    exit 1
  fi
  v="$(package_version "$toml")"
  if [[ -z "$v" ]]; then
    echo "error: $toml has no [package] version field" >&2
    exit 1
  fi
  CURRENT_VERSIONS+=("$v")
  highest="$(printf '%s\n%s\n' "$MAX_VERSION" "$v" | sort -t. -k1,1n -k2,2n -k3,3n | tail -1)"
  MAX_VERSION="$highest"
done

if [[ -n "$EXPLICIT_VERSION" ]]; then
  NEW_VERSION="$EXPLICIT_VERSION"
else
  IFS=. read -r major minor patch <<< "$MAX_VERSION"
  NEW_VERSION="${major}.${minor}.$((patch + 1))"
fi

echo "Current versions (max ${MAX_VERSION}):"
i=0
for member in "${MEMBERS[@]}"; do
  current="${CURRENT_VERSIONS[$i]}"
  if [[ "$current" == "$NEW_VERSION" ]]; then
    printf '  %-24s %s\n' "$member" "$current"
  else
    printf '  %-24s %s -> %s\n' "$member" "$current" "$NEW_VERSION"
  fi
  i=$((i + 1))
done
echo
echo "New unified version: ${NEW_VERSION}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo
  echo "(dry run — no files changed)"
  exit 0
fi

# --- Apply -----------------------------------------------------------------

CHANGED_FILES=()

for member in "${MEMBERS[@]}"; do
  toml="$member/Cargo.toml"
  before="$(cat "$toml")"
  # Rewrite only the first `^version = "..."` line (the [package] field).
  after="$(awk -v new="$NEW_VERSION" '
    !done && /^version[ \t]*=[ \t]*"[0-9]+\.[0-9]+\.[0-9]+"/ {
      print "version = \"" new "\""
      done = 1
      next
    }
    { print }
  ' "$toml")"
  if [[ "$before" != "$after" ]]; then
    printf '%s\n' "$after" > "$toml"
    CHANGED_FILES+=("$toml")
  fi
done

# Every Cargo.toml in the repo (workspace members and anything else, such as
# player-wasm's dev-dependency block) may reference a workspace crate as a
# path dependency with a companion `version = "..."` field. Update just the
# version substring, keeping `path = "..."` intact.
while IFS= read -r -d '' toml; do
  before="$(cat "$toml")"
  after="$before"
  for member in "${MEMBERS[@]}"; do
    # Matches e.g.: record-core = { path = "../record-core", version = "0.1.5" }
    # Replacement keeps everything up to `version = ` and only swaps the value.
    after="$(printf '%s\n' "$after" | sed -E \
      "s#(${member}[[:space:]]*=[[:space:]]*\{[[:space:]]*path[[:space:]]*=[[:space:]]*\"[^\"]*\"[[:space:]]*,[[:space:]]*version[[:space:]]*=[[:space:]]*)\"[0-9]+\.[0-9]+\.[0-9]+\"#\1\"${NEW_VERSION}\"#g")"
  done
  if [[ "$before" != "$after" ]]; then
    printf '%s\n' "$after" > "$toml"
    already_listed=0
    for f in "${CHANGED_FILES[@]:-}"; do
      [[ "$f" == "$toml" ]] && already_listed=1
    done
    [[ "$already_listed" -eq 0 ]] && CHANGED_FILES+=("$toml")
  fi
done < <(find . -name Cargo.toml -not -path "./target/*" -print0)

echo
echo "Updated ${#CHANGED_FILES[@]} Cargo.toml file(s):"
for f in "${CHANGED_FILES[@]:-}"; do
  echo "  $f"
done

if [[ "$NO_CHECK" -eq 0 ]]; then
  echo
  echo "Running \`cargo check --workspace\` to verify and refresh Cargo.lock..."
  cargo check --workspace
fi

echo
echo "Done. Review \`git diff\`, then commit Cargo.toml/Cargo.lock changes before publishing."
