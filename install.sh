#!/usr/bin/env bash
# Install diffusion-planner-dashboard into an ML-Planner checkout.
#
# Run at the ML-Planner repository root:
#   curl -fsSL https://raw.githubusercontent.com/yhisaki/diffusion_planner_dashboard/main/install.sh | bash
#
# The dashboard is cloned into packages/diffusion_planner_dashboard, hidden from git via
# .git/info/exclude, and installed in editable mode into ML-Planner's .venv. It is not a
# uv workspace member, so ML-Planner's pyproject.toml and uv.lock are not modified.
set -euo pipefail

REPO_URL="${DASHBOARD_REPO_URL:-https://github.com/yhisaki/diffusion_planner_dashboard.git}"
BRANCH="${DASHBOARD_BRANCH:-main}"
DEST="packages/diffusion_planner_dashboard"

die() {
  echo "error: $*" >&2
  exit 1
}

command -v git >/dev/null || die "git is not installed"
command -v uv >/dev/null || die "uv is not installed"

if [[ ! -f pyproject.toml ]] || ! grep -q '^name = "diffusion-planner-workspace"' pyproject.toml; then
  die "run this at the ML-Planner repository root"
fi
# A glob member such as packages/* would make the dashboard a workspace member and rewrite uv.lock.
grep -q '"packages/\*"' pyproject.toml &&
  die "ML-Planner's uv workspace uses packages/*; update ML-Planner to list members explicitly"

if [[ -d "$DEST/.git" ]]; then
  echo "Updating $DEST"
  git -C "$DEST" pull --ff-only
elif [[ -e "$DEST" ]]; then
  die "$DEST exists but is not a git repository"
else
  echo "Cloning $REPO_URL into $DEST"
  git clone --branch "$BRANCH" "$REPO_URL" "$DEST"
fi

exclude_file="$(git rev-parse --git-path info/exclude)"
mkdir -p "$(dirname "$exclude_file")"
if ! grep -qxE "/?$DEST/?" "$exclude_file" 2>/dev/null; then
  echo "/$DEST/" >>"$exclude_file"
  echo "Added /$DEST/ to $exclude_file"
fi

# --inexact keeps packages that are not in uv.lock, such as a previous dashboard install.
uv sync --inexact
# Pin shared dependencies to ML-Planner's uv.lock so a later 'uv run' does not reinstall them.
uv pip install --python .venv -e "$DEST" \
  --constraints <(uv export --frozen --no-hashes --no-emit-workspace --quiet)

cat <<MSG

diffusion-planner-dashboard is installed. Start it with:
  uv run diffusion-planner-dashboard
or, with .venv activated:
  diffusion-planner-dashboard

A plain 'uv sync' removes the dashboard from .venv. Use 'uv sync --inexact', or rerun this
script to reinstall it.
MSG
