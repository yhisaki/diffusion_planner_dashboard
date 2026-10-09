# Diffusion Planner Dashboard

Streamlit dashboard for inspecting diffusion planner H5 shards, Parquet indexes, and
model inference. It is an optional add-on for
[ML-Planner](https://github.com/tier4/ML-Planner).

## Install

Run at the ML-Planner repository root:

```bash
curl -fsSL https://raw.githubusercontent.com/yhisaki/diffusion_planner_dashboard/main/install.sh | bash
```

The script clones this repository into `packages/diffusion_planner_dashboard` and
installs it in editable mode into ML-Planner's `.venv`.
ML-Planner's `pyproject.toml` and `uv.lock` are not modified. Rerun the script to update.

## Usage

```bash
uv run diffusion-planner-dashboard
```

or, with `.venv` activated, `diffusion-planner-dashboard`. Then select an H5 shard or
Parquet index from the sidebar.

A plain `uv sync` removes the dashboard from `.venv`. Use `uv sync --inexact`, or rerun
the install script.
