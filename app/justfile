UV_GROUPS := "--group mamba-macos"
MACOS_DYLD_CLEAN := "DYLD_LIBRARY_PATH='' DYLD_FALLBACK_LIBRARY_PATH=''"
MACOS_OBJC_WARNING_FILTER := "2> >(uv run python scripts/filter_macos_objc_warnings.py >&2)"

sync:
    uv sync {{UV_GROUPS}}

sync-base:
    uv sync

sync-mamba-macos:
    uv sync --group mamba-macos

sync-mamba-linux:
    uv sync --group mamba-linux

lint:
    uv run ruff check .

format-python:
    uv run ruff format .

format-web:
    npm exec -- prettier --write --ignore-unknown --no-error-on-unmatched-pattern "app.css" "*.json" "*.yaml" "*.yml" "*.md"

format-toml:
    npm exec -- taplo format "pyproject.toml" "config.toml"

format: format-python format-web format-toml

typecheck:
    uv run mypy

deptry:
    uv run deptry .

check-python:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy
    uv run deptry .

check-web:
    npm exec -- prettier --check --ignore-unknown --no-error-on-unmatched-pattern "app.css" "*.json" "*.yaml" "*.yml" "*.md"

check-toml:
    npm exec -- taplo format --check "pyproject.toml" "config.toml"

check: sync check-python check-web check-toml

fix: sync
    uv run ruff check . --fix
    uv run ruff format .
    npm exec -- prettier --write --ignore-unknown --no-error-on-unmatched-pattern "app.css" "*.json" "*.yaml" "*.yml" "*.md"
    npm exec -- taplo format "pyproject.toml" "config.toml"

check-mamba-import:
    uv run python -c "import importlib; module = importlib.import_module('mamba_ssm'); mamba_cls = getattr(module, 'Mamba'); print('mamba_ssm ok:', mamba_cls)"

check-real-model:
    uv run python -c "from pulse.inference.readiness import main; main()"

check-real-model-macos:
    uv sync --group mamba-macos
    uv run python -c "from pulse.inference.readiness import main; main()"

check-real-model-linux:
    uv sync --group mamba-linux
    uv run python -c "from pulse.inference.readiness import main; main()"

run:
    bash -lc "{{MACOS_DYLD_CLEAN}} uv run python app.py {{MACOS_OBJC_WARNING_FILTER}}"

dev:
    bash -lc "{{MACOS_DYLD_CLEAN}} uv run gradio app.py {{MACOS_OBJC_WARNING_FILTER}}"

run-base:
    uv sync
    bash -lc "{{MACOS_DYLD_CLEAN}} uv run python app.py {{MACOS_OBJC_WARNING_FILTER}}"

dev-base:
    uv sync
    bash -lc "{{MACOS_DYLD_CLEAN}} uv run gradio app.py {{MACOS_OBJC_WARNING_FILTER}}"

run-mamba-macos:
    uv sync --group mamba-macos
    bash -lc "{{MACOS_DYLD_CLEAN}} uv run python app.py {{MACOS_OBJC_WARNING_FILTER}}"

dev-mamba-macos:
    uv sync --group mamba-macos
    bash -lc "{{MACOS_DYLD_CLEAN}} uv run gradio app.py {{MACOS_OBJC_WARNING_FILTER}}"

run-mamba-linux:
    uv sync --group mamba-linux
    uv run python app.py

dev-mamba-linux:
    uv sync --group mamba-linux
    uv run gradio app.py