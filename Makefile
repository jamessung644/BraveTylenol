SHELL := /bin/sh
.DEFAULT_GOAL := help

PYTHON ?= python3.13
VENV ?= .venv
PROFILE ?= dev
ENV_FILE ?= .env
HOST ?= 0.0.0.0
PORT ?= 8000
WEB_CONCURRENCY ?= 1

VENV_PYTHON := $(VENV)/bin/python

ifneq (,$(wildcard $(ENV_FILE)))
include $(ENV_FILE)
CONFIG_VARS := LUNIT_FM_API_KEY LUNIT_FM_API_URL LUNIT_FM_MODEL LUNIT_MCP_URL
CONFIG_VARS += HARNESS_MODE HARNESS_LOG_LEVEL MAX_TOOL_CALLS L2_MAX_ATTEMPTS UPSTREAM_TIMEOUT_SECONDS
CONFIG_VARS += MAX_TOOL_RESULT_CHARS MAX_EVIDENCE_CHARS
ENV_EXPORTS := $(foreach variable,$(CONFIG_VARS),$(if $($(variable)),$(variable)))
ifneq (,$(strip $(ENV_EXPORTS)))
export $(ENV_EXPORTS)
endif
endif

.PHONY: help setup run shell test lint compile check

help:
	@printf '%s\n' \
		'BraveTylenol development commands:' \
		'  make setup    Create/update the Python 3.13 virtual environment' \
		'  make run      Start the API (HOST, PORT and WEB_CONCURRENCY are configurable)' \
		'  make shell    Open a shell with the virtual environment activated' \
		'  make test     Run the deterministic test suite' \
		'  make lint     Run Ruff checks' \
		'  make check    Run tests, lint and bytecode compilation' \
		'' \
		'Overrides: PYTHON=/path/to/python3.13 VENV=.venv PROFILE=dev|runtime'

setup:
	@PYTHON="$(PYTHON)" VENV="$(VENV)" PROFILE="$(PROFILE)" ./scripts/setup.sh

run: setup
	@"$(VENV_PYTHON)" -m uvicorn app:app \
		--host "$(HOST)" \
		--port "$(PORT)" \
		--workers "$(WEB_CONCURRENCY)" \
		$(UVICORN_ARGS)

shell: setup
	@VIRTUAL_ENV="$(abspath $(VENV))" \
		PATH="$(abspath $(VENV))/bin:$$PATH" \
		"$${SHELL:-/bin/sh}"

test: setup
	@"$(VENV_PYTHON)" -m pytest -q $(PYTEST_ARGS)

lint: setup
	@"$(VENV_PYTHON)" -m ruff check app.py harness tests

compile: setup
	@"$(VENV_PYTHON)" -m compileall -q app.py harness

check: test lint compile
