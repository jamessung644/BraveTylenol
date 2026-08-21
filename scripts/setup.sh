#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_root=$(dirname -- "$script_dir")
cd "$project_root"

python_request=${PYTHON:-python3.13}
venv_request=${VENV:-.venv}
profile=${PROFILE:-dev}

case "$profile" in
    dev)
        requirements_file=requirements-dev.txt
        fingerprint_files="requirements.txt requirements-dev.txt"
        ;;
    runtime)
        requirements_file=requirements.txt
        fingerprint_files=requirements.txt
        ;;
    *)
        printf 'Unsupported PROFILE=%s (expected dev or runtime)\n' "$profile" >&2
        exit 2
        ;;
esac

if command -v "$python_request" >/dev/null 2>&1; then
    python_executable=$(command -v "$python_request")
elif [ "$python_request" = "python3.13" ] && [ -x /opt/homebrew/bin/python3.13 ]; then
    python_executable=/opt/homebrew/bin/python3.13
elif [ "$python_request" = "python3.13" ] && [ -x /usr/local/bin/python3.13 ]; then
    python_executable=/usr/local/bin/python3.13
else
    printf 'Python 3.13 was not found. Install it or run PYTHON=/path/to/python3.13 make setup.\n' >&2
    printf 'On macOS with Homebrew: brew install python@3.13\n' >&2
    exit 1
fi

python_version=$(
    "$python_executable" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
)
if [ "$python_version" != "3.13" ]; then
    printf 'Expected Python 3.13, but %s reports %s.\n' "$python_executable" "$python_version" >&2
    exit 1
fi

case "$venv_request" in
    /*) venv_dir=$venv_request ;;
    *) venv_dir=$project_root/$venv_request ;;
esac
venv_python=$venv_dir/bin/python

if [ ! -x "$venv_python" ]; then
    printf 'Creating virtual environment at %s with %s\n' "$venv_dir" "$python_executable"
    "$python_executable" -m venv "$venv_dir"
else
    venv_version=$(
        "$venv_python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
    )
    if [ "$venv_version" != "3.13" ]; then
        printf 'Existing environment %s uses Python %s, not 3.13.\n' "$venv_dir" "$venv_version" >&2
        printf 'Choose another VENV path or remove the generated environment before retrying.\n' >&2
        exit 1
    fi
fi

fingerprint=$(
    "$venv_python" - $fingerprint_files <<'PY'
import hashlib
import pathlib
import sys

digest = hashlib.sha256()
for raw_path in sys.argv[1:]:
    path = pathlib.Path(raw_path)
    digest.update(path.name.encode())
    digest.update(path.read_bytes())
print(digest.hexdigest())
PY
)
stamp_file=$venv_dir/.brave-tylenol-$profile-requirements
installed_fingerprint=""
if [ -f "$stamp_file" ]; then
    installed_fingerprint=$(sed -n '1p' "$stamp_file")
fi

if [ "$fingerprint" != "$installed_fingerprint" ]; then
    printf 'Installing %s dependencies from %s\n' "$profile" "$requirements_file"
    "$venv_python" -m pip install --upgrade pip
    "$venv_python" -m pip install -r "$requirements_file"
    printf '%s\n' "$fingerprint" >"$stamp_file"
else
    printf 'Virtual environment is already up to date (%s profile).\n' "$profile"
fi

printf 'Ready: %s (%s)\n' "$venv_python" "$("$venv_python" --version 2>&1)"
