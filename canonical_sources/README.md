# Canonical build-time prompt snapshot

This directory contains the five reviewed `l2-healthcare-instructions` documents used
to compile the committed runtime artifact. It is build-time provenance, not runtime
model input and not part of the Docker context.

Update these files only by copying a newly reviewed canonical set together, regenerate
the artifact, and commit the source snapshot, bundle, and SHA sidecar atomically.
`python scripts/compile_runtime_artifacts.py --check` must work in a clean standalone
checkout without a sibling repository.
