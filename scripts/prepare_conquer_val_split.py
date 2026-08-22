#!/usr/bin/env python3
"""Prepare the public CoEval ``conquer_val`` development/confirmation split.

Only the published ``conquer_val_ids.json`` manifest is read.  This script does
not download HealthBench, inspect prompt/rubric/answer text, access
``conquer_test``, or use credentials.  The two ID files are intended to live
outside the repository and be supplied to CoEval through ``ids_path``.

The v2 policy puts the three public IDs used by the known historical
``first-3`` judge experiment in development before hash-partitioning the
remaining IDs.  An optional local touched-ID manifest can force more public IDs
into development without recording those IDs in the metadata report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COEVAL_COMMIT = "741263cfafba687f8baeb7422c747ef9557df1c4"
GENERATION = "coeval-741263-dev-v2"
PUBLIC_ID_COUNT = 301
DEV_COUNT = 120
CONFIRM_COUNT = 181
KNOWN_JUDGE_TOUCHED_COUNT = 3
SCHEMA_VERSION = "conquer-val-score-holdout-v2"


@dataclass(frozen=True, slots=True)
class Split:
    dev_ids: tuple[str, ...]
    confirm_ids: tuple[str, ...]
    known_touched_ids: frozenset[str]
    additional_touched_ids: frozenset[str]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _ids_newline_sha256(ids: Sequence[str]) -> str:
    return _sha256(("\n".join(ids) + "\n").encode())


def _load_prompt_ids(path: Path) -> tuple[list[str], bytes]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON") from error

    values = payload.get("prompt_ids") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path} must contain a nonempty prompt_ids list")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{path} contains a non-string or empty prompt ID")
    if len(values) != len(set(values)):
        raise ValueError(f"{path} contains duplicate prompt IDs")
    return values, raw


def build_split(
    public_ids: Sequence[str],
    *,
    additional_touched_ids: Sequence[str] = (),
    dev_count: int = DEV_COUNT,
    known_touched_count: int = KNOWN_JUDGE_TOUCHED_COUNT,
    generation: str = GENERATION,
) -> Split:
    """Build a deterministic score-holdout split from public IDs only."""

    if not generation:
        raise ValueError("generation must be nonempty")
    if not 0 <= known_touched_count <= dev_count < len(public_ids):
        raise ValueError("known_touched_count/dev_count are incompatible with input size")
    if any(not isinstance(value, str) or not value.strip() for value in public_ids):
        raise ValueError("public_ids contains a non-string or empty prompt ID")
    if len(public_ids) != len(set(public_ids)):
        raise ValueError("public_ids contains duplicate prompt IDs")

    public = set(public_ids)
    additional = set(additional_touched_ids)
    unknown_touched = additional - public
    if unknown_touched:
        raise ValueError("touched-ID input contains IDs outside the published conquer_val manifest")

    # The historical scoring helper selected the lexicographically first three
    # published val IDs.  Derive them from the public input rather than embedding
    # their values in source control.
    known = set(sorted(public)[:known_touched_count])
    forced_dev = known | additional
    if len(forced_dev) > dev_count:
        raise ValueError("touched public IDs exceed the development split capacity")

    remaining = public - forced_dev
    keyed = sorted(
        (
            hashlib.sha256(f"{generation}:{prompt_id}".encode()).hexdigest(),
            prompt_id,
        )
        for prompt_id in remaining
    )
    additional_dev_count = dev_count - len(forced_dev)
    dev = forced_dev | {prompt_id for _, prompt_id in keyed[:additional_dev_count]}
    confirm = public - dev

    if len(dev) != dev_count or len(confirm) != len(public) - dev_count:
        raise AssertionError("split count invariant failed")
    if dev & confirm or dev | confirm != public:
        raise AssertionError("split partition invariant failed")
    if not forced_dev <= dev:
        raise AssertionError("touched-ID containment invariant failed")

    # CoEval itself executes each explicit split in prompt-ID order.
    return Split(
        dev_ids=tuple(sorted(dev)),
        confirm_ids=tuple(sorted(confirm)),
        known_touched_ids=frozenset(known),
        additional_touched_ids=frozenset(additional - known),
    )


def _id_file(ids: Sequence[str]) -> bytes:
    # Keep the CoEval ids_path artifact minimal.  All provenance stays in the
    # separate metadata file, which intentionally carries no prompt IDs.
    return _canonical_json_bytes({"prompt_ids": list(ids)})


def build_metadata(
    *,
    source_bytes: bytes,
    split: Split,
    dev_bytes: bytes,
    confirm_bytes: bytes,
    touched_input_bytes: bytes | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "coeval": {
            "commit": COEVAL_COMMIT,
            "suite": "conquer_val",
            "public_id_count": len(split.dev_ids) + len(split.confirm_ids),
            "source_manifest_sha256": _sha256(source_bytes),
            "held_out_ids_accessed": False,
        },
        "partition": {
            "generation": GENERATION,
            "algorithm": "force_touched_then_sha256(generation:prompt_id)",
            "execution_order": "lexicographic_prompt_id",
            "known_prior_judge_touched_forced_dev_count": len(split.known_touched_ids),
            "additional_touched_forced_dev_count": len(split.additional_touched_ids),
            "additional_touched_membership_sha256": (
                _ids_newline_sha256(sorted(split.additional_touched_ids))
                if split.additional_touched_ids
                else None
            ),
            "touched_input_file_sha256": (
                _sha256(touched_input_bytes) if touched_input_bytes is not None else None
            ),
            "strict_prompt_holdout": False,
            "prospective_score_holdout": True,
            "known_prior_full_val_inference_artifact": True,
        },
        "outputs": {
            "dev120": {
                "filename": "dev120.ids.json",
                "prompt_id_count": len(split.dev_ids),
                "prompt_ids_newline_sha256": _ids_newline_sha256(split.dev_ids),
                "file_sha256": _sha256(dev_bytes),
            },
            "confirm181": {
                "filename": "confirm181.ids.json",
                "prompt_id_count": len(split.confirm_ids),
                "prompt_ids_newline_sha256": _ids_newline_sha256(split.confirm_ids),
                "file_sha256": _sha256(confirm_bytes),
            },
        },
        "content_policy": {
            "contains_prompt_ids": False,
            "contains_prompt_text": False,
            "contains_rubrics": False,
            "contains_answers": False,
            "contains_credentials": False,
        },
    }


def _write_private(path: Path, data: bytes, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing split artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o600)
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ids",
        type=Path,
        required=True,
        help="published CoEval src/coeval/data/conquer_val_ids.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="untracked directory for CoEval ids_path files and hash-only metadata",
    )
    parser.add_argument(
        "--touched-ids",
        type=Path,
        help="optional local JSON list/object of additional public val IDs to force into dev",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing generation's three output files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    public_ids, source_bytes = _load_prompt_ids(args.ids)
    if len(public_ids) != PUBLIC_ID_COUNT:
        raise ValueError(
            f"published conquer_val manifest must contain {PUBLIC_ID_COUNT} unique IDs"
        )

    touched_ids: list[str] = []
    touched_input_bytes: bytes | None = None
    if args.touched_ids is not None:
        touched_ids, touched_input_bytes = _load_prompt_ids(args.touched_ids)

    split = build_split(public_ids, additional_touched_ids=touched_ids)
    if len(split.confirm_ids) != CONFIRM_COUNT:
        raise AssertionError("public confirmation count invariant failed")

    dev_bytes = _id_file(split.dev_ids)
    confirm_bytes = _id_file(split.confirm_ids)
    metadata = build_metadata(
        source_bytes=source_bytes,
        split=split,
        dev_bytes=dev_bytes,
        confirm_bytes=confirm_bytes,
        touched_input_bytes=touched_input_bytes,
    )
    metadata_bytes = _canonical_json_bytes(metadata)

    artifacts = (
        (args.output_dir / "dev120.ids.json", dev_bytes),
        (args.output_dir / "confirm181.ids.json", confirm_bytes),
        (args.output_dir / "split-metadata.json", metadata_bytes),
    )
    existing = [str(path) for path, _ in artifacts if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "refusing to overwrite existing split artifacts: " + ", ".join(existing)
        )
    for path, data in artifacts:
        _write_private(path, data, overwrite=args.overwrite)

    # The CLI emits only the ID-free metadata record.  Prompt IDs remain in the
    # two local files needed by CoEval and never appear in terminal logs.
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
