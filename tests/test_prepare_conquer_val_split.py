import hashlib
import json
import stat

import pytest

from scripts.prepare_conquer_val_split import (
    COEVAL_COMMIT,
    CONFIRM_COUNT,
    DEV_COUNT,
    GENERATION,
    KNOWN_JUDGE_TOUCHED_COUNT,
    PUBLIC_ID_COUNT,
    build_split,
    main,
)


def _public_ids() -> list[str]:
    return [f"public-val-{index:03d}" for index in range(PUBLIC_ID_COUNT)]


def test_v2_split_is_deterministic_complete_and_forces_all_touched_ids_into_dev():
    public_ids = _public_ids()
    known_touched = set(sorted(public_ids)[:KNOWN_JUDGE_TOUCHED_COUNT])
    additional_touched = {public_ids[-1], public_ids[-2]}

    first = build_split(public_ids, additional_touched_ids=sorted(additional_touched))
    second = build_split(
        list(reversed(public_ids)),
        additional_touched_ids=list(reversed(sorted(additional_touched))),
    )

    assert first == second
    assert len(first.dev_ids) == DEV_COUNT
    assert len(first.confirm_ids) == CONFIRM_COUNT
    assert set(first.dev_ids).isdisjoint(first.confirm_ids)
    assert set(first.dev_ids) | set(first.confirm_ids) == set(public_ids)
    assert known_touched == first.known_touched_ids
    assert additional_touched == first.additional_touched_ids
    assert known_touched | additional_touched <= set(first.dev_ids)
    assert list(first.dev_ids) == sorted(first.dev_ids)
    assert list(first.confirm_ids) == sorted(first.confirm_ids)


def test_hash_generation_changes_membership_without_moving_known_touched_ids():
    public_ids = _public_ids()

    v2 = build_split(public_ids)
    alternate = build_split(public_ids, generation="independent-test-generation")

    assert v2.known_touched_ids == alternate.known_touched_ids
    assert v2.known_touched_ids <= set(v2.dev_ids)
    assert alternate.known_touched_ids <= set(alternate.dev_ids)
    assert v2.dev_ids != alternate.dev_ids


def test_touched_input_must_be_a_subset_of_published_conquer_val():
    with pytest.raises(ValueError, match="outside the published conquer_val"):
        build_split(_public_ids(), additional_touched_ids=["held-out-or-typo-id"])


def test_cli_writes_private_coeval_id_files_and_id_free_metadata(tmp_path, capsys):
    public_ids = _public_ids()
    source = tmp_path / "conquer_val_ids.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "split": "val",
                "prompt_ids": public_ids,
            }
        )
    )
    touched = tmp_path / "local-touched.json"
    touched.write_text(json.dumps({"prompt_ids": [public_ids[-1]]}))
    output_dir = tmp_path / "sealed-split"

    assert (
        main(
            [
                "--ids",
                str(source),
                "--touched-ids",
                str(touched),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    dev_path = output_dir / "dev120.ids.json"
    confirm_path = output_dir / "confirm181.ids.json"
    metadata_path = output_dir / "split-metadata.json"
    dev_payload = json.loads(dev_path.read_text())
    confirm_payload = json.loads(confirm_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    stdout = capsys.readouterr().out

    assert set(dev_payload) == {"prompt_ids"}
    assert set(confirm_payload) == {"prompt_ids"}
    assert len(dev_payload["prompt_ids"]) == DEV_COUNT
    assert len(confirm_payload["prompt_ids"]) == CONFIRM_COUNT
    assert public_ids[-1] in dev_payload["prompt_ids"]
    assert all(prompt_id not in stdout for prompt_id in public_ids)
    assert all(prompt_id not in metadata_path.read_text() for prompt_id in public_ids)

    assert metadata["coeval"] == {
        "commit": COEVAL_COMMIT,
        "held_out_ids_accessed": False,
        "public_id_count": PUBLIC_ID_COUNT,
        "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "suite": "conquer_val",
    }
    assert metadata["partition"]["generation"] == GENERATION
    assert metadata["partition"]["known_prior_judge_touched_forced_dev_count"] == 3
    assert metadata["partition"]["additional_touched_forced_dev_count"] == 1
    assert metadata["partition"]["strict_prompt_holdout"] is False
    assert metadata["partition"]["prospective_score_holdout"] is True
    assert metadata["content_policy"] == {
        "contains_answers": False,
        "contains_credentials": False,
        "contains_prompt_ids": False,
        "contains_prompt_text": False,
        "contains_rubrics": False,
    }
    assert (
        metadata["outputs"]["dev120"]["file_sha256"]
        == hashlib.sha256(dev_path.read_bytes()).hexdigest()
    )
    assert (
        metadata["outputs"]["confirm181"]["file_sha256"]
        == hashlib.sha256(confirm_path.read_bytes()).hexdigest()
    )
    for path in (dev_path, confirm_path, metadata_path):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_cli_refuses_to_overwrite_a_partition_generation(tmp_path):
    source = tmp_path / "conquer_val_ids.json"
    source.write_text(json.dumps({"prompt_ids": _public_ids()}))
    output_dir = tmp_path / "sealed-split"
    arguments = ["--ids", str(source), "--output-dir", str(output_dir)]

    assert main(arguments) == 0
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(arguments)
