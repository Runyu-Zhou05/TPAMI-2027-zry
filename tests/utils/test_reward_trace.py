import json
from types import SimpleNamespace
import time

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from verl.utils.reward_trace import RewardTraceRecorder


def test_reward_trace_records_every_rollout(tmp_path):
    batch = SimpleNamespace(
        non_tensor_batch={
            "uid": np.array(["a", "a", "b", "b"], dtype=object),
            "data_source": np.array(["rotation"] * 4, dtype=object),
            "extra_info": np.array([{"index": 10}, {"index": 10}, {"index": 11}, {"index": 11}], dtype=object),
            "reward_model": np.array(
                [
                    {"ground_truth": "top-left"},
                    {"ground_truth": "top-left"},
                    {"ground_truth": "bottom-right"},
                    {"ground_truth": "bottom-right"},
                ],
                dtype=object,
            ),
        }
    )
    reward = torch.tensor([[0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.0]])
    recorder = RewardTraceRecorder(tmp_path, "rotation", "run-001")
    computed_at = time.time_ns()
    responses = ["zero degrees", "wrong", "ninety degrees", "wrong again"]
    metrics = recorder.write_step(
        batch,
        reward,
        global_step=3,
        epoch=1,
        responses=responses,
        reward_computed_at_utc_ns=computed_at,
    )

    output = tmp_path / "rotation/step_00000003.parquet"
    table = pq.read_table(output)
    assert table.num_rows == 4
    assert table["run_id"].to_pylist() == ["run-001"] * 4
    assert table["response"].to_pylist() == responses
    assert table["reward_computed_at_utc_ns"].to_pylist() == [computed_at] * 4
    assert all(value >= computed_at for value in table["trace_written_at_utc_ns"].to_pylist())
    assert table["raw_reward"].to_pylist() == [1.0, 0.0, 1.0, 0.0]
    assert table["rollout_id"].to_pylist() == [0, 1, 0, 1]
    assert metrics["difficulty/raw_reward_mean"] == 0.5
    assert metrics["difficulty/all_rollouts_fail_rate"] == 0.0

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        recorder.write_step(
            batch,
            reward,
            global_step=3,
            epoch=1,
            responses=responses,
            reward_computed_at_utc_ns=time.time_ns(),
        )


def test_reward_trace_reconciles_uncommitted_steps(tmp_path):
    batch = SimpleNamespace(
        non_tensor_batch={
            "uid": np.array(["a"], dtype=object),
            "data_source": np.array(["rotation"], dtype=object),
            "extra_info": np.array([{"index": 1}], dtype=object),
            "reward_model": np.array([{"ground_truth": "90"}], dtype=object),
        }
    )
    reward = torch.tensor([[1.0]])
    recorder = RewardTraceRecorder(tmp_path, "rotation", "run-resume")
    for step in (1, 2, 3):
        recorder.write_step(
            batch,
            reward,
            global_step=step,
            epoch=0,
            responses=["<answer>90</answer>"],
            reward_computed_at_utc_ns=time.time_ns(),
        )

    quarantine = recorder.reconcile_to_checkpoint(1)
    assert (recorder.task_dir / "step_00000001.parquet").is_file()
    assert not (recorder.task_dir / "step_00000002.parquet").exists()
    assert not (recorder.task_dir / "step_00000003.parquet").exists()
    assert (quarantine / "step_00000002.parquet").is_file()
    assert (quarantine / "step_00000003.parquet").is_file()
    manifest = json.loads((quarantine / "rollback_manifest.json").read_text())
    assert manifest["committed_step"] == 1
    assert [item["step"] for item in manifest["traces"]] == [2, 3]
