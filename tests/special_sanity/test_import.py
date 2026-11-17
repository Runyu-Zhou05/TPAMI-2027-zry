# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def test_import():
    import verl

    print(verl.__version__)


def test_single_controller_import():
    import verl.single_controller

    print(verl.single_controller.__version__)


def test_fve_production_worker_import():
    """Catch missing source subpackages that shallow `import verl` cannot see."""
    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.models.transformers.qwen2_vl import get_rope_index
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    assert callable(apply_monkey_patch)
    assert callable(get_rope_index)
    assert ActorRolloutRefWorker is not None
