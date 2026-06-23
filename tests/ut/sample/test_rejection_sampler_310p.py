#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.sample.rejection_sampler import (
    AscendRejectionSampler,
    rejection_random_sample_pytorch,
)

PLACEHOLDER_TOKEN_ID = -1


def mock_pin_memory(original_func):
    def func_wo_pin_memory(*args, **kwargs):
        if kwargs.get("pin_memory", False):
            kwargs["pin_memory"] = False
        return original_func(*args, **kwargs)

    return func_wo_pin_memory


class TestRejectionSampler310P(TestBase):
    """Tests for 310P rejection sampling functionality.

    Verifies that:
    1. AscendRejectionSampler310 correctly inherits from AscendRejectionSampler
    2. The rejection sampling logic works correctly for 310P scenarios
       (small vectorcore count, PyTorch fallback paths)
    3. Edge cases specific to 310P constraints are handled
    """

    def test_rejection_sampler_310p_inheritance(self):
        """AscendRejectionSampler310 should inherit from AscendRejectionSampler."""
        from vllm_ascend._310p.sample.rejection_sampler_310p import (
            AscendRejectionSampler310,
        )

        assert issubclass(AscendRejectionSampler310, AscendRejectionSampler)

    def test_rejection_sampler_310p_has_prepare_sampling(self):
        """AscendRejectionSampler310 must have prepare_sampling method
        since NPUModelRunner calls it."""
        from vllm_ascend._310p.sample.rejection_sampler_310p import (
            AscendRejectionSampler310,
        )

        assert hasattr(AscendRejectionSampler310, "prepare_sampling")

    def test_rejection_sampler_310p_init(self):
        """AscendRejectionSampler310 should initialize with a sampler."""
        from vllm_ascend._310p.sample.rejection_sampler_310p import (
            AscendRejectionSampler310,
        )

        mock_sampler = MagicMock()
        sampler = AscendRejectionSampler310(mock_sampler)
        assert sampler.sampler is mock_sampler
        assert sampler._ascend_optimizations_enabled is True

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_random_sample_small_batch(self):
        """Test rejection sampling with batch_size <= 8 (310P vectorcore count).

        On 310P with ~8 vectorcores, small batches result in grid=batch_size,
        block_size=1. This tests the typical 310P scenario.
        """
        batch_size = 4
        max_spec_len = 2
        vocab_size = 4
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2, 4, 6, 7])
        draft_token_ids = torch.tensor([1, 0, 2, 1, 0, 2, 3])
        draft_probs = torch.tensor(
            [
                [0.0, 0.6, 0.0, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.5, 0.0, 0.0],
                [0.0, 0.6, 0.0, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.5, 0.0, 0.0],
                [0.1, 0.2, 0.3, 0.4],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.0, 0.8, 0.0, 0.2],
                [0.2, 0.1, 0.3, 0.4],
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 0.8, 0.0, 0.2],
                [0.2, 0.1, 0.3, 0.4],
                [0.9, 0.1, 0.0, 0.0],
                [0.2, 0.1, 0.3, 0.4],
            ]
        )
        bonus_token_ids = torch.tensor([[100], [200], [300], [400]])
        recovered_token_ids = torch.tensor([1, 2, 3, 1, 2, 3, 2])
        uniform_probs = torch.tensor([0.7, 0.6, 0.5, 0.7, 0.6, 0.5, 0.6])
        is_greedy = torch.tensor([False, False, False, False])

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
        )

        # Verify first request: draft[1]=0.6, target[1]=0.8, ratio=1.33 >= 0.7 -> accept
        assert output_token_ids[0, 0].item() == 1
        # Second token: draft[0]=0.1, target[0]=0.2, ratio=2.0 >= 0.6 -> accept
        assert output_token_ids[0, 1].item() == 0
        # All accepted -> bonus token
        assert output_token_ids[0, 2].item() == 100

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_random_sample_all_accepted(self):
        """Test case where all draft tokens are accepted."""
        batch_size = 1
        max_spec_len = 3
        vocab_size = 4
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([3])
        draft_token_ids = torch.tensor([1, 2, 3])
        draft_probs = torch.tensor(
            [
                [0.0, 0.5, 0.0, 0.5],
                [0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.0, 0.9, 0.0, 0.1],
                [0.0, 0.0, 0.9, 0.1],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        bonus_token_ids = torch.tensor([[99]])
        recovered_token_ids = torch.tensor([0, 0, 0])
        uniform_probs = torch.tensor([0.5, 0.5, 0.5])
        is_greedy = torch.tensor([False])

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
        )

        # All tokens should be accepted (target_prob/draft_prob >= 1.0 >= 0.5)
        assert output_token_ids[0, 0].item() == 1
        assert output_token_ids[0, 1].item() == 2
        assert output_token_ids[0, 2].item() == 3
        # Bonus token appended
        assert output_token_ids[0, 3].item() == 99

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_random_sample_all_rejected(self):
        """Test case where the first draft token is rejected."""
        batch_size = 1
        max_spec_len = 2
        vocab_size = 4
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([1, 2])
        # draft_prob for token 1 is 0.9 but target_prob is 0.1
        # ratio = 0.1/0.9 = 0.11 < 0.5 (uniform_prob) -> reject
        draft_probs = torch.tensor(
            [
                [0.0, 0.9, 0.0, 0.1],
                [0.0, 0.0, 0.9, 0.1],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.0, 0.1, 0.0, 0.9],
                [0.0, 0.0, 0.1, 0.9],
            ]
        )
        bonus_token_ids = torch.tensor([[99]])
        recovered_token_ids = torch.tensor([3, 3])
        uniform_probs = torch.tensor([0.5, 0.5])
        is_greedy = torch.tensor([False])

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
        )

        # First token rejected -> use recovered token
        assert output_token_ids[0, 0].item() == 3
        # No bonus token since not all accepted
        assert output_token_ids[0, 2].item() == PLACEHOLDER_TOKEN_ID

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_random_sample_ngram_mode(self):
        """Test rejection sampling in n-gram mode (NO_DRAFT_PROBS=True)."""
        batch_size = 1
        max_spec_len = 2
        vocab_size = 4
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([1, 2])
        # No draft probs in ngram mode
        target_probs = torch.tensor(
            [
                [0.0, 0.8, 0.0, 0.2],
                [0.0, 0.0, 0.9, 0.1],
            ]
        )
        bonus_token_ids = torch.tensor([[99]])
        recovered_token_ids = torch.tensor([0, 0])
        # In ngram mode, draft_prob=1, so acceptance = target_prob >= uniform_prob
        uniform_probs = torch.tensor([0.5, 0.5])
        is_greedy = torch.tensor([False])

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            None,  # draft_probs is None for ngram
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=True,
        )

        # token 1: target_prob[1]=0.8 / 1 = 0.8 >= 0.5 -> accept
        assert output_token_ids[0, 0].item() == 1
        # token 2: target_prob[2]=0.9 / 1 = 0.9 >= 0.5 -> accept
        assert output_token_ids[0, 1].item() == 2
        # All accepted -> bonus
        assert output_token_ids[0, 2].item() == 99

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_random_sample_spec_len_1(self):
        """Test with spec_len=1 (minimum speculative decoding length)."""
        batch_size = 2
        max_spec_len = 1
        vocab_size = 4
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([1, 2])
        draft_token_ids = torch.tensor([1, 2])
        draft_probs = torch.tensor(
            [
                [0.0, 0.5, 0.0, 0.5],
                [0.0, 0.0, 0.5, 0.5],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.0, 0.9, 0.0, 0.1],
                [0.0, 0.0, 0.1, 0.9],
            ]
        )
        bonus_token_ids = torch.tensor([[99], [88]])
        recovered_token_ids = torch.tensor([3, 3])
        uniform_probs = torch.tensor([0.5, 0.5])
        is_greedy = torch.tensor([False, False])

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
        )

        # Request 0: 1 draft token, ratio=0.9/0.5=1.8 >= 0.5 -> accept
        assert output_token_ids[0, 0].item() == 1
        # All 1 accepted -> bonus
        assert output_token_ids[0, 1].item() == 99

    def test_cal_grid_and_block_size_small_vectorcore(self):
        """Test grid/block calculation with small vectorcore count (310P scenario).

        On 310P with ~8 vectorcores, when batch_size > 8, grid=8 and
        block_size is a power of 2 >= ceil(batch_size/8).
        """
        from unittest.mock import patch as mock_patch

        from vllm_ascend.ops.triton.reject_sample import cal_grid_and_block_size

        # Simulate 310P with 8 vectorcores
        with mock_patch(
            "vllm_ascend.ops.triton.reject_sample.get_vectorcore_num",
            return_value=8,
        ):
            # batch_size <= vectorcore_num: grid=batch_size, block_size=1
            grid, block_size = cal_grid_and_block_size(4)
            assert grid == 4
            assert block_size == 1

            grid, block_size = cal_grid_and_block_size(8)
            assert grid == 8
            assert block_size == 1

            # batch_size > vectorcore_num: grid=8, block_size=next_power_of_2(ceil(bs/8))
            grid, block_size = cal_grid_and_block_size(16)
            assert grid == 8
            assert block_size == 2

            grid, block_size = cal_grid_and_block_size(64)
            assert grid == 8
            assert block_size == 8

            grid, block_size = cal_grid_and_block_size(100)
            assert grid == 8
            assert block_size == 16  # next_power_of_2(ceil(100/8)) = next_power_of_2(13) = 16
