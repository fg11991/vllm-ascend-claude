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

from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler


class AscendRejectionSampler310(AscendRejectionSampler):
    """310P-specific rejection sampler.

    Inherits from AscendRejectionSampler to provide the same Triton-optimized
    rejection sampling on 310P. The Triton kernels automatically adapt to
    310P's vector core count via get_vectorcore_num().

    The 310P model runner previously used the upstream RejectionSampler which
    lacks prepare_sampling() and Ascend-specific optimizations. This class
    enables the full Ascend rejection sampling pipeline on 310P.
    """

    pass
