# rejection_random_sample_kernel 310P 移植设计文档

## 1. 概述

将 `vllm_ascend/ops/triton/reject_sample.py` 中的 `rejection_random_sample_kernel` 及相关算子从 Ascend 910B 移植到 Ascend 310P3。

### 1.1 背景

- **当前状态**：310P 的 `NPUModelRunner310` 使用上游 vLLM 的 `RejectionSampler`（非 Ascend 优化版），不使用 Triton kernel
- **目标**：为 310P 提供 Ascend 优化的 rejection sampling 实现，通过 `AscendRejectionSampler` 使用 Triton kernel 或 PyTorch fallback
- **动机**：提升 310P 上 speculative decoding 的性能

---

## 2. 910B 原始算子分析

### 2.1 涉及的 Kernel 函数

| Kernel | 用途 | 310P 需要移植 |
|--------|------|:---:|
| `rejection_random_sample_kernel` | 随机拒绝采样（非 block verify） | 是 |
| `rejection_random_sample_block_verify_kernel` | 随机拒绝采样（block verify 模式，max_spec_len>=3） | 是 |
| `rejection_greedy_sample_triton` | 贪心拒绝采样 | 是 |
| `rejection_greedy_sample_spec_len_1_triton` | 贪心拒绝采样（spec_len=1 快速路径） | 是 |
| `sample_recovered_tokens_kernel` | 采样恢复 token（被拒绝时的 fallback） | 是 |
| `expand_kernel` | batch 扩展为 token 级别 | 是 |
| `bonus_renew` / `bonus_renew_1` | 辅助函数（bonus token 写入） | 是（被上面的 kernel 调用） |

### 2.2 算子特征

- **算子类型**：Vector-only（无 `tl.dot`），纯标量/向量操作
- **计算模式**：逐请求串行处理，请求间通过 BLOCK_SIZE 并行
- **内存访问**：离散随机访问（按 token_id 索引 vocab 维度）
- **constexpr 分支**：`NO_DRAFT_PROBS`、`ENABLE_REDUCE_SAMPLING`、`BLOCK_VERIFY`

### 2.3 Grid/Block 计算（910B）

```python
def cal_grid_and_block_size(batch_size: int):
    vectorcore_num = get_vectorcore_num()  # 910B: ~40 vectorcores
    if batch_size <= vectorcore_num:
        grid = batch_size
        block_size = 1
    else:
        grid = vectorcore_num
        block_size = triton.next_power_of_2(triton.cdiv(batch_size, grid))
    return grid, block_size
```

**关键**：`get_vectorcore_num()` 通过运行时 `triton.runtime.driver.active.utils.get_device_properties()` 获取，在 310P 上会返回 310P 的实际 vectorcore 数量（预计为 8 或更少）。

---

## 3. 310P 硬件约束分析

### 3.1 关键参数对比

| 参数 | 310P (DAV_2002) | 910B (DAV_2201) |
|------|:---:|:---:|
| NpuArch | 2002 | 2201 |
| UB | 256 KB | 192 KB |
| L0C | 256 KB | 128 KB |
| L2 | 16 MB | 192 MB |
| Memory | 24 GB | 64 GB |
| 核心类型 | AICore + VectorCore（分离） | AICore（集成） |
| Vector Core 数 | ~8 | ~40 |

### 3.2 对 rejection sampling kernel 的影响

1. **Vector Core 数量少**（~8 vs ~40）：`cal_grid_and_block_size()` 的 grid 更小，每个 core 需要处理更多请求（更大的 BLOCK_SIZE）
2. **UB 容量更大**（256 KB vs 192 KB）：实际上有利，Triton 编译器在 310P 上有更多 UB 空间
3. **内存带宽低**（~204.8 GB/s vs ~392 GB/s）：离散访存的代价更高，但 rejection sampling 的主要瓶颈不在带宽
4. **L2 更小**（16 MB vs 192 MB）：对大 vocab_size 的 prob 表查找可能有影响，但影响有限

### 3.3 关键判断

**`rejection_random_sample_kernel` 及其相关 kernel 在 310P 上无需修改 kernel 逻辑本身。** 原因：

- Kernel 是纯 Triton JIT，无 Ascend 特有 API 调用
- `get_vectorcore_num()` 运行时自动获取 310P 的 core 数
- `cal_grid_and_block_size()` 根据实际 vectorcore 数计算 grid/block
- 所有 Triton 编译器适配（UB 管理、指令选择）由 Triton-Ascend 后端自动处理
- 没有使用任何 910B 特有的 API（如 `npu_fusion_attention`）

**真正需要适配的是 Python wrapper 层和集成层**，让 310P 也能使用 `AscendRejectionSampler`。

---

## 4. 移植方案

### 4.1 方案选择

**方案 A（推荐）**：修改 310P `model_runner_310p.py`，使其使用 `AscendRejectionSampler` 替代上游 `RejectionSampler`

```python
# 当前（使用上游 RejectionSampler）
from vllm.v1.sample.rejection_sampler import RejectionSampler
self.rejection_sampler = RejectionSampler(self.sampler)

# 改为（使用 Ascend 优化的 RejectionSampler）
from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler
self.rejection_sampler = AscendRejectionSampler(self.sampler)
```

**为何不需要方案 B（在 `_310p/` 下创建单独的 kernel 文件）**：

- Triton kernel 本身是通用的，通过 `get_vectorcore_num()` 自动适配不同核数
- `cal_grid_and_block_size()` 自动适配 310P
- 无需覆写 kernel 逻辑

### 4.2 潜在风险与处理

| 风险 | 可能性 | 缓解措施 |
|------|:---:|---------|
| 310P Triton 编译器不支持某些 tl 操作 | 低 | Triton-Ascend 后端支持 310P (DAV_2002)；若不支持，fallback 到 PyTorch 实现（rejection_sampler.py 已有 `if HAS_TRITON` 分支） |
| `get_vectorcore_num()` 在 310P 返回异常值 | 低 | `init_device_properties_triton()` 会 assert 检查 |
| 310P 上 `exponential_()` 不可用 | 中 | 310P sampler 已有 `_random_sample_310p()` 用 CPU 生成，rejection sampler 的 `sample_recovered_tokens()` 也用了 `exponential_()`，需要验证 |
| BLOCK_SIZE 过大导致 Triton 编译失败 | 低 | batch_size 通常不大于几百，310P 8 核下 BLOCK_SIZE 最大约 64-128 |

### 4.3 `exponential_()` 兼容性

310P 的 `AscendSampler310` 用 `_random_sample_310p()` 将 `exponential_()` 在 CPU 上执行。`sample_recovered_tokens()` 中也直接在 NPU 上调用 `q.exponential_()`。

需要检查 310P 上 `torch.Tensor.exponential_()` 是否可用。如果不可用，需要在 `_310p/` 下提供 310P 版本的 `sample_recovered_tokens()`，类似 `_random_sample_310p()` 的处理方式。

---

## 5. 实现计划

### 5.1 文件修改

| 文件 | 操作 | 内容 |
|------|------|------|
| `vllm_ascend/_310p/model_runner_310p.py` | 修改 | `RejectionSampler` → `AscendRejectionSampler` |
| `vllm_ascend/_310p/sample/rejection_sampler_310p.py` | 新建 | 310P 版 rejection sampler（处理 `exponential_()` 等兼容性） |
| `tests/ut/sample/test_rejection_sampler_310p.py` | 新建 | 310P 场景的单元测试 |

### 5.2 不需要修改的文件

| 文件 | 原因 |
|------|------|
| `vllm_ascend/ops/triton/reject_sample.py` | Triton kernel 通用，无需改动 |
| `vllm_ascend/ops/triton/triton_utils.py` | 运行时检测，自动适配 |
| `vllm_ascend/utils.py` | 不需要在 `is_310p()` 分支注册新算子（rejection sampler 不走 `REGISTERED_ASCEND_OPS`） |

### 5.3 开发顺序

1. 创建 `_310p/sample/rejection_sampler_310p.py`，处理 310P 兼容性
2. 修改 `_310p/model_runner_310p.py`，使用 310P 版 rejection sampler
3. 编写单元测试
4. （需要 NPU 硬件）验证 Triton kernel 在 310P 上的编译和执行

---

## 6. 测试计划

### 6.1 单元测试（无需 NPU）

- 验证 310P rejection sampler 的 PyTorch fallback 路径
- 覆盖所有 constexpr 分支组合：
  - `NO_DRAFT_PROBS=True/False`
  - `ENABLE_REDUCE_SAMPLING=True/False`
  - `using_block_verify=True/False`
- 边界条件：batch_size=1、全接受、全拒绝、spec_len=1

### 6.2 集成测试（需要 310P NPU）

- Triton kernel 编译正确性
- `get_vectorcore_num()` 返回值正确
- `exponential_()` 行为验证
- 端到端 speculative decoding 精度对齐

---

## 7. 迁移进度检查清单

```
- [x] 识别输入来源与算子类型（Vector-only Triton kernel，910 → 310P）
- [x] 分析 310P 硬件约束（8 vectorcores, 256KB UB, 204.8 GB/s）
- [x] 先做最小迁移（修改 model_runner 使用 AscendRejectionSampler310）
- [x] 创建 AscendRejectionSampler310 类（继承 AscendRejectionSampler）
- [x] 更新 _310p/sample/__init__.py 导出新类
- [x] 编写单元测试（7 个测试用例）
- [x] 代码格式化和 lint 检查通过
- [ ] 验证 Triton 在 310P 上编译（需要硬件）
- [ ] 验证端到端精度
```
