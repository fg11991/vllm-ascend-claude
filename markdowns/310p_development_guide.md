# Ascend 310P 算子开发注意事项

本文档面向在 vllm-ascend 项目中进行 310P 适配开发的工程师，整理了 310P 与 910B/C 的关键差异以及开发中需要额外关注的事项。

---

## 目录

- [硬件规格对比](#硬件规格对比)
- [架构差异与影响](#架构差异与影响)
- [vllm-ascend 中的 310P 适配架构](#vllm-ascend-中的-310p-适配架构)
- [算子开发关键差异](#算子开发关键差异)
- [Attention 与 KV Cache 差异](#attention-与-kv-cache-差异)
- [量化支持差异](#量化支持差异)
- [内存与性能约束](#内存与性能约束)
- [开发检查清单](#开发检查清单)

---

## 硬件规格对比

| 参数 | Ascend 310P3 | Ascend 910B (B2/B3) | Ascend 910C |
|------|-------------|---------------------|-------------|
| **定位** | 推理 | 训练 + 推理 | 训练 + 推理 |
| **AI Core 数量** | 8 | 20-24 | 更多 |
| **显存** | 24 GB LPDDR4X | 64 GB HBM2e | 128 GB HBM3 |
| **内存带宽** | 204.8 GB/s | ~392 GB/s | ~784 GB/s |
| **INT8 算力** | 140 TOPS | 752-828 TOPS | 1600 TOPS |
| **FP16 算力** | 70 TFLOPs | - | - |
| **FP32 算力** | - | 313-414 TFLOPs | 800 TFLOPs |
| **功耗** | 72W | 310W | - |
| **Cube 单元** | 每 Core 1 个 | 每 Core 1 个 | 每 Core 1 个 |
| **Vector 单元** | 每 Core 2 个 | 每 Core 2 个 | 每 Core 2 个 |

### 关键差异总结

1. **AI Core 数量差距大**：310P 仅 8 个 Core，910B 有 20-24 个，多核并行能力差距显著
2. **内存体系不同**：310P 使用 LPDDR4X（低带宽），910B 使用 HBM2e（高带宽）
3. **算力量级差异**：310P INT8 算力仅为 910B 的约 1/5
4. **定位差异**：310P 专注推理场景，不支持训练

---

## 架构差异与影响

### 1. 内存带宽瓶颈

310P 的内存带宽（204.8 GB/s）不到 910B（392 GB/s）的一半。这意味着：

- **访存密集型算子受限更大**：Softmax、LayerNorm 等 bandwidth-bound 算子在 310P 上的性能瓶颈更突出
- **Tiling 策略需要更激进地减少 GM 访问**：尽可能复用 UB 中的数据
- **双缓冲的收益更明显**：因为 DMA 延迟相对更高

### 2. AI Core 数量限制

8 个 AI Core 对多核 Tiling 有直接影响：

- 910B 上的 Tiling 可以假设 20+ Core 并行，310P 需要重新设计分块粒度
- 如果单个 Block 的计算量太小，8 Core 无法充分利用
- 如果单个 Block 太大，超出 UB 容量

### 3. 数据格式要求

310P 在内存布局上有特殊要求：

- **FRACTAL_NZ 格式强制启用**：310P 始终将支持的权重转换为 FRACTAL_NZ 格式（见 `vllm_ascend/utils.py`）
- 910B 上 FRACTAL_NZ 是可选的（由 `VLLM_ASCEND_ENABLE_NZ` 环境变量控制）
- 需要确保算子能正确处理 NZ 格式的输入/输出

```python
# vllm_ascend/utils.py 中的逻辑
# 310P always converts to NZ.
if is_310p():
    return True  # 强制启用 FRACTAL_NZ
```

---

## vllm-ascend 中的 310P 适配架构

### 目录结构

项目中 310P 的适配代码集中在 `vllm_ascend/_310p/` 目录下：

```
vllm_ascend/_310p/
├── __init__.py
├── model_runner_310p.py       # NPUModelRunner310 - 310P 专用模型运行器
├── worker_310p.py             # 310P Worker 实现
├── npu_input_batch.py         # 310P 输入批处理
├── block_table.py             # 310P 块表管理
├── sharded_state_loader_310p.py
├── attention/
│   ├── attention_v1.py        # AscendAttentionBackend310 / AscendAttentionBackendImpl310
│   ├── attention_mask.py      # AttentionMaskBuilder310
│   └── metadata_builder.py    # AscendAttentionMetadataBuilder310
├── fused_moe/
│   ├── fused_moe.py           # AscendFusedMoE310
│   ├── moe_mlp.py
│   ├── moe_comm_method.py
│   ├── token_dispatcher.py
│   └── experts_selector.py
├── ops/
│   ├── conv.py                # AscendConv3dLayer310
│   ├── rotary_embedding.py    # AscendRotaryEmbedding310 / AscendMRotaryEmbedding310
│   ├── vocab_parallel_embedding.py
│   ├── causal_conv1d.py
│   └── fla/                   # Flash Linear Attention
│       ├── gdn_310.py         # AscendGatedDeltaNetAttention310
│       ├── chunk_gated_delta_rule.py
│       └── fused_gdn_gating.py
├── quantization/
│   ├── modelslim_config.py
│   └── methods/
│       ├── w8a8_base.py
│       ├── w8a8_dynamic.py
│       ├── w8a8_static.py
│       ├── w8a8s.py
│       ├── w8a8sc.py
│       └── registry.py
└── sample/
    └── sampler.py             # AscendSampler310
```

### 适配模式

310P 适配采用**继承 + 覆写**模式：

```python
# model_runner_310p.py
class NPUModelRunner310(NPUModelRunner):
    # 继承 910 的 NPUModelRunner，覆写 310P 特有行为
    ...

# attention/attention_v1.py
class AscendAttentionBackend310(AscendAttentionBackend):
    # 覆写 KV Cache shape、metadata builder 等
    ...

class AscendAttentionBackendImpl310(AscendAttentionBackendImpl):
    # 覆写 prefill/decode 的 attention 实现
    ...
```

### 设备检测与算子注册

在 `vllm_ascend/utils.py` 中，通过 `is_310p()` 函数判断当前设备类型，并在初始化时注册 310P 专用算子实现：

```python
# utils.py 中的条件注册
if is_310p():
    from vllm_ascend._310p.fused_moe.fused_moe import AscendFusedMoE310
    from vllm_ascend._310p.ops.activation import AscendSiluAndMul310
    from vllm_ascend._310p.ops.conv import AscendConv3dLayer310
    from vllm_ascend._310p.ops.layernorm import ...
    from vllm_ascend._310p.ops.rotary_embedding import AscendRotaryEmbedding310
    # ... 替换注册表中的算子实现
```

---

## 算子开发关键差异

### 1. KV Cache 形状不同

**910B** 使用标准 4D KV Cache：
```python
# 标准形状
(num_blocks, block_size, num_kv_heads, head_size)
```

**310P** 使用 5D KV Cache，需要 16 对齐：
```python
# 310P 特殊形状 - head_size 维度按 16 拆分
(2, num_blocks, (num_kv_heads * head_size) // 16, block_size, 16)
```

这是由 310P 硬件内存对齐要求决定的，所有涉及 KV Cache 读写的算子都必须适配。

### 2. Attention 算子差异

310P 使用不同的底层 API：

| 操作 | 910B | 310P |
|------|------|------|
| Prefill | `torch_npu.npu_fusion_attention` | `torch_npu._npu_flash_attention` |
| Decode | `torch_npu.npu_paged_attention` | `torch_npu._npu_paged_attention` |
| Mask 构建 | `AscendAttentionMaskBuilder` | `AttentionMaskBuilder310` |

310P 的 prefill 还需要处理**内存对齐 padding**：
```python
# 310P prefill 中的 padding 处理
real_tokens = int(attn_metadata.seq_lens.sum().item())
aligned_tokens = int(query.shape[0])
delta = aligned_tokens - real_tokens
# 当有 padding 时，调整最后一个请求的 seq_len
```

### 3. 支持的 Block Size

```python
# 910B 支持更多 block size 选项
get_supported_kernel_block_sizes() → [128, 64, 32, 16]

# 310P 仅支持
get_supported_kernel_block_sizes() → [128, 64]
```

### 4. FRACTAL_NZ 格式处理

310P 上所有支持的权重张量会被自动转换为 FRACTAL_NZ 格式。开发新算子时需要：

- 确认输入张量是否为 NZ 格式
- 如果算子不支持 NZ 格式输入，需要在算子内部做格式转换
- 注意 NZ 格式对 shape 推导的影响

### 5. Sampler 差异

310P 使用专用的 `AscendSampler310`，在采样逻辑上可能有设备特定的优化或限制。

### 6. Seq_lens 设备放置

310P 的 paged attention 需要显式检查 `seq_lens` 张量的设备：
```python
# 310P attention 中的设备检查
if attn_metadata.seq_lens.device != query.device:
    attn_metadata.seq_lens = attn_metadata.seq_lens.to(device=query.device, non_blocking=True)
```

---

## 量化支持差异

310P 有专门的量化实现，位于 `vllm_ascend/_310p/quantization/`：

| 量化方法 | 910B | 310P |
|----------|------|------|
| W8A8 动态量化 | 支持 | 支持（专用实现） |
| W8A8 静态量化 | 支持 | 支持（专用实现） |
| W8A8S | - | 支持 |
| W8A8SC | - | 支持 |
| ModelSlim 配置 | 通用 | 专用配置 |

开发新的量化算子时，需要同时考虑两个平台的实现。

---

## 内存与性能约束

### 310P 特殊约束

| 约束项 | 详情 |
|--------|------|
| **显存仅 24 GB** | 模型大小受限，大模型必须量化或分片 |
| **LPDDR4X 带宽** | 访存瓶颈更突出，需更激进的计算-访存优化 |
| **8 AI Core** | 多核并行度低，需要合理的工作负载分配 |
| **不支持训练** | 仅推理场景，反向传播相关算子不需要适配 |
| **Block Size 限制** | 仅支持 128 和 64 的 block size |

### 性能优化建议

1. **减少 GM 访问次数**：310P 的内存带宽是主要瓶颈
2. **增大单 Core 工作量**：AI Core 少，每个 Core 应分配更多计算
3. **利用 FRACTAL_NZ**：NZ 格式本身是为提升内存访问效率设计的，不要做多余的格式转换
4. **量化优先**：24 GB 显存下，W8A8 量化几乎是必选项
5. **避免不必要的 tensor.item()**：310P 上的 CPU-NPU 同步开销可能更高（LPDDR vs HBM）

---

## 开发检查清单

在为 310P 开发或适配算子时，请逐项确认：

### 功能正确性

- [ ] 算子是否正确处理 FRACTAL_NZ 格式的输入？
- [ ] KV Cache 是否使用 5D 形状 `(2, num_blocks, hidden_dim_aligned, block_size, 16)`？
- [ ] Block Size 是否仅使用 128 或 64？
- [ ] 数据对齐是否满足 16 倍数要求？
- [ ] `seq_lens` 张量是否在正确的设备上？

### 适配架构

- [ ] 310P 专用代码是否放在 `vllm_ascend/_310p/` 目录下？
- [ ] 是否通过继承 910 的基类实现，只覆写差异部分？
- [ ] 是否在 `utils.py` 的 `is_310p()` 分支中注册了新算子？
- [ ] 公共逻辑是否保留在父类中，避免代码重复？

### 性能验证

- [ ] 在 310P 实际硬件上测试通过？
- [ ] Tiling 策略是否针对 8 Core 和 204.8 GB/s 带宽优化？
- [ ] 是否避免了热路径上的 `tensor.item()` 同步？
- [ ] 内存使用是否在 24 GB 限制内？

### 测试

- [ ] 单元测试覆盖 310P 路径？
- [ ] 精度与 910B 实现对齐？
- [ ] 边界条件测试（极小 batch、极大 seq_len）？

---

## 参考资料

- [昇腾 NPU 硬件参数汇总](https://blog.ailemon.net/2025/05/24/huawei-ascend-npu-params-for-ai/)
- [昇腾芯片命名体系](https://blog.csdn.net/weixin_44659309/article/details/145998682)
- [Ascend 310 → 310P 应用迁移](https://www.cnblogs.com/xiaowangyun/p/16329914.html)
- [AscendC 算子开发工程指南](https://bbs.huaweicloud.com/blogs/439155)
- [Ascend C 高性能算子开发全解析](https://blog.csdn.net/2501_94641745/article/details/155916861)
- [310P 推理应用开发体验](https://blog.csdn.net/qq_40679625/article/details/125136174)
- [vLLM Ascend 官方文档](https://docs.vllm.ai/projects/ascend/en/latest/)
