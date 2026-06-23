# AscendC 算子开发技能与资源指南

本文档整理了开发 AscendC 算子所需的核心技能、开发工具、学习资源以及最佳实践。

---

## 目录

- [AscendC 概述](#ascendc-概述)
- [核心技能树](#核心技能树)
- [开发工具链](#开发工具链)
- [开发流程](#开发流程)
- [达芬奇架构关键概念](#达芬奇架构关键概念)
- [Agent Skills 生态](#agent-skills-生态)
- [学习资源](#学习资源)

---

## AscendC 概述

AscendC 是华为昇腾 CANN 针对算子开发场景推出的编程语言，具有以下特点：

- **原生 C/C++ 支持**：兼容标准 C/C++ 规范，降低学习门槛
- **多层接口抽象**：从简单到灵活的分层 API，满足不同开发需求
- **自动并行计算**：编译器自动完成内存分配、数据搬运和流水线并行
- **孪生调试**：支持 CPU 侧模拟调试，无需 NPU 硬件即可验证算子逻辑

与 CUDA 的主要差异：AscendC 采用 **Block-Thread + 显式流水线模型**，需要主动调度硬件资源（Cube/Vector/DMA），提供更高性能上限但需要更细致的优化。

---

## 核心技能树

### 1. C/C++ 编程基础

| 技能 | 说明 | 重要度 |
|------|------|--------|
| C++ 模板与类 | 核函数以类的形式组织（Init/Process/CopyIn/Compute/CopyOut） | 必备 |
| 指针与内存管理 | 理解片上 Buffer 的显式分配与回收 | 必备 |
| 宏与预处理 | CANN 头文件大量使用宏定义 | 重要 |

### 2. 达芬奇架构知识

| 技能 | 说明 | 重要度 |
|------|------|--------|
| AI Core 结构 | Cube 单元、Vector 单元、Scalar 引擎的职责划分 | 必备 |
| 内存层次 | HBM → Unified Buffer → L1/L0 → 寄存器的五级存储 | 必备 |
| SIMD 与矩阵运算 | Vector 1024-bit SIMD，Cube 16×16×16 矩阵乘 | 必备 |
| 流水线并行 | 三级流水线（CopyIn → Compute → CopyOut）与双缓冲 | 核心 |

### 3. Tiling（分块）策略

| 技能 | 说明 | 重要度 |
|------|------|--------|
| UB 容量估算 | Unified Buffer 2MB 限制下的最大分块计算 | 核心 |
| 对齐约束 | Cube 输入需 16 倍数对齐 | 核心 |
| Bank 冲突规避 | 合理排列数据避免存储 Bank 冲突 | 重要 |
| 多核 Tiling | 在多个 AI Core 间均匀分配工作负载 | 重要 |

### 4. 性能优化

| 技能 | 说明 | 重要度 |
|------|------|--------|
| 双缓冲 (Double Buffering) | 计算与搬运重叠，隐藏 DMA 延迟 | 核心 |
| 计算-搬运比 | 平衡 Cube/Vector 计算密度与 DMA 搬运开销 | 重要 |
| Profiling 分析 | 使用 msprof 工具定位性能瓶颈 | 重要 |
| 内存复用 | 最大化 UB 利用率，减少 HBM 访问 | 重要 |

### 5. 框架集成

| 技能 | 说明 | 重要度 |
|------|------|--------|
| PyTorch 自定义算子注册 | 通过 `torch.library` 或 `torch_npu` 注册算子 | 必备 |
| MindSpore AOT 编译 | 通过 `custom_compiler` 工具离线编译部署 | 按需 |
| ACLNN 接口封装 | 将 AscendC 算子封装为 ACLNN API 供 Python 调用 | 重要 |

### 6. Triton-Ascend（替代路径）

| 技能 | 说明 | 重要度 |
|------|------|--------|
| Triton 编程模型 | `@triton.jit` 装饰的 tile/block 分片模式 | 按需 |
| CUDA Triton 迁移 | GPU Triton kernel → Ascend Triton kernel 转换 | 按需 |
| vllm_ascend.ops.triton | 项目内 Triton 算子的两层模式（device kernel + Python wrapper） | 按需 |

---

## 开发工具链

### 核心工具

| 工具 | 用途 | 命令示例 |
|------|------|----------|
| **msOpGen** | 从算子原型 JSON 生成开发工程骨架 | `msopgen gen -i op.json -c ai_core-Ascend910B3 -lan cpp` |
| **msOpST** | 自动生成并执行算子测试用例 | `msopst run -i op.json -soc Ascend910B3` |
| **毕昇编译器** | AscendC 异构代码编译优化 | 集成于 `build.sh` |
| **msprof** | NPU 性能 Profiling | `msprof --application="./test" --output=./prof` |
| **npu-smi** | 查询设备型号与状态 | `npu-smi info`（获取 soc_version） |
| **MindStudio** | 华为 IDE，集成算子开发/调试/Profiling | GUI 工具 |

### 编译与部署

```bash
# 1. 生成算子工程
msopgen gen -i add_custom.json -c ai_core-<soc_version> -lan cpp

# 2. 编译算子包
cd AddCustom && ./build.sh

# 3. 安装到 OPP 算子库
./custom_opp_<os>_<arch>.run --install-path=$ASCEND_OPP_PATH/vendors/customize

# 4. 运行测试
msopst run -i add_custom.json -soc <soc_version>
```

---

## 开发流程

### AscendC 算子开发六步法

```
1. 算子分析        → 明确输入/输出 shape、dtype、计算逻辑
       ↓
2. 核函数设计      → 定义 KernelXxx 类，实现 Init/Process 方法
       ↓
3. Tiling 设计     → host 侧计算分块参数，满足 UB 容量和对齐约束
       ↓
4. 核函数实现      → device 侧 CopyIn → Compute → CopyOut 三级流水
       ↓
5. Host 侧封装     → 实现 shape 推导、tiling 计算、核函数调用
       ↓
6. 编译测试部署    → msOpGen/build.sh 编译，msOpST 测试
```

### 核函数结构模板

```cpp
class KernelAdd {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum) {
        // 初始化 GlobalTensor、分配 UB Buffer
    }

    __aicore__ inline void Process() {
        for (int32_t i = 0; i < tileNum; i++) {
            CopyIn(i);    // GM → UB
            Compute(i);   // UB 上计算
            CopyOut(i);   // UB → GM
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t idx) { /* DataCopy GM → UB */ }
    __aicore__ inline void Compute(int32_t idx) { /* Add/Mul/... */ }
    __aicore__ inline void CopyOut(int32_t idx) { /* DataCopy UB → GM */ }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
};
```

### 性能黄金法则

> **"让 Cube 不停工，让 UB 不空闲，让 GM 少访问"**

- 充分利用片上 Unified Buffer，最小化 HBM 访问
- 使用双缓冲隐藏 DMA 搬运延迟
- Cube 利用率可达 92% 以上

---

## 达芬奇架构关键概念

### 五级内存层次

| 层级 | 名称 | 大小 | 带宽 | 可编程性 |
|------|------|------|------|----------|
| L0 | 主机 DDR | 数百 GB | - | 通过 AscendCL |
| L1 | 全局内存 (HBM) | 32-128 GB | ~1.1 TB/s | Global Tensor |
| L2 | Unified Buffer | 2 MB/Core | 极高 | 显式管理 |
| L3 | L1/L0 Cache | 自动 | - | 自动管理 |
| L4 | 寄存器 | - | - | 编译器管理 |

**关键约束**：所有计算必须在 Unified Buffer 中进行，数据需通过 DMA 从 HBM 搬入。

### 计算单元对比

| 单元 | 功能 | 典型操作 |
|------|------|----------|
| **Cube** | 矩阵乘 (16×16×16) | MatMul, Conv, FC |
| **Vector** | SIMD 向量运算 (1024-bit) | Activation, LayerNorm, Softmax, RoPE |
| **Scalar** | 标量控制与地址计算 | 循环控制, 条件分支 |
| **DMA** | 双通道数据搬运 | GM↔UB 数据传输 |

---

## Agent Skills 生态

### awesome-ascend-skills

GitHub 仓库 [ascend-ai-coding/awesome-ascend-skills](https://github.com/ascend-ai-coding/awesome-ascend-skills) 提供了 100+ 结构化 Agent Skills，兼容 Claude Code、OpenCode、Cursor 等 AI 编程工具。

### 官方安装 Bundles

| Bundle | 目标用户 | 核心覆盖 |
|--------|----------|----------|
| `ascend-base` | 所有新用户 | 环境配置、设备管理、PyTorch 集成 |
| `ascend-ops` | **算子开发者** | **AscendC、op-plugin、Triton 迁移、优化** |
| `ascend-inference` | 模型推理 | ATC、vLLM、量化、Diffusers |
| `ascend-training` | 模型训练 | HCCL 通信、MindSpeed、VERL |
| `ascend-profiling` | 性能分析 | 数据采集、瓶颈定位、MFU 分析 |

### 安装方式

```bash
# 安装算子开发相关 skills
npx skills add https://github.com/ascend-ai-coding/awesome-ascend-skills -s '*'
```

### ascend-ops 包含的关键 Skills

| Skill 方向 | 覆盖内容 |
|------------|----------|
| 设计与架构 | 算子架构设计、tiling 策略、内存规划、设计文档生成 |
| 代码实现 | 从设计规格生成代码、框架注册（PyTorch）、多级算子组合 |
| 质量保证 | 精度对比评估、性能基准测试、代码审查 |
| 性能优化 | Tiling 参数调优、内存层次优化、计算利用率提升 |
| 调试诊断 | 运行时错误诊断、精度不匹配排查、死锁分析 |

---

## 学习资源

### 官方文档与教程

| 资源 | 说明 | 链接 |
|------|------|------|
| AscendC 官网 | 算子开发入口 | [hiascend.com/cann/ascend-c](https://www.hiascend.com/cann/ascend-c) |
| CANN Samples | 官方示例代码库 | [gitee.com/ascend/samples](https://gitee.com/ascend/samples) |
| MindSpore 教程 | MindSpore 集成指南 | [mindspore.cn](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.1/operation/op_custom_ascendc.html) |
| MindStudio 文档 | IDE 算子开发指南 | [hiascend.com/document](https://www.hiascend.com/document/detail/en/mindstudio/600/msug/msug_000108.html) |
| Triton-Ascend | Triton 编译器昇腾后端 | [github.com/triton-lang/triton-ascend](https://github.com/triton-lang/triton-ascend) |

### 社区资源

| 资源 | 说明 | 链接 |
|------|------|------|
| Ascend C 保姆级教程 | 入门第一份代码 | [知乎专栏](https://zhuanlan.zhihu.com/p/653497102) |
| CANN 架构深度解析 | 架构与算子开发入门 | [阿里云开发者社区](https://developer.aliyun.com/article/1643821) |
| Agent Skills 仓库 | AI 辅助开发 Skills | [GitHub](https://github.com/ascend-ai-coding/awesome-ascend-skills) |
| AscendOptimizer 论文 | 基于 Agent 的算子优化 | [arXiv](https://arxiv.org/html/2603.23566) |
| CANN 训练营笔记 | 芯片运算单元与 API | [CSDN](https://blog.csdn.net/weixin_54022960/article/details/133934582) |

### vllm-ascend 项目内参考

在本项目中，算子开发的实际案例和规范参见：

- **自定义算子开发指南**：`markdowns/custom_operator_development.md`
- **Triton 算子开发指南**：`markdowns/triton_operator_development.md`
- **Python 算子封装**：`vllm_ascend/ops/` 目录
- **C++/ACLNN 算子**：`csrc/` 目录
- **Triton 内核**：`vllm_ascend/ops/triton/` 目录

---

## 附：在 vllm-ascend 项目中开发算子的建议路径

1. **优先选择 Triton-Ascend**：如果算子逻辑可以用 Triton 表达，优先使用 Triton（参考 `triton_operator_development.md`），开发效率高
2. **需要极致性能时用 AscendC**：矩阵运算、Attention 等核心算子，使用 AscendC + ACLNN 路径
3. **Python wrapper 统一入口**：无论底层实现方式，都通过 `vllm_ascend/ops/` 提供统一的 Python 接口
4. **遵循项目规范**：参考 `AGENTS.md` 中的开发指南，特别是 NPU 特殊考量（避免 `tensor.item()` 热路径同步等）
