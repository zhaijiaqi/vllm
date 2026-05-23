# vLLM 三天学习计划

这份计划假设你已经读过 vLLM paper，并且理解原始 PagedAttention 的核心：

- 每个请求拥有自己的 logical block table。
- logical blocks 映射到非连续的 physical KV blocks。
- attention 通过 block table 读取 KV，而不是假设 KV cache 在物理内存中连续。
- vLLM 通过 scheduler、KV cache manager、allocator、copy-on-write、preemption、distributed execution，把 PagedAttention 变成完整的 serving engine。

接下来三天的目标是：把这个 paper 时代的理解升级到当前 vLLM 代码库，尤其是 V1 架构，并形成一张完整的项目地图。整个路线不要求你在本地跑大模型。

## 总体视角：从论文 vLLM 到当前 vLLM

建议把当前项目分成五层来读：

1. **服务入口层**
   - Python offline API：`vllm/entrypoints/llm.py`
   - OpenAI-compatible server 和 CLI：`vllm/entrypoints/openai/`、`vllm/entrypoints/cli/`
   - 核心问题：用户输入如何变成 vLLM 内部 request？

2. **V1 engine core**
   - core process、request lifecycle、scheduler、output processing：
     `vllm/v1/engine/`、`vllm/v1/request.py`
   - 核心问题：vLLM 如何持续推进多个并发请求？

3. **scheduler 与 KV cache**
   - scheduler：`vllm/v1/core/sched/`
   - KV cache manager 和 block pool：
     `vllm/v1/core/kv_cache_manager.py`、
     `vllm/v1/core/kv_cache_coordinator.py`、
     `vllm/v1/core/single_type_kv_cache_manager.py`、
     `vllm/v1/core/block_pool.py`
   - 核心问题：chunked prefill、prefix caching、preemption、KV allocation 如何共享同一套 token-budget 调度模型？

4. **worker、model runner 与 attention backend**
   - executor：`vllm/v1/executor/`
   - GPU worker 和 model runner：`vllm/v1/worker/gpu_worker.py`、
     `vllm/v1/worker/gpu_model_runner.py`、`vllm/v1/worker/gpu/`
   - attention 抽象：`vllm/v1/attention/`
   - 核心问题：scheduler 的结果如何变成真实的 model forward？

5. **现代特性层**
   - prefix caching、hybrid KV cache、speculative decoding、structured outputs、
     multimodal input、LoRA、quantization、MoE/expert parallelism、
     disaggregated prefill、metrics、compilation、plugin systems。
   - 核心问题：哪些特性属于核心架构，哪些特性是插件、后端或执行期扩展？

## 你需要补上的当前新特性地图

### 1. V1 Core Architecture

当前代码库已经以 V1 为中心，V0 已经 deprecated。V1 重新组织了旧 engine，重点简化并模块化这些组件：

- scheduler
- KV cache manager
- worker
- sampler
- API server
- process architecture

重点文档和代码：

- `docs/usage/v1_guide.md`
- `docs/design/arch_overview.md`
- `vllm/v1/engine/core.py`
- `vllm/v1/engine/async_llm.py`
- `vllm/v1/core/sched/scheduler.py`

理解升级：

> 不要再把 vLLM 理解成“prefill scheduler + decode scheduler”。V1 用统一的 scheduling budget 处理 prompt tokens 和 output tokens，让 chunked prefill、prefix caching、speculative decoding、decode 可以放进同一套调度模型里。

### 2. 默认启用的 Chunked Prefill

chunked prefill 现在是 V1 调度中的常规机制。长 prompt 可以被拆成多个 chunk，并和 decode 请求混合执行。

重点文档和代码：

- `docs/configuration/optimization.md`
- `vllm/v1/core/sched/scheduler.py`

要回答的问题：

- `max_num_batched_tokens` 如何影响 TTFT 和 ITL？
- 为什么 decode 请求通常会被优先处理？
- 一个长 prefill 在什么时候会被拆成多次调度？

### 3. 作为 KV Manager 特性的 Prefix Caching

automatic prefix caching 已经深度集成进 KV cache manager。V1 使用 hash-based cache key，hash 内容包括 block tokens、parent hash，以及 LoRA ID、multimodal hash、cache salt 等 extra hashes。

重点文档和代码：

- `docs/features/automatic_prefix_caching.md`
- `docs/design/prefix_caching.md`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/kv_cache_utils.py`
- `benchmarks/benchmark_prefix_caching.py`
- `benchmarks/benchmark_prefix_block_hash.py`

要回答的问题：

- 为什么 vLLM 只缓存 full blocks？
- free queue 如何同时承担 eviction 结构的职责？
- multimodal placeholders 或 LoRA adapters 进入 prompt 时，cache hit 的含义如何变化？
- cache salt 为什么对多租户隐私重要？

### 4. Hybrid KV Cache Manager

现代模型可能混合 full attention、sliding-window attention、local attention、Mamba-like state-space layers，或者 KV-sharing layers。KV cache 管理不再是“所有层共享同一种 block layout”。

重点文档和代码：

- `docs/design/hybrid_kv_cache_manager.md`
- `vllm/v1/core/kv_cache_coordinator.py`
- `vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/v1/kv_cache_interface.py`

要回答的问题：

- 什么是 KV cache group？
- 为什么不同 group 需要对齐 page size？
- full attention 和 sliding-window attention 的 prefix caching 有什么不同？
- 哪些部分是稳定设计，哪些仍在演进？

### 5. Speculative Decoding

vLLM 现在支持多种 speculative decoding 方法：EAGLE、MTP、draft model、parallel draft model、MLP speculator、n-gram、suffix decoding，以及 custom proposer。

重点文档和代码：

- `docs/features/speculative_decoding/README.md`
- `docs/features/speculative_decoding/eagle.md`
- `docs/features/speculative_decoding/mtp.md`
- `vllm/v1/spec_decode/`
- `vllm/v1/sample/rejection_sampler.py`

要回答的问题：

- 哪些方法需要额外模型，哪些不需要？
- 为什么 speculation 对 memory-bound decode workload 更有价值？
- proposed tokens 如何被 accept 或 reject？
- speculative decoding 如何影响 scheduling 和 KV cache 增长？

### 6. Disaggregated Prefill 与 KV Transfer

disaggregated prefill 把 prefill 和 decode 放到不同 vLLM 实例中，然后在实例之间传输 KV cache。它主要解决 latency 控制和基础设施灵活性，不是直接提升总吞吐。

重点文档和代码：

- `docs/features/disagg_prefill.md`
- `docs/features/nixl_connector_usage.md`
- `docs/features/nixl_connector_compatibility.md`
- `vllm/distributed/kv_transfer/`
- `benchmarks/disagg_benchmarks/`

要回答的问题：

- 为什么要把 TTFT tuning 和 ITL tuning 分开？
- producer、consumer、connector、lookup buffer、pipe 分别是什么？
- layer-by-layer KV transfer 如何接入 attention execution？

### 7. Structured Outputs、Tool Calling 与 Reasoning Outputs

serving 已经不只是“生成自由文本”。vLLM 支持 structured output backends 和 OpenAI-compatible 的高级 API 能力。

重点文档和代码：

- `docs/features/structured_outputs.md`
- `docs/features/tool_calling.md`
- `docs/features/reasoning_outputs.md`
- `docs/features/interleaved_thinking.md`
- `vllm/v1/structured_output/`
- `vllm/v1/worker/gpu/structured_outputs.py`

要回答的问题：

- constraints 在哪里被编译或表示？
- constrained decoding 如何接入 sampler？
- 哪些是 request-time 参数，哪些是 server startup config？

### 8. Multimodal、Pooling 与非经典 Decoder Workloads

vLLM 支持 multimodal inputs、pooling models、reward/classification/embed models，以及部分 Mamba/hybrid models。

重点文档和代码：

- `docs/features/multimodal_inputs.md`
- `docs/models/pooling_models/README.md`
- `docs/models/supported_models.md`
- `vllm/multimodal/`
- `vllm/model_executor/models/`
- `vllm/v1/pool/`

要回答的问题：

- image、video、audio 如何变成 placeholders 和 embeddings？
- multimodal hashes 如何被 prefix caching 使用？
- pooling output 和 token generation output 的生命周期有什么差异？

### 9. Quantization、MoE 与 Parallelism

当前 vLLM 也是一个 hardware-aware execution framework。quantization、expert parallelism、tensor parallelism、pipeline parallelism、data parallelism 都是项目理解的一部分。

重点文档和代码：

- `docs/features/quantization/README.md`
- `docs/configuration/optimization.md`
- `docs/design/fused_moe_modular_kernel.md`
- `docs/design/moe_kernel_features.md`
- `vllm/model_executor/layers/quantization/`
- `vllm/model_executor/layers/fused_moe/`
- `vllm/v1/engine/coordinator.py`
- `vllm/v1/worker/gpu/dp_utils.py`

要回答的问题：

- 哪种 parallelism 解决模型放不下的问题，哪种解决吞吐扩展问题？
- 为什么 MoE layers 特殊？
- 哪些 quantization 影响 weights、activations 或 KV cache？

### 10. Compilation、CUDA Graphs、Metrics 与 Plugins

这些不是前三天最优先的主线，但它们解释了为什么生产环境中的 vLLM 不只是一个 engine loop。

重点文档和代码：

- `docs/design/torch_compile.md`
- `docs/design/cuda_graphs.md`
- `docs/design/model_runner_v2.md`
- `docs/design/metrics.md`
- `docs/design/plugin_system.md`
- `docs/design/io_processor_plugins.md`
- `vllm/compilation/`
- `vllm/v1/metrics/`

要回答的问题：

- 哪些执行路径会被 capture 或 compile？
- 哪些 runtime metrics 能暴露 scheduler 和 KV 压力？
- 哪些 extension points 可以避免直接改 core vLLM？

## 三天安排

### Day 1：建立当前 V1 骨架

目标：把 paper 时代的“engine + block manager”图，升级成当前 V1 的 process architecture 和 request lifecycle 图。

#### 上午：V1 架构地图

阅读：

- `docs/usage/v1_guide.md`
- `docs/design/arch_overview.md`

快速扫代码：

- `vllm/entrypoints/llm.py`
- `vllm/entrypoints/openai/api_server.py`
- `vllm/v1/engine/async_llm.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/engine/core_client.py`

产出：

- 画一张一页的进程图：
  `API server -> EngineCore -> Executor -> Worker -> ModelRunner`。
- 写一段话解释为什么 DP 会产生多个 engine cores。

检查问题：

- 哪个 process 负责 scheduling？
- 哪个 process 负责 HTTP streaming？
- 哪个 process 持有 GPU memory 和 model weights？
- offline `LLM` 和 online serving 的路径有什么不同？

#### 下午：Request Lifecycle

阅读：

- `vllm/v1/request.py`
- `vllm/v1/engine/input_processor.py`
- `vllm/v1/engine/output_processor.py`
- `vllm/v1/engine/logprobs.py`
- `vllm/outputs.py`

快速扫测试：

- `tests/basic_correctness/test_basic_correctness.py`
- `tests/engine/`

产出：

- 写一条 request state timeline：
  `raw prompt -> tokenized/processed input -> waiting/running request -> model output -> sampler -> detokenized output`。

检查问题：

- prompt tokens 存在哪里？
- generated token IDs 在哪里变成文本？
- logprobs 在哪里处理？
- scheduler、sampler、output processor 分别需要 request 的哪些字段？

#### 晚上：Scheduler 第一遍

阅读：

- `docs/configuration/optimization.md` 中 preemption 和 chunked prefill 相关部分
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/core/sched/request_queue.py`
- `vllm/v1/core/sched/output.py`

产出：

- 整理一份 scheduler vocabulary：
  waiting、running、scheduled tokens、token budget、preemption、recompute、
  priority、FCFS、chunked prefill。

检查问题：

- 被 schedule 的到底是 requests、sequences，还是 token counts？
- V1 如何避免严格区分 prefill/decode 两个调度器？
- 为什么 recomputation 有时比 swapping 更简单或更便宜？

## Day 2：掌握 KV、Attention 与执行路径

目标：把你对 PagedAttention 的理解连接到当前 KV cache manager、hybrid cache support 和 attention backend execution。

#### 上午：KV Cache Manager 与 Prefix Caching

阅读：

- `docs/design/prefix_caching.md`
- `docs/features/automatic_prefix_caching.md`

代码：

- `vllm/v1/core/block_pool.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/kv_cache_utils.py`

快速扫：

- `benchmarks/benchmark_prefix_caching.py`
- `benchmarks/benchmark_prefix_block_hash.py`

产出：

- 画一张图：
  `BlockPool + free queue + cache hash map + request block table`。
- 手写一个 prompt 部分命中 prefix cache 的例子。

检查问题：

- 为什么只缓存 full blocks？
- LRU eviction 如何通过 free queue 发生？
- 除 token IDs 外，哪些 extra values 会进入 block hash？
- 从 paper 中偏 COW 的心智模型，到 V1 的 cache manager 和 append-oriented block table 行为，发生了什么变化？

#### 下午：Hybrid KV Cache

阅读：

- `docs/design/hybrid_kv_cache_manager.md`

代码：

- `vllm/v1/kv_cache_interface.py`
- `vllm/v1/core/kv_cache_coordinator.py`
- `vllm/v1/core/single_type_kv_cache_manager.py`

产出：

- 做一张对比表：
  full attention vs sliding window vs Mamba/hybrid state。

检查问题：

- 为什么 sliding-window attention 会让 prefix caching 变复杂？
- KV cache group 的目的是什么？
- 为什么 page-size alignment 是必要的？
- 当不同模型层的 cache/state shape 不同时，vLLM 如何处理？

#### 傍晚：Attention Backends

阅读：

- `docs/design/attention_backends.md`
- `docs/design/paged_attention.md`

代码：

- `vllm/v1/attention/backend.py`
- `vllm/v1/attention/selector.py`
- `vllm/v1/attention/backends/registry.py`
- `vllm/v1/attention/ops/paged_attn.py`
- `benchmarks/kernels/benchmark_paged_attention.py`
- `benchmarks/attention_benchmarks/README.md`

产出：

- 画一张 backend map：
  backend selection -> metadata -> kernel/op -> KV block access。

检查问题：

- 什么是 backend-independent metadata？
- 哪些 backends 面向 CUDA、ROCm、CPU、Triton？
- block table 在哪里进入真正的 attention op？

#### 晚上：Worker 与 Model Runner

阅读：

- `vllm/v1/executor/abstract.py`
- `vllm/v1/executor/uniproc_executor.py`
- `vllm/v1/executor/multiproc_executor.py`
- `vllm/v1/worker/gpu_worker.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/gpu/input_batch.py`

产出：

- 画一张 step diagram：
  `SchedulerOutput -> Executor.execute_model -> Worker -> ModelRunner -> Model forward -> Sampler`。

检查问题：

- 哪个对象负责准备 model input tensors？
- CUDA graphs 或 compilation hooks 在哪里接入？
- LoRA、speculative decode、structured outputs 分别在哪里接入 GPU execution path？

## Day 3：梳理现代特性并整合全局

目标：不死记每个模型和 backend，而是按架构位置分类每个特性。

#### 上午：Serving Features

阅读：

- `docs/features/structured_outputs.md`
- `docs/features/tool_calling.md`
- `docs/features/reasoning_outputs.md`
- `docs/features/interleaved_thinking.md`

代码：

- `vllm/v1/structured_output/`
- `vllm/v1/sample/`
- `vllm/entrypoints/openai/`

产出：

- 做一张表，列：
  feature、user API、internal component、performance risk、correctness risk。

检查问题：

- structured outputs 如何约束 sampling？
- 哪些逻辑发生在 API server，哪些在 engine，哪些在 sampler？
- 哪些特性属于 OpenAI-compatible API surface，而不是 core engine mechanics？

#### 中午前：Speculative Decoding

阅读：

- `docs/features/speculative_decoding/README.md`
- `docs/features/speculative_decoding/eagle.md`
- `docs/features/speculative_decoding/mtp.md`
- `docs/features/speculative_decoding/n_gram.md`
- `docs/features/speculative_decoding/suffix.md`

代码：

- `vllm/v1/spec_decode/`
- `vllm/v1/sample/rejection_sampler.py`
- `benchmarks/benchmark_ngram_proposer.py`

产出：

- 做一张 method matrix：
  draft model、EAGLE、MTP、n-gram、suffix、custom proposer。

检查问题：

- 哪些方法需要额外 weights？
- 哪些方法最容易启用？
- acceptance/rejection 如何保持输出分布正确？
- speculation 如何改变 KV allocation 压力？

#### 下午：Infrastructure 与 Scale-Out Features

阅读：

- `docs/features/disagg_prefill.md`
- `docs/features/nixl_connector_usage.md`
- `docs/configuration/optimization.md` 中 parallelism 相关部分
- `docs/design/p2p_nccl_connector.md`
- `docs/design/nixl_kv_cache_lease.md`

代码：

- `vllm/distributed/kv_transfer/`
- `vllm/v1/engine/coordinator.py`
- `vllm/v1/worker/gpu/dp_utils.py`
- `benchmarks/disagg_benchmarks/`

产出：

- 画一张 deployment map：
  single engine、TP、PP、DP、EP、disaggregated prefill。

检查问题：

- 哪种 parallelism 用于放下模型？
- 哪种 parallelism 用于扩展吞吐？
- disaggregated prefill 为什么能分别调 TTFT 和 ITL？
- DP coordinator 具体协调什么？

#### 傍晚：Model Breadth 与 Hardware Features

阅读：

- `docs/features/multimodal_inputs.md`
- `docs/models/supported_models.md`
- `docs/models/pooling_models/README.md`
- `docs/features/quantization/README.md`
- `docs/design/fused_moe_modular_kernel.md`

代码：

- `vllm/multimodal/`
- `vllm/v1/pool/`
- `vllm/model_executor/models/`
- `vllm/model_executor/layers/quantization/`
- `vllm/model_executor/layers/fused_moe/`

产出：

- 做一张 feature classification：
  model support、input processing、execution kernel、memory optimization、
  serving API。

检查问题：

- 新增一个 model architecture 需要改哪些层？
- 新增一个 quantization method 需要接入哪里？
- 为什么 MoE models 需要特殊 kernel 和 parallelism？
- multimodal embeddings 在哪里合并进 language-model tokens？

#### 晚上：最终整合

给自己写一份最终笔记，包含这些部分：

1. **Request lifecycle**
   - 从 API input 到 final output。

2. **Scheduling lifecycle**
   - waiting queue、running requests、token budget、chunked prefill、
     preemption/recompute。

3. **KV lifecycle**
   - block pool、allocation、prefix cache hit、eviction、free、hybrid groups。

4. **Execution lifecycle**
   - executor、worker、model runner、attention backend、sampler。

5. **Feature integration map**
   - prefix caching、speculative decoding、structured outputs、LoRA、
     multimodal、quantization、MoE、disaggregated prefill。

最终自测：

- 你能解释为什么 V1 deprecated GPU-CPU KV swapping，但仍然使用 recompute preemption 吗？
- 你能解释 chunked prefill 和 speculative decoding 如何同时放进 scheduler 吗？
- 你能解释 prefix caching 如何改变 block allocation，但不改变模型正确性吗？
- 你能解释 disaggregated prefill 如何改变 deployment topology，但不改变 KV cache 的核心含义吗？
- 你能追踪一个 request 经过 API server、engine core、scheduler、KV cache manager、worker、attention backend、sampler、output processor 的完整路径吗？

## 推荐的只读检索命令

这些命令只帮助你导航代码，不需要跑模型：

```bash
rg "class .*Scheduler|def schedule" vllm/v1
rg "KVCacheManager|BlockPool|SingleTypeKVCacheManager" vllm/v1
rg "prefix caching|computed_blocks|block_hash" vllm/v1 docs
rg "speculative|Speculative|proposer|rejection" vllm/v1 docs/features
rg "structured_output|StructuredOutputs" vllm/v1 vllm/entrypoints docs
rg "kv_transfer|Connector|LookupBuffer|Nixl" vllm docs examples benchmarks
rg "enable_expert_parallel|data_parallel|tensor_parallel|pipeline_parallel" vllm/v1 vllm/config
```

## 前三天建议先跳过的内容

除非它们和你当前目标直接相关，否则先跳过：

- `vllm/model_executor/models/` 下的大多数单模型实现，只挑一两个代表模型看即可。
- 大部分低层 CUDA、ROCm、Triton、CUTLASS kernel。
- 每个 quantization backend 的完整实现。
- 完整 Kubernetes 和部署文档。
- 每个 benchmark 变体。

前三天的目标不是掌握所有细节，而是让整个项目变得“可导航”。kernel 和 backend 细节可以留到下一轮深入。

## 可选 Day 4 主题

如果三天后继续深入，推荐这些方向：

- 写一个只读 trace，观察 request scheduling 的关键状态。
- 完整读一条 attention backend 路径。
- 完整读一个 MoE model 加 fused MoE kernel 路径。
- 完整读一个 multimodal model 和它的 input processor。
- 从 scheduler/KV-cache 视角对比 vLLM、SGLang、TensorRT-LLM、Hugging Face TGI。
