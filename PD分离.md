PD分离

# PD 分离（Prefill-Decode Disaggregation）原理与实现详解

## 一、PD 分离是什么？

PD 分离（Prefill-Decode Disaggregation）是大语言模型推理服务中的一种系统架构优化技术。

它将原本运行在同一组 GPU/XPU 上的 Prefill 和 Decode 两个计算阶段，拆分到不同的计算实例上执行。

主要目标：

1. 解决 Prefill 和 Decode 之间的资源竞争。
2. 降低 Prefill 对 Decode 延迟的干扰。
3. 根据不同阶段的计算特征分配计算资源。
4. 支持 Prefill 和 Decode 独立扩容。
5. 在满足延迟 SLO 的条件下提升整体系统吞吐量。

PD 分离的核心思想是：

> Prefill 负责输入 Prompt 的并行计算，Decode 负责逐 Token 自回归生成。由于二者计算特征不同，将其部署在不同的计算实例上，可以分别优化资源利用率和调度策略。
> PD 分离不会改变 Transformer 的数学计算原理，主要改变的是计算资源分配、调度方式以及 KV Cache 的存储和传输方式。

---

## 二、Prefill 和 Decode 的本质区别

### 2.1 Prefill 阶段

Prefill 阶段负责处理用户输入的 Prompt。

假设输入序列为：

$$
X=[x_1,x_2,\ldots,x_S]
$$

其中 $S$ 表示输入 Token 数量。

在标准 Transformer Attention 中：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

随后计算：

$$
Attention(Q,K,V)
=
Softmax\left(
\frac{QK^T}{\sqrt{d}}+M
\right)V
$$

其中 $M$ 为因果注意力掩码。

Prefill 阶段会处理整个输入序列，并生成后续 Decode 所需要的 KV Cache。

例如：

```
Input Prompt:
[x1, x2, x3, x4, x5, x6]

          |
          v

    Prefill Forward
          |
          v

    Hidden States
          |
          v

    KV Cache
```

Prefill 的主要特点：

* 单次 Forward 处理多个 Token。
* GEMM 通常具有较大的矩阵维度。
* 能够利用矩阵计算单元的并行计算能力。
* 通常属于 Compute-intensive 阶段。
* 主要关注输入 Token 吞吐与 TTFT。
* 输入长度越长，需要计算的历史上下文通常越多。

需要注意，Prefill 不一定始终是 Compute-bound。小 Batch、短 Prompt、极长上下文和特殊算子也可能导致访存、调度或通信成为瓶颈。

### 2.2 Decode 阶段

Decode 阶段负责自回归生成输出 Token。

对于已经完成 Prefill 的请求：

```
Prefill:
[x1, x2, x3, x4] -> KV Cache

Decode Step 1:
[y1] + KV Cache -> y2

Decode Step 2:
[y2] + KV Cache -> y3

Decode Step 3:
[y3] + KV Cache -> y4
```

每一步 Decode 主要执行：

1. 对当前 Token 计算 Q、K、V。
2. 读取历史 KV Cache。
3. 执行 Attention。
4. 将当前 Token 的 K、V 写入 KV Cache。
5. 计算下一个 Token 的 Logits。
6. 执行 Sampling，获得下一个 Token。

与 Prefill 不同，普通 Decode 每次通常仅处理每个请求的一个新 Token。

因此，在较小 Batch Size 下：

* GEMM 规模较小。
* 计算并行度相对较低。
* 模型权重读取成本较高。
* 历史 KV Cache 的读取开销持续增加。
* 通常容易受到内存带宽限制。

Decode 在这种情况下通常表现为 Memory-bound。

不过随着 Batch Size 增加，或者使用 MoE、投机解码等技术，Decode 的计算瓶颈也可能发生变化。

### 2.3 Prefill 和 Decode 对比


| 维度           | Prefill         | Decode              |
| :------------- | :-------------- | :------------------ |
| 主要任务       | 处理输入 Prompt | 自回归生成输出      |
| 单次处理 Token | 多个            | 通常每请求 1 个     |
| 典型计算特征   | 大矩阵 GEMM     | 较小矩阵 GEMM       |
| 典型性能瓶颈   | Compute-bound   | Memory-bound        |
| 计算并行度     | 较高            | 较低，依赖 Batch    |
| KV Cache 操作  | 大规模生成      | 读取历史 KV 并追加  |
| 主要性能指标   | TTFT、输入吞吐  | TPOT、ITL、输出吞吐 |
| 资源关注点     | 算力、显存      | 带宽、KV Cache 容量 |

核心结论：

**Prefill 和 Decode 的计算特征不同，对硬件资源的需求也不同。**

---

## 三、为什么需要 PD 分离？

### 3.1 传统 PD 混部架构

传统推理架构中，Prefill 和 Decode 通常由同一个推理实例执行。

```
                 User Requests
                       |
                       v
                     Router
                       |
                       v
              Inference Scheduler
                       |
              +--------+--------+
              |                 |
              v                 v
           Prefill            Decode
              |                 |
              +--------+--------+
                       |
                       v
              Same GPU/XPU Group
                       |
                       v
                 KV Cache Pool
```

这种模式中：

* Prefill 和 Decode 共享计算设备。
* 两个阶段共享设备计算资源。
* 两个阶段竞争显存和执行时间。
* Scheduler 需要在 Prefill 和 Decode 之间进行调度。

假设已有三个请求正在 Decode：

```
Request A: Decode
Request B: Decode
Request C: Decode
```

此时新请求 D 到达：

```
Request D: Prefill
```

如果请求 D 的输入长度很大，Prefill Forward 可能占用较长的执行时间。

因此，正在执行 Decode 的请求可能需要等待。

### 3.2 Prefill 对 Decode 的干扰

假设每个 Decode Step 耗时 10 ms。

正常执行：

```
Time -->

| Decode | Decode | Decode | Decode |
   10ms     10ms     10ms     10ms
```

如果中间插入一个耗时 80 ms 的 Prefill：

```
Time -->

| Decode |     Prefill     | Decode | Decode |
   10ms          80ms           10ms     10ms
```

原本需要每隔约 10 ms 输出一个 Token 的请求，可能因为 Prefill 的插入产生明显的 Token 间隔。

这就是 Prefill-Decode Interference。

主要影响：

* TPOT（Time Per Output Token）
* ITL（Inter-Token Latency）
* P95/P99 Token Latency
* 流式输出的稳定性

需要注意，Continuous Batching、Chunked Prefill 等技术可以缓解这种干扰，但不能完全消除同一计算实例上的资源竞争。

### 3.3 PD 分离架构

PD 分离将 Prefill 和 Decode 拆分成独立的 Worker。

```
                  User Request
                       |
                       v
                  PD Router
                       |
                       v
                Prefill Worker
                       |
                       v
                Prompt Forward
                       |
                       v
                Generate KV Cache
                       |
                       v
                KV Cache Transfer
                 (RDMA / NIXL)
                       |
                       v
                 Decode Worker
                       |
                       v
                 Restore KV Cache
                       |
                       v
                 Token Generation
                       |
                       v
                    Response
```

其中：

**Prefill Worker**

负责：

* 接收输入 Prompt。
* 执行 Prefill Forward。
* 生成各层 KV Cache。
* 将 KV Cache 传输给 Decode Worker。

**Decode Worker**

负责：

* 接收 Prefill 的 KV Cache。
* 恢复请求执行上下文。
* 逐 Token 生成。
* 持续更新 KV Cache。
* 返回模型输出。

在这种架构下，Prefill 和 Decode 不再争用同一计算实例的执行时间。

---

## 四、PD 分离最核心的问题：KV Cache Transfer

### 4.1 为什么需要传输 KV Cache？

在传统 PD 混部模式下：

```
GPU/XPU
   |
   +--- Prefill
   |       |
   |       v
   |    KV Cache
   |       |
   +--- Decode
           |
           v
       Read KV Cache
```

Prefill 和 Decode 使用同一实例管理的 KV Cache。

但是在 PD 分离模式下：

```
Prefill XPU                    Decode XPU
     |                              |
     v                              |
Prompt Forward                     |
     |                              |
     v                              |
Generate KV Cache                  |
     |                              |
     +--------- Transfer ---------->|
                                    |
                                    v
                              Restore KV Cache
                                    |
                                    v
                                  Decode
```

Prefill 和 Decode 运行在不同设备或不同实例中。

因此，Decode 无法直接假设可以访问 Prefill 的 KV Cache 内存。

必须通过通信机制进行数据传输。

### 4.2 完整请求执行流程

假设：

```
Input Tokens  = 4096
Output Tokens = 1000
```

执行流程如下。

**Step 1：Router 选择实例**

Router 根据负载、KV Cache 容量以及调度策略，为请求选择 Prefill 和 Decode Worker。

**Step 2：分配 KV Cache 空间**

Decode Worker 准备目标 KV Cache 存储空间。

通常涉及：

```
Request ID
KV Block Allocation
Block Index Mapping
Token Position
Cache Metadata
```

具体分配时机由实现决定。

**Step 3：Prefill Forward**

Prefill Worker 处理 4096 个输入 Token：

```
Prompt
   |
   v
Embedding
   |
   v
Transformer Layers
   |
   v
Generate KV Cache
```

同时计算下一 Token 的预测分布或相关状态。

**Step 4：KV Cache Transfer**

通过支持的传输后端：

```
Prefill KV Blocks
        |
        v
  Transfer Engine
        |
        v
Decode KV Blocks
```

将历史 KV Cache 数据传输到 Decode Worker。

**Step 5：Decode 恢复执行**

当必要的 KV 数据以及请求状态准备完成后：

```
Restore Request State
        |
        v
Load KV Cache Mapping
        |
        v
Continue Decode
```

Decode Worker 开始后续 Token 生成。

**Step 6：资源回收**

请求完成后：

* 释放或缓存 KV Blocks。
* 更新请求状态。
* 回收必要的传输资源。
* 将计算资源分配给后续请求。

### 4.3 传输的是模型权重吗？

不是。

标准 PD 分离通常需要 Prefill 和 Decode 两侧各自加载对应的模型权重。

逐请求传输的主要是：


| 数据                   | 是否通常传输 |
| :--------------------- | :----------- |
| 模型权重               | 否           |
| Prompt KV Cache        | 是           |
| Request ID             | 需要协调     |
| Token ID / Position    | 需要协调     |
| KV Block Mapping       | 需要协调     |
| 采样及生成状态         | 按实现需要   |
| 每层完整 Hidden States | 通常不需要   |

PD 分离最重要的是让 Decode 可以直接复用 Prefill 的计算结果，而不必重新计算整个 Prompt。

---

## 五、KV Cache 传输量如何计算？

### 5.1 KV Cache 大小公式

对于普通 MHA/GQA 模型，假设采用标准 KV Cache 布局：

$$
\boxed{
M_{KV}
=
2\times L\times S\times H_{KV}\times D_h\times B
}
$$

其中：

* $L$：Transformer 层数。
* $S$：输入 Token 数量。
* $H_{KV}$：KV Head 数。
* $D_h$：Head Dimension。
* $B$：每个元素的字节数。
* 2：分别对应 K 和 V。

例如，假设：

```
Layers       = 32
Prompt       = 8192 tokens
KV Heads     = 8
Head Dim     = 128
KV Dtype     = BF16
Bytes        = 2
```

则：

$$
M_{KV}
=
2\times32\times8192\times8\times128\times2
$$

计算得到：

$$
\boxed{
M_{KV}=1\ \text{GiB}
}
$$

也就是说，一个 8192 Token 的请求可能需要传输约 1 GiB 的 KV Cache。

这里没有计入额外对齐、元数据、传输缓冲区等开销。

### 5.2 不同 Prompt 长度的 KV Cache 大小

使用上面的模型配置：


| Prompt Length | KV Cache |
| :------------ | -------: |
| 1024          |  128 MiB |
| 4096          |  512 MiB |
| 8192          |    1 GiB |
| 16384         |    2 GiB |
| 32768         |    4 GiB |
| 65536         |    8 GiB |

可见，KV Cache 的大小与序列长度近似呈线性关系。

因此，长上下文请求可能对 PD 分离的网络带宽提出很高要求。

### 5.3 KV Cache 传输时间

最简单的传输时间公式：

$$
\boxed{
T_{\text{transfer}}
\approx
\frac{M_{KV}}{BW_{\text{effective}}}
}
$$

其中：

* $M_{KV}$：需要传输的数据量。
* $BW_{\text{effective}}$：实际有效传输带宽。

例如：

```
KV Cache Size = 1 GiB
Network       = 200 Gbps
```

忽略额外开销时：

$$
T_{\text{transfer}}
\approx
\frac{1\text{ GiB}\times8}{200\text{ Gbps}}
$$

得到：

$$
T_{\text{transfer}}\approx43\text{ ms}
$$

注意这是理想情况下的理论传输时间。

实际传输还受到：

* 网络协议开销。
* RDMA 实际有效带宽。
* 网络拥塞。
* KV Cache 数据布局。
* 数据拷贝和重排。
* 多请求竞争。
* 设备内存带宽。

等因素影响。

### 5.4 RDMA 的作用

RDMA（Remote Direct Memory Access）允许设备通过支持的网络和内存注册机制进行高效的数据移动，减少传统 TCP/IP 数据传输中不必要的 CPU 参与及内存复制开销。

在 PD 分离中，常见传输方案包括：

* RDMA。
* NIXL。
* Mooncake Transfer Engine。
* 特定设备的高速互联。
* TCP 或其他传输后端。

具体支持哪些后端取决于推理框架和硬件平台。

### 5.5 MLA 模型的特殊情况

对于 DeepSeek 等采用 MLA（Multi-head Latent Attention）的模型，KV Cache 不一定采用普通 MHA/GQA 的完整 K、V Head 布局。

MLA 通常缓存压缩后的 KV Latent 表示以及相应的位置编码状态。

因此 KV Cache 大小需要根据实际模型结构与缓存布局计算。

但 PD 分离的基本原理不变：

```
Prefill Generate Attention State
             |
             v
       Transfer State
             |
             v
       Decode Reuse State
```

---

## 六、PD 分离为什么能够提升性能？

### 6.1 跨请求流水线并行

假设有三个请求：

```
Request A
Request B
Request C
```

在 PD 分离架构下，可以形成如下执行流程：

```
Time ------------------------------------------------>

Prefill Worker:

| Prefill A | Prefill B | Prefill C | ...

Decode Worker:

            | Decode A ------------------------->
                        | Decode A+B ------------>
                                    | Decode A+B+C ->
```

这里表达的是不同请求之间的执行重叠。

当 Prefill Worker 正在处理请求 B 时，Decode Worker 可以继续处理请求 A。

需要注意：

**同一个请求的 Decode 必须依赖已经完成并传输就绪的必要 Prefill 状态。**

PD 分离主要实现跨请求流水线，而不是让同一个请求的 Prefill 和 Decode 无条件并行。

### 6.2 资源专用化

Prefill 和 Decode 的计算特征不同，因此可以使用不同的资源配置。

例如：


| 资源       | Prefill Worker        | Decode Worker              |
| :--------- | :-------------------- | :------------------------- |
| 优化目标   | 高输入吞吐            | 低 TPOT、高输出吞吐        |
| 主要计算   | 大 GEMM               | Decode GEMM + KV Attention |
| Batch 策略 | Prompt Token Batching | Continuous Batching        |
| KV Cache   | 生成并导出            | 接收、读取、追加           |
| 扩容方式   | 增加 P 实例           | 增加 D 实例                |
| 主要瓶颈   | 算力/计算效率         | 带宽/Cache/通信            |

通过独立优化，可以降低两种计算负载互相影响的程度。

### 6.3 P:D 比例如何计算？

假设：

* 请求到达速率为 $\lambda$，单位 requests/s。
* 平均输入长度为 $I$，单位 tokens/request。
* 平均输出长度为 $O$，单位 tokens/request。
* 单个 Prefill 实例的有效处理能力为 $C_P$，单位 input tokens/s。
* 单个 Decode 实例的有效处理能力为 $C_D$，单位 output tokens/s。

那么理想化的资源需求：

$$
N_P\geq\frac{\lambda I}{C_P}
$$

$$
N_D\geq\frac{\lambda O}{C_D}
$$

因此：

$$
\boxed{
N_P:N_D
\approx
\frac{I}{C_P}:\frac{O}{C_D}
}
$$

举个例子：

```
QPS                  = 10
Average Input Length = 4000
Average Output Length= 1000

Prefill Capacity     = 20000 input tokens/s/instance
Decode Capacity      = 5000 output tokens/s/instance
```

则：

$$
N_P\geq\frac{10\times4000}{20000}=2
$$

$$
N_D\geq\frac{10\times1000}{5000}=2
$$

理想化结果：

$$
P:D=2:2
$$

也就是 1:1。

但这个公式只是容量估算。

实际部署还需要考虑：

* TTFT 和 TPOT SLO。
* Batch Size。
* 输入输出长度分布。
* Prefix Cache 命中率。
* KV Transfer 带宽。
* 节点利用率。
* 网络拓扑。
* TP/EP 并行配置。

因此，最优 P:D 比例通常需要通过真实业务压测确定。

---

## 七、PD 分离与 TP、EP、DP 的关系

PD 分离与 TP、EP、DP 是不同维度的优化方式。

### 7.1 各种并行方式的区别


| 技术 | 全称                          | 主要作用                     |
| :--- | :---------------------------- | :--------------------------- |
| PD   | Prefill-Decode Disaggregation | 按执行阶段拆分实例           |
| TP   | Tensor Parallelism            | 拆分张量计算                 |
| EP   | Expert Parallelism            | 将 MoE Expert 分布到多个设备 |
| DP   | Data Parallelism              | 通过并行副本处理请求或数据   |

PD 分离决定哪个实例执行 Prefill，哪个实例执行 Decode。

TP/EP/DP 决定每个实例内部如何组织模型计算。

### 7.2 PD + TP + EP

例如：

```
                    Router
                       |
              +--------+--------+
              |                 |
              v                 v
       Prefill Cluster    Decode Cluster
              |                 |
            TP=8              TP=8
            EP=8              EP=8
              |                 |
              +---- KV Cache ---+
                    Transfer
```

此时：

* Prefill 使用 TP8/EP8。
* Decode 使用 TP8/EP8。
* 两边独立进行模型计算。
* KV Cache 通过通信后端传输。

### 7.3 Prefill 和 Decode 的 TP 必须相同吗？

不一定。

例如：

```
Prefill: TP=4

Decode:  TP=8
```

但由于 Attention TP 的分片方式可能不同，KV Cache 的物理存储布局也可能不同。

这时需要执行 KV Cache Reshard。

简化过程：

```
Prefill TP4
     |
     v
KV Cache Shards
     |
     v
Gather / Repack
     |
     v
Transfer
     |
     v
Scatter / Reshard
     |
     v
Decode TP8
```

其中：

* Gather：聚合需要传输的 KV 数据。
* Repack：转换布局。
* Transfer：跨实例传输。
* Scatter：按 Decode TP 布局分发。

实际是否支持异构 TP，需要检查具体框架的 KV Transfer 实现。

### 7.4 PD 分离与 MoE EP

对于 MoE 模型：

Prefill 和 Decode 可能具有不同的 Expert 负载特征。

因此，需要分别考虑：

* Expert Routing。
* Expert Load Balancing。
* All-to-All 通信。
* EP 组划分。
* Expert GEMM 效率。
* TP/EP 组合。
* KV Cache 的 Attention 分片布局。

特别要注意：

**KV Cache 的传输布局主要与 Attention 结构及其并行切分方式相关，而不是简单由 EP 数量决定。**

EP 主要决定 MoE Expert 的计算和通信布局。

因此，对于 MoE + PD 分离，Attention KV Transfer 与 MoE Expert All-to-All 是两类不同的通信开销。

---

## 八、SGLang 中的 PD 分离实现

### 8.1 整体架构

SGLang 中的 PD 分离可以概括为：

```
                  Client
                     |
                     v
                  Router
                     |
          +----------+----------+
          |                     |
          v                     v
    Prefill Server        Decode Server
          |                     |
          v                     |
    Prefill Scheduler           |
          |                     |
          v                     |
      Model Forward             |
          |                     |
          v                     |
      Generate KV               |
          |                     |
          v                     |
      KV Transfer --------------+
                                |
                                v
                         Decode Scheduler
                                |
                                v
                         Restore Request
                                |
                                v
                         Decode Forward
                                |
                                v
                         Token Generation
```

核心模块包括：

1. Router：选择 P/D Worker。
2. Scheduler：管理各阶段的请求调度。
3. KV Cache Allocator：分配和回收 KV Blocks。
4. Disaggregation Transfer：负责 KV Cache 数据传输。
5. Model Runner：执行 Prefill/Decode Forward。

### 8.2 Prefill Worker

Prefill Worker 主要执行：

```
Receive Request
      |
      v
Tokenize / Prepare Input
      |
      v
Allocate Prefill KV
      |
      v
Prefill Forward
      |
      v
Generate KV Cache
      |
      v
Transfer KV Cache
      |
      v
Finish Prefill Stage
```

### 8.3 Decode Worker

Decode Worker 主要执行：

```
Receive Request Metadata
      |
      v
Allocate Decode KV
      |
      v
Wait for KV Transfer
      |
      v
Check Transfer Completion
      |
      v
Restore Request State
      |
      v
Join Running Batch
      |
      v
Decode Forward
      |
      v
Sample Token
      |
      v
Update KV Cache
```

### 8.4 为什么需要 KV Block Mapping？

在 Paged KV Cache 中，逻辑 Token 位置和物理 KV Block 之间通常存在映射关系。

例如：

```
Request A:

Logical Token Blocks:

[0] [1] [2] [3]

Prefill Physical Blocks:

[7] [2] [9] [5]

Decode Physical Blocks:

[4] [8] [1] [6]
```

P 和 D 两边分配的物理 KV Block 位置可能不同。

因此，不能简单地将 Prefill 的物理地址当作 Decode 的物理地址。

需要建立：

```
Prefill Block Index
          |
          v
Transfer Mapping
          |
          v
Decode Block Index
```

否则 Decode 可能读取错误的 KV Cache 数据。

### 8.5 Chunked Prefill 与异步传输

一些 PD 分离实现支持将较长的 Prefill 分为多个 Chunk。

例如：

```
Prompt = 8192 Tokens

Chunk 1 = 2048
Chunk 2 = 2048
Chunk 3 = 2048
Chunk 4 = 2048
```

可以形成流水线：

```
Time ------------------------------------------------>

Prefill:
| Chunk 1 | Chunk 2 | Chunk 3 | Chunk 4 |

Transfer:
          | KV 1    | KV 2    | KV 3    | KV 4 |
```

这样可以使部分 KV Cache 传输与后续 Prefill 计算重叠。

理想情况下，可以减少总延迟。

但是实际能否重叠，还取决于：

* KV Cache 布局。
* 通信后端是否支持异步传输。
* Transfer Stream 和 Compute Stream。
* Tensor 生命周期。
* 网络带宽。
* 传输完成事件同步。

---

## 九、PD 分离是否一定提升性能？

不一定。

PD 分离会引入额外的通信和调度开销。

是否能够带来收益，取决于原有 Prefill/Decode 竞争有多严重，以及新增 KV Transfer 成本有多高。

### 9.1 TTFT 分析

对于 PD 混部：

$$
TTFT_{\text{mix}}
\approx
T_{\text{queue}}+T_{\text{prefill}}
$$

这里省略了其他较小开销。

对于 PD 分离：

$$
TTFT_{\text{PD}}
\approx
T_{\text{queue,P}}
+
T_{\text{prefill}}
+
T_{\text{transfer}}
+
T_{\text{queue,D}}
+
T_{\text{resume}}
$$

实际执行中部分阶段可以重叠，因此需要按关键路径计算，而不是机械地将全部耗时相加。

PD 分离额外引入：

* KV Cache Transfer。
* P/D 之间的请求协调。
* Decode 恢复执行。
* 可能的 KV Reshard。

因此，单请求 TTFT 不一定下降。

### 9.2 TPOT 分析

PD 混部时：

```
Decode -> Prefill -> Decode
```

Decode 可能受到 Prefill 干扰。

PD 分离时：

```
Decode -> Decode -> Decode
```

由于长 Prefill 不再在同一计算实例上抢占执行时间，Decode 的 Token 间隔通常更加稳定。

因此，PD 分离的典型收益之一是降低尾部 ITL/TPOT。

但是如果 Decode 侧负载很高，或者 MoE 通信、KV Cache 带宽成为瓶颈，TPOT 仍可能增加。

### 9.3 哪些情况比较适合 PD 分离？


| 场景                  | 是否适合         | 原因                               |
| :-------------------- | :--------------- | :--------------------------------- |
| 高并发在线推理        | 通常适合         | 能减少 P/D 干扰                    |
| 长 Prompt             | 需要综合判断     | Prefill 计算量大，但 KV 传输量也大 |
| 长输出推理            | 通常适合         | Decode 可独立扩容                  |
| 低并发服务            | 未必适合         | 资源分离可能带来浪费               |
| 网络带宽受限          | 需要谨慎         | KV Transfer 可能成为瓶颈           |
| Prefix Cache 命中率高 | 需要结合缓存架构 | 计算和传输成本可能改变             |

---

## 十、PD 分离如何进行性能压测？

建议在相同模型和相同总硬件资源预算下，对比：

```
方案 A：PD 混部

方案 B：PD 分离
```

并保持：

* 输入数据集一致。
* 输入长度分布一致。
* 输出长度分布一致。
* 并发一致。
* 采样参数一致。
* TP/EP 配置尽量可比。
* CUDA/XPU Graph 配置可比。
* 模型权重和量化精度一致。

### 10.1 Prefill 指标

重点观察：

```
Prefill Latency
Input Tokens/s
Prefill Batch Size
Prefill Queue Time
Prefill Device Utilization
```

### 10.2 KV Transfer 指标

重点观察：

```
Transfer Bytes
Transfer Latency
Effective Bandwidth
Transfer Queue Time
Transfer Failure Rate
```

有效传输带宽：

$$
BW_{\text{effective}}
=
\frac{TransferBytes}{TransferTime}
$$

### 10.3 Decode 指标

重点观察：

```
TTFT
TPOT
ITL P50/P95/P99
Output Tokens/s
Decode Batch Size
KV Cache Utilization
```

### 10.4 整体指标


| 指标             | 作用                |
| :--------------- | :------------------ |
| QPS              | 请求吞吐量          |
| ITPS             | 输入 Token 吞吐     |
| OTPS             | 输出 Token 吞吐     |
| TTFT             | 首 Token 延迟       |
| TPOT             | 平均输出 Token 间隔 |
| ITL P99          | 尾部 Token 延迟     |
| KV Transfer Time | P/D 传输开销        |
| XPU Utilization  | 计算资源利用率      |

最终应该比较：

**在满足相同 TTFT/TPOT SLO 的情况下，哪种架构能够提供更高 QPS 或更高输出吞吐。**

单纯看吞吐量最高，不一定意味着实际线上性能最好。

---

## 十一、昆仑芯 P800 + SGLang 的适配关注点

对于昆仑芯 P800 上的 SGLang/PeakInfer 推理框架，PD 分离需要重点关注以下问题。

### 11.1 KV Transfer 后端

确认：

* 是否存在可用的 XPU KV Transfer 实现。
* 是否支持跨卡和跨节点传输。
* 是否支持设备内存直接传输。
* 是否需要 CPU 中转。
* 是否支持异步传输和事件同步。

不能直接假设 GPU 上使用的 RDMA/NIXL/Mooncake 路径可以原样运行在 XPU 上。

### 11.2 KV Cache 内存布局

需要确认：

* Prefill 和 Decode 的 KV Cache Layout。
* Block Size。
* Page Size。
* KV Dtype。
* Attention TP 分片方式。
* KV Block Index Mapping。
* MLA/GQA 等不同 Attention 结构。

如果两侧 Layout 不一致，需要执行 KV Reshard 或数据格式转换。

### 11.3 XPU Graph

在 Decode Worker 中使用 XPU Graph 时，需要考虑：

* KV Cache 地址是否稳定。
* Decode KV Block Mapping 是否正确。
* Graph Replay 是否使用最新的请求状态。
* 异步传输结束前是否错误读取 KV。
* 传输缓冲区是否发生生命周期冲突。
* 不同 Batch Size 下的 Graph Capture 是否正确。

特别是接收外部 KV 后，必须保证相应数据已经传输完成，才能由 Decode Kernel 消费。

### 11.4 MoE EP

对于 TP8EP8 等 MoE 部署：

* P/D 两侧可能拥有不同的 Expert Batch 分布。
* Prefill 的 Expert GEMM 规模通常更大。
* Decode 侧 Expert 负载可能更稀疏。
* All-to-All 通信成本可能不同。
* EPLB 策略可能需要分阶段调优。

因此，P/D 分离后可以分别分析 MoE 计算和通信性能。

### 11.5 投机解码

如果 Decode Worker 使用 EAGLE/MTP：

```
Prefill Worker
      |
      v
Target KV Cache
      |
      v
KV Transfer
      |
      v
Decode Worker
      |
      v
Draft Forward
      |
      v
Target Verify
      |
      v
Accept / Reject
      |
      v
Update KV Cache
```

还需要检查：

* Draft Model 的必要状态如何初始化。
* Draft KV 与 Target KV 的管理方式。
* EAGLE/MTP 的数据依赖是否被正确恢复。
* Speculative Verify 是否与 Decode KV Cache Layout 兼容。
* 接受长度变化是否影响 KV Block 管理。

PD 分离与投机解码可以组合使用，但需要确保相关状态一致。

---

## 十二、总结

PD 分离的核心可以归纳为三个方面。

### 1. 计算分离

Prefill 和 Decode 采用不同的计算实例。

$$
\text{Prefill Worker}\neq\text{Decode Worker}
$$

目的是降低阶段间的计算资源竞争。

### 2. KV Cache 传输

Prefill 计算完成后，需要将 Decode 所需的历史 Attention 状态传递给 Decode Worker。

$$
KV_{\text{Prefill}}
\xrightarrow{\text{Transfer}}
KV_{\text{Decode}}
$$

这是 PD 分离最重要的工程基础。

### 3. 独立资源调度

Prefill 和 Decode 可以采用不同的：

* 实例数量。
* Batch 策略。
* 并行策略。
* 资源配置。
* 扩容方式。

最终目标是在满足 TTFT、TPOT 等延迟要求的前提下，提高推理集群的整体服务能力。

**一句话总结：**

PD 分离通过将 Prefill 和 Decode 部署到不同的计算实例，结合 KV Cache 跨实例传输，实现计算资源专用化和跨请求流水线并行，从而缓解 Prefill 对 Decode 的性能干扰，并支持两个阶段独立调优与扩容。

---

## 十三、参考资料

1. SGLang GitHub[https://github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)
2. SGLang PD Disaggregation 文档[https://docs.sglang.ai/advanced_features/pd_disaggregation.html](https://docs.sglang.ai/advanced_features/pd_disaggregation.html)
3. vLLM Disaggregated Prefilling[https://docs.vllm.ai/en/stable/features/disagg_prefill/](https://docs.vllm.ai/en/stable/features/disagg_prefill/)
4. Mooncake 项目[https://github.com/kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)
5. NVIDIA NIXL[https://github.com/ai-dynamo/nixl](https://github.com/ai-dynamo/nixl)
