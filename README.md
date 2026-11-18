# Documents

技术学习笔记与原理解析，当前主要记录大模型推理中的拒绝采样与投机解码。

## 文档目录

| 文档 | 内容 |
| --- | --- |
| [拒绝采样（Rejection Sampling）原理与实现详解](./拒绝采样.md) | 拒绝采样的数学原理、投机解码流程、PyTorch 示例及工程实现注意事项 |

## 阅读内容

- 拒绝采样在投机解码中的作用。
- 接受概率、残差分布与无偏性证明。
- 单 Token 与多 Token 的 PyTorch 实现示例。
- Rejection Sampling 与 Greedy Verification 的区别。
- SGLang 实现流程与昆仑芯 XPU 适配注意事项。
- 接受率、性能分析及正确性验证建议。

## 阅读方式

点击上方文档链接即可在 GitHub 中阅读，也可以克隆到本地：

```bash
git clone https://github.com/XiaoDiandian-623/Documents.git
cd Documents
```

文档采用 Markdown 格式，包含数学公式和代码示例，建议使用支持公式渲染的阅读器。
