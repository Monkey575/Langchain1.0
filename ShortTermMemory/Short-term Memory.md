# Short-term Memory *(短期记忆)*
https://docs.langchain.com/oss/python/langchain/short-term-memory#prompt
## Overview
- 短期记忆可以让智能体记住单一线程或对话之前的内容。
- 长期记忆难以实现，因为llm上下文窗口有限
- 短期记忆通过将之前的对话存放在消息列表中实现，随着对话的进行，可以将部分无用对话从消息列表去除，从而实现长期记忆的功能

## 短期记忆、长期记忆、RAG比对
- 短期记忆（Checkpointer）: 记住会话历史。
- 长期记忆（Store + Tool）： 记住用户 ID 和偏好。
- RAG（向量数据库 + 检索器）： 记住公司知识库。

## 实现
在创建agent时指定checkpoint
- 可以基于内存实现 InMemorySaver() *（仅限单个进程使用）*
- 可以基于数据库实现 SqliteSaver() *（当然不止sqlite数据库，存储在硬盘中，可跨进程使用）*
- 可以通过stateAgent来自定义标识特殊数据，减少llm读取数据量

## 控制llm上下文窗口
在短期记忆中，llm上下文窗口大小有限，需要一定方法控制该窗口大小，常见的方法有：修剪、删除、摘要
| 特性 | 修剪消息（trim_messages） | 删除消息（delete_old_messages） |
| :--- | :--- | :--- |
| **执行时机** | 在大模型生成之前那 | 在大模型生成之后 |
| **机制** | 替换：计算出新的消息列表，然后用新列表 **替换** 旧列表。 | 指定移除：返回一个 RemoveMessage 列表，LangGraph 收到指令后，根据消息 ID 精确删除。 |
| **消息保留** | 保留策略通常是 **“首条消息 + 滑动窗口”**。例如：保留第一条消息和最近的 N 条消息。 | 保留策略是 **“丢弃最早的 N 条消息”**。例如：丢弃最早的两条消息，其余保留。 |
| **状态操作** | 返回一个包含完整新消息列表的字典。 | 返回一个包含 `RemoveMessage` 对象的特殊列表，它是一个 **操作指令**。 |
| **代码目的** | 确保 LLM **推理前** 输入的消息数量在限制内。 | 确保 LLM **推理后** 存储在 Checkpoint 中的消息数量在限制内。 |

- 摘要
使用摘要中间件 from langchain.agents.middleware import SummarizationMiddleware
通过调用额外的模型对旧消息进行摘要概括，作为一个更简短的消息

## 访问存储器
- 通过tool读取、写入记忆
通过 ToolRunTime参数：from langchain.tools import ToolRuntime
- 通过prompt访问记忆