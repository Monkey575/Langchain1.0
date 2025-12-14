# Langchain1.0

#### 文档

官网文档：https://docs.langchain.com/

api文档：https://reference.langchain.com/python/langchain/



# Search

## Retrieval 检索

**LLM限制**：

- 有限的上下文，无法读取过长的整段文本
- 静态的知识库，llm的训练数据取决于上次的训练

**retrieval**通过在查询时获取外部信息来解决这些问题，这也是**RAG（检索增强生成）**的基础

### 1、Building a knowledge base 构建知识库

***	knowledge base**即检索中使用的文档或结构化数据库（sql数据库、向量数据库等）*

#### RAG（检索增强生成）

Retrieval允许llm在运行时查询外部信息，而RAG则更进一步，将外部信息与llm生成结合

#### 检索工作流程

![image-20251213112147529](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20251213112147529.png)

- 将消息来源（**Sources**）通过文档转化器（**Document** **Loaders**）转化为**Document**对象，将其分块（**Split** **into** **chunks**）（避免超出llm文本限制），再将其变为计算机可以识别的向量（**Turn** **into** **chunks**），再由这些向量构成一个向量数据库（**Vector** **Store**），该向量库即本次的知识库。
- 将用户查询（**User** **Query**）转化为向量（**Query** **embedding**），进入检索器（**Rertriever**），遍历向量库与查询向量比对，找出最相似的文本向量，大模型使用检索到的信息（**LLM uses retrieved info**），生成最终答案（**Answer**）
