import json
import os
import time
from collections import defaultdict

import numpy as np
import tiktoken
from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm


# 加载 .env 文件中的环境变量，例如 MODEL、EMBEDDING_MODEL、OPENAI_API_KEY 等
load_dotenv()


# 用 Jinja2 模板定义最终发给大模型的 Prompt 结构
# 这里把问题和检索到的上下文拼接起来，让模型基于 Context 回答 Question
PROMPT = """
# Question: 
{{QUESTION}}

# Context: 
{{CONTEXT}}

# Short answer:
"""


class RAGManager:
    def __init__(self, data_path="dataset/locomo10_rag.json", chunk_size=500, k=1):
        # 从环境变量中读取用于生成回答的模型名称
        self.model = os.getenv("MODEL")

        # 初始化 OpenAI 客户端，后续会同时用于生成 embedding 和调用 chat completion
        self.client = OpenAI()

        # 数据集路径，默认读取 locomo10_rag.json
        self.data_path = data_path

        # 每个文本块的 token 大小
        # 如果 chunk_size == -1，则不切块，直接把整段对话作为上下文
        self.chunk_size = chunk_size

        # 检索时返回 top-k 个最相关的 chunk
        self.k = k

    def generate_response(self, question, context):
        # 根据 PROMPT 模板，把当前问题和检索到的上下文渲染成完整 prompt
        template = Template(PROMPT)
        prompt = template.render(CONTEXT=context, QUESTION=question)

        # 设置最多重试次数，用于处理临时 API 错误
        max_retries = 3
        retries = 0

        # 尝试调用大模型生成答案；如果失败，则等待后重试
        # 注意这里是 while retries <= max_retries，因此最多会尝试 4 次
        while retries <= max_retries:
            try:
                # 记录模型生成开始时间，用于统计 response_time
                t1 = time.time()

                # 调用 Chat Completions API
                # system message 约束模型必须基于给定 context 回答，并尽量简短
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that can answer "
                            "questions based on the provided context."
                            "If the question involves timing, use the conversation date for reference."
                            "Provide the shortest possible answer."
                            "Use words directly from the conversation when possible."
                            "Avoid using subjects in your answer.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )

                # 记录模型生成结束时间
                t2 = time.time()

                # 返回模型回答和本次生成耗时
                return response.choices[0].message.content.strip(), t2 - t1

            except Exception as e:
                # 如果调用失败，累计重试次数
                retries += 1

                # 超过最大重试次数后，不再吞掉异常，直接抛出
                if retries > max_retries:
                    raise e

                # 简单等待 1 秒后再重试，避免短时间内连续请求失败
                time.sleep(1)  # Wait before retrying

    def clean_chat_history(self, chat_history):
        # 将原始对话历史转换成统一的纯文本格式
        # 每条消息都保留 timestamp、speaker 和 text，方便后续切块和检索
        cleaned_chat_history = ""

        for c in chat_history:
            cleaned_chat_history += f"{c['timestamp']} | {c['speaker']}: {c['text']}\n"

        return cleaned_chat_history

    def calculate_embedding(self, document):
        # 对单个文档或文本块计算 embedding
        # embedding 模型名称从环境变量 EMBEDDING_MODEL 中读取
        response = self.client.embeddings.create(model=os.getenv("EMBEDDING_MODEL"), input=document)

        # OpenAI embedding 接口返回的是列表，这里只取第一个输入对应的 embedding 向量
        return response.data[0].embedding

    def calculate_similarity(self, embedding1, embedding2):
        # 使用余弦相似度衡量 query embedding 和 chunk embedding 的相似程度
        # 值越大，表示两个向量语义越接近
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def search(self, query, chunks, embeddings, k=1):
        """
        Search for the top-k most similar chunks to the query.

        Args:
            query: The query string
            chunks: List of text chunks
            embeddings: List of embeddings for each chunk
            k: Number of top chunks to return (default: 1)

        Returns:
            combined_chunks: The combined text of the top-k chunks
            search_time: Time taken for the search
        """

        # 记录检索开始时间，用于统计 search_time
        t1 = time.time()

        # 先将问题本身转换为 embedding，方便与所有 chunk 的 embedding 做相似度比较
        query_embedding = self.calculate_embedding(query)

        # 分别计算 query 与每个 chunk 的余弦相似度
        similarities = [self.calculate_similarity(query_embedding, embedding) for embedding in embeddings]

        # Get indices of top-k most similar chunks
        if k == 1:
            # Original behavior - just get the most similar chunk

            # 当只取一个 chunk 时，直接找到相似度最高的下标
            top_indices = [np.argmax(similarities)]
        else:
            # Get indices of top-k chunks

            # 当需要多个 chunk 时，先按相似度排序，再取最高的 k 个
            # np.argsort(similarities) 返回的是从小到大的索引
            # [-k:] 取最后 k 个最高分，再 [::-1] 反转成从高到低
            top_indices = np.argsort(similarities)[-k:][::-1]

        # Combine the top-k chunks

        # 将检索出的多个 chunk 合并成一个上下文字符串
        # 使用 <-> 作为分隔符，帮助模型区分不同来源的文本块
        combined_chunks = "\n<->\n".join([chunks[i] for i in top_indices])

        # 记录检索结束时间
        t2 = time.time()

        # 返回合并后的上下文，以及检索耗时
        return combined_chunks, t2 - t1

    def create_chunks(self, chat_history, chunk_size=500):
        """
        Create chunks using tiktoken for more accurate token counting
        """

        # Get the encoding for the model

        # 根据 embedding 模型选择对应的 tokenizer
        # 这样可以按真实 token 数切分，而不是按字符数粗略切分
        encoding = tiktoken.encoding_for_model(os.getenv("EMBEDDING_MODEL"))

        # 先把结构化的 chat_history 转成纯文本
        documents = self.clean_chat_history(chat_history)

        # 如果 chunk_size 设置为 -1，则跳过切块和 embedding
        # 这种模式适合直接使用完整对话作为上下文
        if chunk_size == -1:
            return [documents], []

        chunks = []

        # Encode the document

        # 将完整对话文本编码成 token 序列
        tokens = encoding.encode(documents)

        # Split into chunks based on token count

        # 按 chunk_size 对 token 序列进行切片
        # 每个切片再 decode 回文本，形成一个 chunk
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk = encoding.decode(chunk_tokens)
            chunks.append(chunk)

        # 对每个 chunk 预先计算 embedding
        # 后续每个问题检索时，可以直接复用这些 embedding，避免重复计算 chunk embedding
        embeddings = []

        for chunk in chunks:
            embedding = self.calculate_embedding(chunk)
            embeddings.append(embedding)

        # 返回切分后的文本块，以及每个文本块对应的 embedding
        return chunks, embeddings

    def process_all_conversations(self, output_file_path):
        # 读取整个数据集
        # 数据结构通常是：每个 key 对应一段 conversation 和一组 question
        with open(self.data_path, "r") as f:
            data = json.load(f)

        # 使用 defaultdict(list) 存储最终结果
        # 每个 conversation key 对应多个问题的回答结果
        FINAL_RESULTS = defaultdict(list)

        # 外层循环：逐个处理数据集中的 conversation
        for key, value in tqdm(data.items(), desc="Processing conversations"):
            # 当前样本中的完整对话历史
            chat_history = value["conversation"]

            # 当前对话对应的一组问题
            questions = value["question"]

            # 对当前对话进行切块，并为每个 chunk 计算 embedding
            # 这一步只对每个 conversation 做一次，后续多个问题共享 chunks 和 embeddings
            chunks, embeddings = self.create_chunks(chat_history, self.chunk_size)

            # 内层循环：对当前 conversation 下的每个问题进行检索和回答
            for item in tqdm(questions, desc="Answering questions", leave=False):
                # 取出问题文本
                question = item["question"]

                # 数据集中提供的标准答案；如果没有 answer 字段，则默认为空字符串
                answer = item.get("answer", "")

                # 问题类别，用于后续按类别分析效果
                category = item["category"]

                # 如果 chunk_size == -1，说明不使用 RAG 检索，而是直接把完整对话作为上下文
                if self.chunk_size == -1:
                    context = chunks[0]
                    search_time = 0
                else:
                    # 否则，先用问题在 chunks 中检索最相关的 top-k 上下文
                    context, search_time = self.search(question, chunks, embeddings, k=self.k)

                # 将问题和检索到的上下文交给大模型，生成简短答案
                response, response_time = self.generate_response(question, context)

                # 保存当前问题的完整实验结果
                # 包括原问题、标准答案、问题类别、检索上下文、模型回答以及耗时
                FINAL_RESULTS[key].append(
                    {
                        "question": question,
                        "answer": answer,
                        "category": category,
                        "context": context,
                        "response": response,
                        "search_time": search_time,
                        "response_time": response_time,
                    }
                )

                # 每回答完一个问题就写一次文件
                # 好处是即使程序中途失败，也能保留已经完成的结果
                # 代价是频繁写文件会带来一些 I/O 开销
                with open(output_file_path, "w+") as f:
                    json.dump(FINAL_RESULTS, f, indent=4)

        # Save results

        # 所有 conversation 都处理完成后，再整体保存一次最终结果
        # 这一步和上面循环中的增量保存逻辑有重复，但可以确保最终文件完整写入
        with open(output_file_path, "w+") as f:
            json.dump(FINAL_RESULTS, f, indent=4)
