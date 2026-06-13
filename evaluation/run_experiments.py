import argparse
import os

# 不同实验方法的具体实现类
# 每个类通常负责一种记忆 / 检索 / 预测流程
from src.langmem import LangMemManager
from src.memzero.add import MemoryADD
from src.memzero.search import MemorySearch
from src.openai.predict import OpenAIPredict
from src.rag import RAGManager
from src.utils import METHODS, TECHNIQUES
from src.zep.add import ZepAdd
from src.zep.search import ZepSearch


class Experiment:
    def __init__(self, technique_type, chunk_size):
        # 保存当前实验使用的技术类型，例如 mem0、rag、langmem、zep、openai
        self.technique_type = technique_type

        # 保存分块大小，主要用于 RAG 这类需要 chunk 的方法
        self.chunk_size = chunk_size

    def run(self):
        # 这里只是一个简单的实验入口示例
        # 当前主逻辑并没有真正使用 Experiment 类，而是直接在 main() 中分发任务
        print(f"Running experiment with technique: {self.technique_type}, chunk size: {self.chunk_size}")


def main():
    # 创建命令行参数解析器
    # 这个脚本通过命令行参数控制运行哪种 memory technique、执行 add 还是 search、输出到哪里等
    parser = argparse.ArgumentParser(description="Run memory experiments")

    # 指定实验使用的技术路线
    # choices=TECHNIQUES 限制只能选择项目中预定义的技术类型
    parser.add_argument("--technique_type", choices=TECHNIQUES, default="mem0", help="Memory technique to use")

    # 指定当前技术路线下要执行的方法
    # 一般 add 表示构建 / 写入记忆，search 表示检索 / 问答
    parser.add_argument("--method", choices=METHODS, default="add", help="Method to use")

    # 指定文本处理时的 chunk 大小
    # 主要用于 RAG，因为 RAG 会把对话切块后再检索
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for processing")

    # 指定实验结果输出目录
    parser.add_argument("--output_folder", type=str, default="results/", help="Output path for results")

    # 指定检索时返回的 top-k 条 memory
    # 主要用于 mem0 search
    parser.add_argument("--top_k", type=int, default=30, help="Number of top memories to retrieve")

    # 是否对检索到的 memories 做进一步过滤
    # action="store_true" 表示命令行中出现该参数时为 True，否则为默认值 False
    parser.add_argument("--filter_memories", action="store_true", default=False, help="Whether to filter memories")

    # 是否使用 graph-based search
    # 主要影响 mem0 的 add / search 逻辑
    parser.add_argument("--is_graph", action="store_true", default=False, help="Whether to use graph-based search")

    # 指定 RAG 检索时取多少个 chunk
    # 虽然参数名是 num_chunks，但在 RAGManager 中对应的是 k
    parser.add_argument("--num_chunks", type=int, default=1, help="Number of chunks to process")

    # 解析命令行输入，得到所有实验配置
    args = parser.parse_args()

    # Add your experiment logic here

    # 打印当前实验的基础配置，方便确认命令行参数是否生效
    print(f"Running experiments with technique: {args.technique_type}, chunk size: {args.chunk_size}")

    # 根据 technique_type 选择不同的实验路线
    # 每个 technique 内部可能还会根据 method 决定是写入记忆还是检索记忆
    if args.technique_type == "mem0":

        # mem0 的 add 阶段：把原始对话数据写入 / 构建成 memory
        if args.method == "add":
            # 使用 locomo10.json 作为输入数据
            # is_graph 决定是否启用图结构记忆
            memory_manager = MemoryADD(data_path="dataset/locomo10.json", is_graph=args.is_graph)

            # 处理所有对话并完成 memory 添加
            memory_manager.process_all_conversations()

        # mem0 的 search 阶段：基于已经构建好的 memory 进行检索和回答
        elif args.method == "search":
            # 根据 top_k、filter_memories、is_graph 等参数生成结果文件名
            # 这样不同实验设置的结果不会互相覆盖
            output_file_path = os.path.join(
                args.output_folder,
                f"mem0_results_top_{args.top_k}_filter_{args.filter_memories}_graph_{args.is_graph}.json",
            )

            # 初始化 mem0 检索器
            # top_k 控制检索多少条 memory
            # filter_memories 控制是否过滤 memory
            # is_graph 控制是否使用图结构搜索
            memory_searcher = MemorySearch(output_file_path, args.top_k, args.filter_memories, args.is_graph)

            # 在原始数据集上执行检索 / 问答流程，并把结果写入 output_file_path
            memory_searcher.process_data_file("dataset/locomo10.json")

    elif args.technique_type == "rag":
        # RAG 路线不需要 add / search 两阶段
        # 它会在运行时对对话切块、计算 embedding、检索相关 chunk，然后生成答案

        # 根据 chunk_size 和 num_chunks 生成结果文件名
        # chunk_size 表示每块多大，num_chunks 表示每个问题检索几个 chunk
        output_file_path = os.path.join(args.output_folder, f"rag_results_{args.chunk_size}_k{args.num_chunks}.json")

        # 初始化 RAGManager
        # data_path 使用 locomo10_rag.json，通常是适配 RAG 格式的数据
        # chunk_size 控制切块大小
        # k=args.num_chunks 控制每个问题检索 top-k 个 chunk
        rag_manager = RAGManager(data_path="dataset/locomo10_rag.json", chunk_size=args.chunk_size, k=args.num_chunks)

        # 对所有 conversation 执行 RAG 问答，并保存结果
        rag_manager.process_all_conversations(output_file_path)

    elif args.technique_type == "langmem":
        # LangMem 路线：使用 LangMemManager 管理完整实验流程
        # 这里没有区分 add / search，说明封装类内部可能已经处理了记忆构建和查询流程

        # 指定 LangMem 实验结果输出路径
        output_file_path = os.path.join(args.output_folder, "langmem_results.json")

        # 初始化 LangMemManager，输入数据使用 locomo10_rag.json
        langmem_manager = LangMemManager(dataset_path="dataset/locomo10_rag.json")

        # 处理所有 conversation，并把结果保存到输出文件
        langmem_manager.process_all_conversations(output_file_path)

    elif args.technique_type == "zep":

        # Zep 的 add 阶段：把原始对话写入 Zep memory/session
        if args.method == "add":
            # 初始化 ZepAdd，读取原始 locomo10.json 数据
            zep_manager = ZepAdd(data_path="dataset/locomo10.json")

            # 处理所有 conversation
            # 参数 "1" 很可能表示某个固定的 user_id、session_id 或实验 namespace
            zep_manager.process_all_conversations("1")

        # Zep 的 search 阶段：从 Zep 中检索相关记忆并回答问题
        elif args.method == "search":
            # 指定 Zep 检索实验结果输出路径
            output_file_path = os.path.join(args.output_folder, "zep_search_results.json")

            # 初始化 ZepSearch
            zep_manager = ZepSearch()

            # 使用 locomo10.json 作为问题来源
            # "1" 应与 add 阶段使用的标识保持一致，确保搜索的是同一批写入的记忆
            zep_manager.process_data_file("dataset/locomo10.json", "1", output_file_path)

    elif args.technique_type == "openai":
        # OpenAI baseline 路线
        # 通常表示直接使用 OpenAI 模型预测，而不是显式使用 mem0、RAG、Zep 等记忆系统

        # 指定 OpenAI baseline 的结果输出路径
        output_file_path = os.path.join(args.output_folder, "openai_results.json")

        # 初始化 OpenAIPredict
        openai_manager = OpenAIPredict()

        # 在 locomo10.json 上执行预测，并保存结果
        openai_manager.process_data_file("dataset/locomo10.json", output_file_path)

    else:
        # 理论上 argparse 的 choices 已经限制了 technique_type 的合法取值
        # 这里是额外的防御性检查，避免非法配置进入实验流程
        raise ValueError(f"Invalid technique type: {args.technique_type}")


if __name__ == "__main__":
    # 当该文件作为脚本直接运行时，进入 main()
    # 如果该文件被其他模块 import，则不会自动执行实验
    main()
