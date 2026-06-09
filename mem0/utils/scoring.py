"""
Scoring utilities for hybrid retrieval.

Provides:
- **BM25 normalization**: Sigmoid normalization of raw BM25 scores to [0, 1].
- **BM25 parameter selection**: Query-length-adaptive sigmoid parameters.
- **Additive scoring**: Combined scoring with semantic + BM25 + entity boost.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def get_bm25_params(query: str, *, lemmatized: Optional[str] = None) -> tuple:
    """Get BM25 sigmoid parameters based on query length.

    Longer queries tend to have higher raw BM25 scores, so we adjust
    the sigmoid midpoint and steepness accordingly.

    Returns:
        (midpoint, steepness) for sigmoid normalization.
    """

    # Step 1: 判断调用方是否已经传入 lemmatized query。
    # lemmatized 是经过词形还原 / 规范化后的 query，
    # 用它来统计 query term 数量会比直接使用原始 query 更稳定。
    if lemmatized is None:
        # Step 1.1: 如果调用方没有传入 lemmatized query，
        # 则在函数内部导入 lemmatize_for_bm25。
        # 这里使用局部导入，可以避免模块加载时产生额外依赖或循环导入问题。
        from mem0.utils.lemmatization import lemmatize_for_bm25

        # Step 1.2: 对原始 query 做 BM25 用的词形还原 / 规范化。
        lemmatized = lemmatize_for_bm25(query)

    # Step 2: 统计 query 中的 term 数量。
    # 如果 lemmatized 非空，就按空格 split 后计数；
    # 如果 lemmatized 为空，则兜底认为至少有 1 个 term。
    num_terms = len(lemmatized.split()) if lemmatized else 1

    # Step 3: 根据 query 长度选择 sigmoid normalization 的参数。
    # 背后逻辑是：
    # - 短 query 的 raw BM25 分数通常较低，所以 midpoint 设置得低一些；
    # - 长 query 的 raw BM25 分数通常更高，所以 midpoint 设置得高一些；
    # - steepness 控制 sigmoid 曲线变化速度，query 越长一般设置得越平缓。
    if num_terms <= 3:
        # Step 3.1: 短 query，1~3 个 term。
        # midpoint=5.0 表示 raw_score=5.0 时归一化后约为 0.5；
        # steepness=0.7 表示曲线相对更陡。
        return 5.0, 0.7

    elif num_terms <= 6:
        # Step 3.2: 中短 query，4~6 个 term。
        # 由于 query 更长，BM25 原始分数可能更高，因此 midpoint 提高到 7.0。
        return 7.0, 0.6

    elif num_terms <= 9:
        # Step 3.3: 中等长度 query，7~9 个 term。
        # midpoint 继续提高到 9.0，steepness 降到 0.5，让归一化更平滑。
        return 9.0, 0.5

    elif num_terms <= 15:
        # Step 3.4: 较长 query，10~15 个 term。
        # midpoint=10.0，steepness=0.5。
        return 10.0, 0.5

    else:
        # Step 3.5: 很长 query，超过 15 个 term。
        # 长 query 更容易得到较大的 raw BM25 分数，
        # 所以 midpoint 进一步提高到 12.0。
        return 12.0, 0.5


def normalize_bm25(raw_score: float, midpoint: float, steepness: float) -> float:
    """Normalize BM25 score to [0, 1] using logistic sigmoid.

    Args:
        raw_score: Raw BM25 score (unbounded, typically 0-20+).
        midpoint: Score at which sigmoid outputs 0.5.
        steepness: Controls how quickly sigmoid transitions.

    Returns:
        Normalized score in range [0, 1].
    """

    # Step 4: 使用 logistic sigmoid 将原始 BM25 分数归一化到 [0, 1]。
    #
    # 公式：
    # normalized = 1 / (1 + exp(-steepness * (raw_score - midpoint)))
    #
    # 含义：
    # - 当 raw_score == midpoint 时，normalized ≈ 0.5；
    # - 当 raw_score > midpoint 时，normalized 接近 1；
    # - 当 raw_score < midpoint 时，normalized 接近 0；
    # - steepness 越大，曲线越陡，分数变化越敏感；
    # - steepness 越小，曲线越平缓，分数变化越温和。
    return 1.0 / (1.0 + math.exp(-steepness * (raw_score - midpoint)))


# Step 5: 定义实体增强的最大权重。
# 在 score_and_rank() 中，entity_boost 会和 semantic_score、bm25_score 相加。
# ENTITY_BOOST_WEIGHT=0.5 表示实体信号最多按 0.5 这个量级参与综合打分。
ENTITY_BOOST_WEIGHT = 0.5


def score_and_rank(
    semantic_results: List[Dict[str, Any]],
    bm25_scores: Dict[str, float],
    entity_boosts: Dict[str, float],
    threshold: float,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Score candidates additively and return top-k results.

    For each candidate:
        semantic_score is taken from the result's score field.
        combined = (semantic + bm25 + entity_boost) / max_possible

    Threshold gates the semantic score BEFORE combining -- candidates
    below the threshold are excluded even if BM25/entity would boost them.

    The divisor adapts based on which signals are active:
        - Semantic only: max_possible = 1.0
        - Semantic + BM25: max_possible = 2.0
        - Semantic + BM25 + entity: max_possible = 2.5
        - Semantic + entity (no BM25): max_possible = 1.5

    Returns:
        List of scored result dicts sorted by combined score descending.
    """

    # Step 1: 判断本次搜索是否有 BM25 分数参与。
    # 如果 bm25_scores 非空，说明 keyword search / BM25 检索产生了有效信号。
    has_bm25 = bool(bm25_scores)

    # Step 2: 判断本次搜索是否有实体增强分数参与。
    # 如果 entity_boosts 非空，说明 query 中抽取出了实体，
    # 并且 entity_store 找到了相关实体和 linked memory。
    has_entity = bool(entity_boosts)

    # Step 3: 初始化最大可能分数。
    # 默认至少有 semantic score，因此基础 max_possible = 1.0。
    max_possible = 1.0

    # Step 4: 如果有 BM25 信号，则最大可能分数增加 1.0。
    # 也就是说，semantic + BM25 的理论最大值是 2.0。
    if has_bm25:
        max_possible += 1.0

    # Step 5: 如果有 entity boost 信号，则最大可能分数增加 ENTITY_BOOST_WEIGHT。
    # ENTITY_BOOST_WEIGHT 通常是一个全局常量，比如 0.5。
    # 所以 semantic + entity 的理论最大值一般是 1.5；
    # semantic + BM25 + entity 的理论最大值一般是 2.5。
    if has_entity:
        max_possible += ENTITY_BOOST_WEIGHT

    # Step 6: 初始化最终 scored 结果列表。
    # 每个元素最终形如：
    # {
    #     "id": "...",
    #     "score": combined_score,
    #     "payload": ...
    # }
    scored: List[Dict[str, Any]] = []

    # Step 7: 遍历语义检索返回的候选结果。
    # 注意：这里的主候选集合来自 semantic_results。
    # BM25 和 entity_boost 只对这些候选做加分，不会单独引入新的候选。
    for result in semantic_results:
        # Step 7.1: 取出 memory id。
        mem_id = result.get("id")

        # Step 7.2: 如果候选结果没有 id，则无法关联 BM25/entity 分数，直接跳过。
        if mem_id is None:
            continue

        # Step 7.3: 取出语义相似度分数。
        # 如果没有 score 字段，则默认按 0.0 处理。
        semantic_score = result.get("score", 0.0)

        # Step 7.4: 先用 semantic_score 做 threshold 过滤。
        # 这是一个重要设计：
        # 如果语义分数低于 threshold，即使 BM25 或 entity_boost 很高，也不会进入最终结果。
        if semantic_score < threshold:
            continue

        # Step 7.5: 将 memory id 转成字符串。
        # 因为 bm25_scores 和 entity_boosts 的 key 都是字符串形式的 memory_id。
        mem_id_str = str(mem_id)

        # Step 7.6: 取出该 memory 对应的 BM25 分数。
        # 如果没有命中 BM25，则默认为 0.0。
        bm25_score = bm25_scores.get(mem_id_str, 0.0)

        # Step 7.7: 取出该 memory 对应的实体增强分数。
        # 如果没有 entity boost，则默认为 0.0。
        entity_boost = entity_boosts.get(mem_id_str, 0.0)

        # Step 7.8: 将三个信号直接相加，得到原始综合分数。
        # raw_combined = 语义分数 + BM25 分数 + 实体增强分数。
        raw_combined = semantic_score + bm25_score + entity_boost

        # Step 7.9: 将原始综合分数除以 max_possible，归一化到 0~1 区间。
        # min(..., 1.0) 用于兜底，避免由于异常分数导致最终 score 超过 1。
        combined = min(raw_combined / max_possible, 1.0)

        # Step 7.10: 把计算后的结果加入 scored 列表。
        # payload 原样保留，供后续格式化成 MemoryItem。
        scored.append(
            {
                "id": mem_id_str,
                "score": combined,
                "payload": result.get("payload"),
            }
        )

    # Step 8: 按综合分数从高到低排序。
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Step 9: 只返回前 top_k 条结果。
    return scored[:top_k]
