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
    if lemmatized is None:
        from mem0.utils.lemmatization import lemmatize_for_bm25

        lemmatized = lemmatize_for_bm25(query)
    num_terms = len(lemmatized.split()) if lemmatized else 1

    if num_terms <= 3:
        return 5.0, 0.7
    elif num_terms <= 6:
        return 7.0, 0.6
    elif num_terms <= 9:
        return 9.0, 0.5
    elif num_terms <= 15:
        return 10.0, 0.5
    else:
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
    return 1.0 / (1.0 + math.exp(-steepness * (raw_score - midpoint)))


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
