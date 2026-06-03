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

    # Step 1: 判断是否有 BM25 分数和实体增强分数
    has_bm25 = bool(bm25_scores)
    has_entity = bool(entity_boosts)

    # Step 2: 初始化最大可能分数（max_possible），用于归一化
    max_possible = 1.0  # 语义分本身最大值为 1.0
    if has_bm25:
        max_possible += 1.0  # 如果 BM25 存在，最大可能值增加 1.0
    if has_entity:
        max_possible += ENTITY_BOOST_WEIGHT  # 如果 entity boost 存在，增加全局权重

    # Step 3: 用于存储最终评分后的候选结果
    scored: List[Dict[str, Any]] = []

    # Step 4: 遍历每个 semantic candidate
    for result in semantic_results:
        # Step 4.1: 获取 memory id
        mem_id = result.get("id")
        if mem_id is None:
            continue  # 如果没有 id，跳过该 candidate

        # Step 4.2: 获取 semantic 分数
        semantic_score = result.get("score", 0.0)

        # Step 4.3: 应用 threshold 门限
        # 如果 semantic 分数低于 threshold，即使 BM25 或 entity boost 很高也会被排除
        if semantic_score < threshold:
            continue

        mem_id_str = str(mem_id)

        # Step 4.4: 获取 BM25 分数（如果没有则默认为 0）
        bm25_score = bm25_scores.get(mem_id_str, 0.0)

        # Step 4.5: 获取 entity boost 分数（如果没有则默认为 0）
        entity_boost = entity_boosts.get(mem_id_str, 0.0)

        # Step 4.6: 计算原始组合分数
        raw_combined = semantic_score + bm25_score + entity_boost

        # Step 4.7: 归一化组合分数到 [0,1]，防止超出 1
        combined = min(raw_combined / max_possible, 1.0)

        # Step 4.8: 将 memory id、归一化分数和 payload 封装到 scored 列表
        scored.append(
            {
                "id": mem_id_str,
                "score": combined,
                "payload": result.get("payload"),
            }
        )

    # Step 5: 根据 combined score 降序排序
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Step 6: 返回 top_k 个候选结果
    return scored[:top_k]
