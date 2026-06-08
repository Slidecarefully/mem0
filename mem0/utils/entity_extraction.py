"""
Entity extraction from text using spaCy NLP.

Extracts four types of entities from text:
- **Proper nouns**: Capitalized multi-word sequences (person names, places, brands)
- **Quoted text**: Text in single or double quotes (titles, specific terms)
- **Noun compounds**: Multi-word noun phrases with specific modifiers (e.g., "machine learning")
- **Noun fallback**: Single nouns from circumstantial compound patterns

Public API:
    extract_entities(text: str) -> List[Tuple[str, str]]

Internal:
    _extract_entities_from_doc(doc) -> List[Tuple[str, str]]
"""

# Step 0: 启用 postponed evaluation of annotations。
# 这样类型注解不会在函数定义时立即求值，有助于避免前向引用或运行时导入问题。
from __future__ import annotations

# Step 1: 导入标准库。
# logging 用于记录日志；re 用于正则抽取 quoted text 和清理文本；
# List / Tuple 用于类型标注。
import logging
import re
from typing import List, Tuple

# Step 2: 初始化当前模块的 logger。
logger = logging.getLogger(__name__)

# Step 3: 定义过于泛化的名词 head。
# 如果 noun chunk 的核心词是这些词，比如 thing / stuff / time，
# 通常无法形成有区分度的实体，因此后续会过滤掉。
# Words that are too generic to be useful as entity heads
_GENERIC_HEADS = {
    "thing", "stuff", "way", "time", "experience", "situation", "case",
    "fact", "matter", "issue", "idea", "thought", "feeling", "place",
    "area", "part", "kind", "type", "sort", "lot", "bit", "day", "year",
    "week", "month", "moment", "instance", "example", "technique",
    "method", "approach", "process", "step", "tool", "result", "outcome",
    "goal", "task", "item", "topic", "scale", "size", "level", "degree",
    "amount", "number", "style", "look", "color", "colour", "shape",
    "form", "piece", "section", "side", "end", "edge", "surface", "point",
}

# Step 4: 定义“环境/场景性修饰词”。
# 例如 solo / team / first / final 这类词更像上下文状态，
# 不一定能让 noun compound 成为更具体的实体。
# Modifiers that describe circumstance, not content
_CIRCUMSTANTIAL_MODS = {
    "solo", "individual", "team", "group", "joint", "collaborative",
    "first", "last", "next", "previous", "final", "initial", "main", "side",
}

# Step 5: 定义过于模糊的形容词集合。
# 如果 compound 里只有 good / big / recent / important 这类泛化形容词，
# 通常不认为它构成有意义的实体。
# Adjectives too vague to make a compound entity specific
_NON_SPECIFIC_ADJ = {
    "many", "few", "several", "some", "any", "all", "most", "more",
    "less", "much", "little", "enough", "various", "numerous", "multiple",
    "countless", "great", "good", "bad", "nice", "terrible", "awful",
    "awesome", "amazing", "wonderful", "horrible", "excellent", "poor",
    "best", "worst", "fine", "okay", "new", "old", "recent", "past",
    "future", "current", "previous", "next", "last", "first", "latest",
    "early", "late", "former", "modern", "ancient", "big", "small",
    "large", "tiny", "huge", "enormous", "long", "short", "tall", "high",
    "low", "wide", "narrow", "thick", "thin", "deep", "shallow",
    "similar", "different", "same", "other", "another", "such", "certain",
    "important", "main", "major", "minor", "key", "primary", "real",
    "actual", "true", "whole", "entire", "full", "complete", "total",
    "basic", "simple", "interesting", "boring", "exciting", "special",
    "particular", "general", "common", "unique", "rare", "typical",
    "usual", "normal", "regular", "possible", "likely", "potential",
    "available", "necessary", "only", "solo", "individual", "team",
    "group", "joint", "collaborative", "final", "initial", "side",
}

# Step 6: 定义 compound 末尾需要剥离的泛化尾词。
# 例如 "project details"、"shopping options" 中 details/options 信息量低，
# 后续会尝试从 compound 实体末尾去掉这类词。
# Generic tail words to strip from compound entities
_GENERIC_ENDINGS = {
    "work", "works", "job", "jobs", "task", "tasks", "stuff", "things",
    "thing", "info", "information", "details", "data", "content",
    "material", "materials", "activities", "activity", "efforts", "effort",
    "options", "option", "choices", "choice", "results", "result",
    "output", "outputs", "products", "product", "items", "item",
}

# Step 7: 定义首字母大写但过于泛化的词。
# 这些词即使被误认为 proper noun，也不应该作为实体保留。
# Capitalized single words that are too generic to be proper nouns
_GENERIC_CAPS = {
    "works", "items", "things", "stuff", "resources", "options", "tips",
    "ideas", "steps", "ways", "methods", "tools", "features", "benefits",
    "examples", "details", "notes", "instructions", "guidelines",
    "recommendations", "suggestions", "overview", "summary", "conclusion",
    "introduction", "pros", "cons", "advantages", "disadvantages",
}

# Step 8: 定义 Markdown 或格式化符号。
# 后续 proper noun sequence 抽取时会跳过这些 token，
# 避免把 bullet、标题符号等当成实体的一部分。
# Markdown/formatting markers to skip during extraction
_FORMATTING_MARKERS = {"*", "-", "+", "\u2022", "\u2013", "\u2014", "#", "##", "###", "**", "__"}


def _is_sentence_start(tokens: list, idx: int) -> bool:
    """Check if a token is at the start of a sentence or after formatting."""

    # Step 9: 判断当前 token 是否位于句子开头或格式化标记之后。
    # 这个函数主要用于 proper noun 识别：
    # 单词出现在句首时大写不一定代表专有名词，可能只是句首大写。

    # Step 9.1: 如果 idx 是 0，说明 token 在全文开头，视为句首。
    if idx == 0:
        return True

    # Step 9.2: 取出当前 token。
    tok = tokens[idx]

    # Step 9.3: 如果 spaCy 标注该 token 是句子开头，返回 True。
    if tok.is_sent_start:
        return True

    # Step 9.4: 取出前一个 token 文本。
    prev = tokens[idx - 1].text

    # Step 9.5: 如果前一个 token 是句末标点、冒号、格式化标记，
    # 或前一个 token 中含换行，则当前 token 也近似视为句首/段首。
    return prev in ".!?:" or prev in _FORMATTING_MARKERS or "\n" in prev


def _strip_generic_ending(toks: list) -> list:
    """Remove generic trailing words from compound token sequences."""

    # Step 10: 从 compound token 序列末尾去掉泛化尾词。
    # 例如 "machine learning details" 可以尝试去掉 details。
    # 但为了避免过度裁剪，只有长度大于 2 时才去掉末尾泛化词。

    # Step 10.1: 如果 token 数量小于等于 1，没有必要处理，直接返回原序列。
    if len(toks) <= 1:
        return toks

    # Step 10.2: 取最后一个 token 的 lemma。
    # 如果是 spaCy token，则使用 lemma_；否则兼容普通字符串对象。
    last = toks[-1].lemma_.lower() if hasattr(toks[-1], "lemma_") else toks[-1].lower()

    # Step 10.3: 如果末尾词属于 _GENERIC_ENDINGS，且序列长度大于 2，则去掉最后一个 token。
    # 否则保持原样。
    return toks[:-1] if last in _GENERIC_ENDINGS and len(toks) > 2 else toks


def _lemmatize_compound(toks: list) -> str:
    """Join compound tokens, lemmatizing nouns."""

    # Step 11: 将 compound tokens 拼接成字符串。
    # 其中名词使用 lemma_，其他词保留原文本。
    # 这样可以让实体形式更归一化，例如复数名词变为单数 lemma。
    return " ".join(t.lemma_ if t.pos_ == "NOUN" else t.text for t in toks)


def _has_artifacts(txt: str) -> bool:
    """Check for formatting artifacts that indicate non-entity text."""

    # Step 12: 判断文本是否包含明显格式噪声。
    # 如果有 markdown、重复空格、换行、过长文本、bullet 前缀等，
    # 通常说明它不是一个干净的实体。
    return any(
        [
            # Step 12.1: 包含 markdown 加粗符号或异常的 ":*"。
            "**" in txt or "__" in txt or ":*" in txt,

            # Step 12.2: 出现孤立星号或首尾星号。
            re.search(r"\s\*\s|\s\*$|^\*\s", txt),

            # Step 12.3: 包含多空格、换行或 tab。
            "  " in txt or "\n" in txt or "\t" in txt,

            # Step 12.4: 实体文本过长，超过 100 个字符，可能是句子而非实体。
            len(txt) > 100,

            # Step 12.5: 以 bullet 或破折号等格式标记开头。
            txt.startswith(("\u2022", "-", "+", "\u2013", "\u2014")),
        ]
    )


def extract_entities(text: str) -> List[Tuple[str, str]]:
    """Extract named entities, quoted text, and noun compounds from text.

    This is the public API that accepts a string. It loads the spaCy model
    internally and delegates to _extract_entities_from_doc().

    Args:
        text: Input text to extract entities from.

    Returns:
        Deduplicated list of (entity_type, entity_text) tuples.
        Entity types: PROPER, QUOTED, COMPOUND, NOUN.
        Returns empty list if spaCy is unavailable.
    """

    # Step 13: 单文本实体抽取的公开 API。
    # 输入普通字符串 text，内部加载 spaCy 完整模型，然后委托给 _extract_entities_from_doc()。

    # Step 13.1: 延迟导入 spaCy 模型加载函数。
    # 这样模块导入时不会立刻加载 spaCy，减少初始化开销。
    from mem0.utils.spacy_models import get_nlp_full

    # Step 13.2: 获取完整 spaCy NLP pipeline。
    nlp = get_nlp_full()

    # Step 13.3: 如果 spaCy 不可用，直接返回空列表。
    # 这样实体抽取失败不会影响主流程。
    if nlp is None:
        return []

    # Step 13.4: 用 spaCy 处理输入文本，得到 Doc。
    doc = nlp(text)

    # Step 13.5: 从 Doc 中抽取实体。
    return _extract_entities_from_doc(doc)


def extract_entities_batch(texts: List[str], batch_size: int = 32) -> List[List[Tuple[str, str]]]:
    """Extract entities from multiple texts using spaCy's nlp.pipe() for batched NER.

    Uses spaCy's efficient batch processing pipeline instead of calling
    nlp() individually per text. Significantly faster for multiple texts.

    Args:
        texts: List of input texts to extract entities from.
        batch_size: Number of texts to process in each spaCy batch.

    Returns:
        List of entity lists, one per input text. Each entity list contains
        (entity_type, entity_text) tuples. Returns list of empty lists if
        spaCy is unavailable.
    """

    # Step 14: 批量实体抽取 API。
    # 和 extract_entities() 相比，这里使用 nlp.pipe() 批处理，
    # 更适合 add() 批量写入 memory 后做 batch entity linking。

    # Step 14.1: 如果输入 texts 为空，直接返回空列表。
    if not texts:
        return []

    # Step 14.2: 延迟导入 spaCy 模型加载函数。
    from mem0.utils.spacy_models import get_nlp_full

    # Step 14.3: 获取完整 spaCy NLP pipeline。
    nlp = get_nlp_full()

    # Step 14.4: 如果 spaCy 不可用，需要返回和输入长度一致的空结果。
    # 这样调用方可以按索引继续对应每条输入文本。
    if nlp is None:
        return [[] for _ in texts]

    # Step 14.5: 初始化批量结果列表。
    results = []

    # Step 14.6: 使用 nlp.pipe() 批量处理文本。
    # batch_size 控制每批送入 spaCy 的文本数量。
    for doc in nlp.pipe(texts, batch_size=batch_size):
        # Step 14.7: 对每个 Doc 调用核心抽取函数，并加入 results。
        results.append(_extract_entities_from_doc(doc))

    # Step 14.8: 返回与 texts 一一对应的实体列表。
    return results


def _extract_entities_from_doc(doc) -> List[Tuple[str, str]]:
    """Extract entities from a spaCy Doc object.

    Ported from platform's shared.core.utils.entity_extraction.extract_entities().
    """

    # Step 15: 核心实体抽取函数。
    # 输入是 spaCy Doc，输出是去重、清洗后的实体列表。
    # 主要抽取四类：
    # - PROPER: 专有名词序列
    # - QUOTED: 引号中的文本
    # - COMPOUND: 多词复合名词
    # - NOUN: fallback 情况下的单名词

    # Step 15.1: 初始化实体列表。
    entities: List[Tuple[str, str]] = []

    # Step 15.2: 保存原始文本，用于后续正则抽取 quoted text。
    text = doc.text

    # Step 15.3: 将 spaCy Doc 转成 token 列表，方便按索引扫描。
    tokens = list(doc)

    # === PROPER NOUN SEQUENCES ===
    # Step 16: 抽取首字母大写的专有名词序列。
    # 典型目标包括人名、地名、品牌、产品名等。
    i = 0

    # Step 16.1: 用 while 循环从左到右扫描所有 token。
    while i < len(tokens):
        # Step 16.2: 取当前 token。
        tok = tokens[i]

        # Step 16.3: 跳过 Markdown / bullet 等格式标记。
        if tok.text in _FORMATTING_MARKERS:
            i += 1
            continue

        # Step 16.4: 判断当前 token 是否首字母大写。
        is_cap = tok.text and tok.text[0].isupper()

        # Step 16.5: 判断当前 token 是否像字段标签。
        # 例如 "Name:"、"User:" 这种不应被当作实体起点。
        is_label = i + 1 < len(tokens) and tokens[i + 1].text == ":"

        # Step 16.6: 当前 token 满足首字母大写、不是标签，
        # 且 POS 是 PROPN / NOUN / ADJ 时，才尝试构造 proper noun sequence。
        if is_cap and not is_label and tok.pos_ in {"PROPN", "NOUN", "ADJ"}:
            # Step 16.7: 初始化专有名词序列，保存 token 和原始索引。
            seq = [(tok, i)]

            # Step 16.8: 从下一个 token 开始向右扩展。
            j = i + 1
            while j < len(tokens):
                t = tokens[j]

                # Step 16.9: 如果后续 token 也是首字母大写，
                # 或者是连接词/function word，如 of / the / in / and，
                # 则继续纳入当前 sequence。
                if (t.text and t.text[0].isupper()) or t.text.lower() in {
                    "'s", "of", "the", "in", "and", "for", "at", "is",
                }:
                    seq.append((t, j))
                    j += 1
                else:
                    break

            # Step 16.10: 去掉 sequence 末尾的 function words。
            # 例如 "University of" 末尾的 of 不应保留。
            # Strip trailing function words
            while seq and seq[-1][0].text.lower() in {"of", "the", "in", "and", "for", "at", "is", "'s"}:
                seq.pop()

            # Step 16.11: 如果 sequence 非空，继续判断它是否真的像专有名词。
            if seq:
                # Step 16.12: 判断 sequence 中是否存在“非句首位置的大写词”。
                # 这是为了避免把普通句首大写单词误判为 proper noun。
                has_mid_cap = any(
                    not _is_sentence_start(tokens, idx)
                    for (t, idx) in seq
                    if t.text[0].isupper() and t.text.lower() not in {"'s", "of", "the", "in", "and", "for", "at", "is"}
                )

                # Step 16.13: 只有存在中间位置大写词时，才认为它更像专有名词。
                if has_mid_cap:
                    # Step 16.14: 按原始空白拼接 sequence。
                    phrase = "".join(t.text_with_ws for (t, idx) in seq).strip()

                    # Step 16.15: 长度大于 2 才加入实体列表。
                    if len(phrase) > 2:
                        entities.append(("PROPER", phrase))

            # Step 16.16: 跳到已经扫描过的 j 位置，避免重复处理。
            i = j
        else:
            # Step 16.17: 如果当前 token 不满足 proper noun 起点条件，则继续扫描下一个。
            i += 1

    # === QUOTED TEXT ===
    # Step 17: 抽取双引号中的文本。
    # 例如 "The Matrix" 会被抽成 QUOTED 实体。
    for m in re.finditer(r'"([^"]+)"', text):
        # Step 17.1: 引号内部长度大于 2 才保留。
        if len(m.group(1).strip()) > 2:
            entities.append(("QUOTED", m.group(1).strip()))

    # Step 17.2: 抽取单引号中的文本。
    # 正则要求单引号前后有合理边界，避免误抽英文缩写或所有格。
    for m in re.finditer(r"(?:^|[\s\(\[{,;])'([^']+)'(?=[\s\.,;:!?\)\]]|$)", text):
        # Step 17.3: 单引号内部长度大于 2 才保留。
        if len(m.group(1).strip()) > 2:
            entities.append(("QUOTED", m.group(1).strip()))

    # === NOUN-NOUN COMPOUNDS ===
    # Step 18: 从 spaCy noun_chunks 中抽取 noun compound。
    # 目标是 machine learning、security audit、memory system 这类多词名词短语。
    for chunk in doc.noun_chunks:
        # Step 18.1: 将 noun chunk 转成 token 列表。
        chunk_tokens = list(chunk)

        # Step 18.2: split_indices 用于记录需要切分 chunk 的位置。
        split_indices: list = []

        # Step 18.3: poss_splits 专门记录所有格切分位置，例如 John's book 中的 's。
        poss_splits: list = []

        # Step 18.4: 遍历 chunk 内部 token，查找所有格或引号等切分点。
        for idx, tok in enumerate(chunk_tokens):
            # Step 18.5: 如果 token 是所有格 case，例如 's / ’s / '，记录切分点。
            if tok.dep_ == "case" and tok.text in {"'s", "\u2019s", "'"}:
                split_indices.append(idx)
                poss_splits.append(idx)

            # Step 18.6: 如果 token 是引号类标点，也作为切分点。
            elif tok.pos_ == "PUNCT" and tok.text in {"'", '"', "\u2018", "\u2019", "\u201c", "\u201d"}:
                split_indices.append(idx)

        # Step 18.7: 如果存在切分点，则将 noun chunk 拆成多个 group。
        if split_indices:
            groups: list = []
            prev = 0

            # Step 18.8: 逐个处理切分点。
            for split_idx in split_indices:
                # Step 18.9: 将 split_idx 之前的内容作为一个 group。
                if split_idx > prev:
                    groups.append(chunk_tokens[prev:split_idx])

                # Step 18.10: 如果当前切分点是所有格，需要判断所有格后面的 owned 部分。
                if split_idx in poss_splits:
                    # Step 18.11: 找到下一个切分点，用于确定 owned 范围。
                    next_split = next((s for s in split_indices if s > split_idx), None)

                    # Step 18.12: 取所有格后面的 owned token。
                    owned = chunk_tokens[split_idx + 1: next_split if next_split else len(chunk_tokens)]

                    # Step 18.13: 如果 owned 非空，需要判断 owned 的第一个内容词是否大写。
                    if owned:
                        first_content = next((t for t in owned if t.pos_ not in {"PUNCT", "PART"}), None)

                        # Step 18.14: 如果 owned 的首个内容词不是大写，
                        # 则跳过这一段所有格后的普通内容，避免误抽泛化短语。
                        if not (first_content and first_content.text and first_content.text[0].isupper()):
                            prev = next_split if next_split else len(chunk_tokens)
                            continue

                # Step 18.15: 更新 prev，准备处理下一个 group。
                prev = split_idx + 1

            # Step 18.16: 如果最后还有剩余 token，则加入 groups。
            if prev < len(chunk_tokens):
                groups.append(chunk_tokens[prev:])
        else:
            # Step 18.17: 如果没有切分点，整个 noun chunk 作为一个 group。
            groups = [chunk_tokens]

        # Step 19: 对拆分后的每个 group 做 compound 实体识别。
        for group in groups:
            # Step 19.1: 空 group 跳过。
            if not group:
                continue

            # Step 19.2: 从 group 末尾向前找名词/专名作为 head。
            # head 是判断该短语是否有实体价值的核心词。
            head = next((t for t in reversed(group) if t.pos_ in {"NOUN", "PROPN"}), None)

            # Step 19.3: 如果没有名词或专名 head，跳过。
            if not head:
                continue

            # Step 19.4: 判断 head 是否是泛化 head。
            head_generic = head.lemma_.lower() in _GENERIC_HEADS

            # Step 19.5: 过滤掉限定词、代词、标点、介词、数字等低信息量 token。
            # 保留 ADJ，以及非 stopword 的内容词。
            content = [
                t
                for t in group
                if t.pos_ not in {"DET", "PRON", "PUNCT", "PART", "ADP", "SCONJ", "NUM"} and (t.pos_ == "ADJ" or not t.is_stop)
            ]

            # Step 19.6: 如果过滤后没有内容词，跳过。
            if not content:
                continue

            # Step 19.7: 找出 compound 依存关系的 token。
            compound_toks = [t for t in content if t.dep_ == "compound"]

            # Step 19.8: 找出形容词修饰词。
            adj_toks = [t for t in content if t.pos_ == "ADJ" or t.dep_ == "amod"]

            # Step 19.9: 判断是否存在具体形容词。
            # 如果形容词不在 _NON_SPECIFIC_ADJ 中，则认为它有区分度。
            has_spec_adj = any(t.lemma_.lower() not in _NON_SPECIFIC_ADJ for t in adj_toks)

            # Step 19.10: 如果 head 很泛化，且没有具体形容词，也没有 compound token，
            # 则说明这个短语信息量不足，跳过。
            if head_generic and not has_spec_adj and not compound_toks:
                continue

            # Step 20: 如果有 compound token，优先按 compound 逻辑处理。
            if compound_toks:
                # Step 20.1: 判断 compound token 是否只是场景性修饰词。
                # 例如 team project 中 team 可能只是上下文状态，而不是核心实体。
                is_circ = any(t.lemma_.lower() in _CIRCUMSTANTIAL_MODS for t in compound_toks)

                # Step 20.2: 如果是场景性修饰词，则退化为只抽 head 名词。
                if is_circ:
                    val = head.lemma_ if head.pos_ == "NOUN" else head.text

                    # Step 20.3: head 长度大于 2 才作为 NOUN 实体加入。
                    if len(val) > 2:
                        entities.append(("NOUN", val))
                else:
                    # Step 20.4: 如果 compound token 有实体价值，则构造 COMPOUND。
                    # 同时过滤掉非具体形容词。
                    filtered = _strip_generic_ending(
                        [t for t in content if not (t.pos_ == "ADJ" and t.lemma_.lower() in _NON_SPECIFIC_ADJ)]
                    )

                    # Step 20.5: 如果过滤后仍有 token，则 lemmatize 并拼接。
                    if filtered:
                        phrase = _lemmatize_compound(filtered)

                        # Step 20.6: 只有长度大于 3 且包含空格的短语才作为 COMPOUND。
                        if len(phrase) > 3 and " " in phrase:
                            entities.append(("COMPOUND", phrase))

            # Step 21: 如果没有 compound token，但存在多个内容词，并且有具体形容词，
            # 则也可以构成 COMPOUND。
            elif len(content) > 1 and has_spec_adj:
                # Step 21.1: 过滤掉非具体形容词，并剥离泛化结尾。
                filtered = _strip_generic_ending(
                    [t for t in content if not ((t.pos_ == "ADJ" or t.dep_ == "amod") and t.lemma_.lower() in _NON_SPECIFIC_ADJ)]
                )

                # Step 21.2: 如果过滤后仍有 token，则构造 phrase。
                if filtered:
                    phrase = _lemmatize_compound(filtered)

                    # Step 21.3: 只有多词短语才作为 COMPOUND。
                    if len(phrase) > 3 and " " in phrase:
                        entities.append(("COMPOUND", phrase))

    # === FALLBACK: Mis-tagged VERB heads ===
    # Step 22: fallback 逻辑。
    # 有些名词短语的 head 可能被 spaCy 误标成 VERB；
    # 这里尝试从这类 VERB head 周围补救抽取 compound。
    processed = {e[1].lower() for e in entities if e[0] == "COMPOUND"}

    # Step 22.1: 定义可能被当作泛化 verb head 的词。
    generic_verb_heads = _GENERIC_HEADS | {"find", "buy", "purchase", "sale", "deal", "trip", "visit"}

    def collect_compounds(head):
        # Step 22.2: 收集依存关系中以 head 为中心的 compound 修饰词。
        return [t for t in doc if t.head == head and t.dep_ == "compound"]

    # Step 22.3: 遍历 doc 中所有 token，寻找可能被误标为 VERB 的 head。
    for tok in doc:
        # Step 22.4: 只处理 POS 是 VERB，且依存关系像名词角色的 token。
        if tok.pos_ == "VERB" and tok.dep_ in {"pobj", "dobj", "nsubj"}:
            # Step 22.5: 收集该 token 的 compound 修饰词，并按原文顺序排序。
            comps = sorted(collect_compounds(tok), key=lambda t: t.i)

            # Step 22.6: 如果存在 compound 修饰词，则尝试构造短语。
            if comps:
                # Step 22.7: 如果当前 verb head 本身很泛化，
                # 则只使用 compound 修饰词；
                # 否则使用 compound 修饰词 + 当前 token。
                phrase_toks = comps if tok.lemma_.lower() in generic_verb_heads else comps + [tok]

                # Step 22.8: 拼接短语。
                phrase = " ".join(t.text for t in phrase_toks)

                # Step 22.9: 如果该 compound 还没处理过、长度足够、且是多词短语，则加入实体。
                if phrase.lower() not in processed and len(phrase) > 3 and " " in phrase:
                    entities.append(("COMPOUND", phrase))
                    processed.add(phrase.lower())

    # === DEDUPLICATION & CLEANUP ===
    # Step 23: 第一轮去重。
    # 按实体文本的小写形式去重，保留首次出现的实体类型和文本。
    seen: set = set()
    deduped = []

    # Step 23.1: 遍历所有候选实体。
    for t, e in entities:
        # Step 23.2: 用小写 + 去空格作为去重 key。
        k = e.lower().strip()

        # Step 23.3: 如果 key 没出现过且长度大于 2，则保留。
        if k not in seen and len(k) > 2:
            seen.add(k)
            deduped.append((t, e))

    # Step 24: 清理实体文本。
    cleaned: List[Tuple[str, str]] = []

    # Step 24.1: 遍历去重后的实体。
    for etype, etext in deduped:
        # Step 24.2: 去掉首尾 markdown 星号。
        txt = re.sub(r"^\*+\s*|\s*\*+$", "", etext.strip())

        # Step 24.3: 去掉末尾冒号。
        txt = re.sub(r"\s*:+$", "", txt)

        # Step 24.4: 去掉编号前缀，例如 "1. xxx"。
        txt = re.sub(r"^\d+\s*\.\s*", "", txt)

        # Step 24.5: 如果清理后为空、过短、或包含格式噪声，则跳过。
        if not txt or len(txt) <= 2 or _has_artifacts(txt):
            continue

        # Step 24.6: 如果是单词 PROPER 且属于泛化大写词，则跳过。
        if etype == "PROPER" and " " not in txt and txt.lower() in _GENERIC_CAPS:
            continue

        # Step 24.7: 保留清洗后的实体。
        cleaned.append((etype, txt))

    # Step 25: 同一个实体文本可能被不同规则抽成不同类型。
    # 这里按优先级保留最佳类型：
    # PROPER > COMPOUND > QUOTED > NOUN > VERB。
    # Keep best type per entity (PROPER > COMPOUND > QUOTED > NOUN)
    type_pri = {"PROPER": 0, "COMPOUND": 1, "QUOTED": 2, "NOUN": 3, "VERB": 4}
    best: dict = {}

    # Step 25.1: 遍历清洗后的实体。
    for t, e in cleaned:
        # Step 25.2: 用实体文本小写作为 key。
        k = e.lower()

        # Step 25.3: 如果该实体还没出现过，
        # 或当前类型优先级高于已有类型，则更新 best。
        if k not in best or type_pri.get(t, 99) < type_pri.get(best[k][0], 99):
            best[k] = (t, e)

    # Step 25.4: 得到最佳类型去重后的实体列表。
    deduped = list(best.values())

    # Step 26: 去掉被更长实体包含的短实体。
    # 例如已经有 "machine learning model" 时，
    # 可以去掉它的子串 "machine learning"。
    # Remove entities that are substrings of longer entities
    all_lower = [e[1].lower() for e in deduped]

    # Step 26.1: 返回最终实体列表。
    # 条件是：当前实体不能是另一个更长实体的子串。
    return [(t, e) for t, e in deduped if not any(e.lower() != o and e.lower() in o for o in all_lower)]