# ==================== 中文逻辑注释版 ====================
# 说明：原始代码和原有英文注释均已保留；新增的中文注释以“逻辑注释”标识。
# 注释重点解释每段代码在整体记忆系统中的作用，而不是简单翻译语法。
# =======================================================

# 逻辑注释：导入异步、GC、哈希、JSON、日志、路径、UUID 等基础能力，后面分别用于异步封装、去重、解析、审计和唯一 ID。
import asyncio
import gc
import hashlib
import json
import logging
import os
import uuid
import warnings
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# 逻辑注释：引入 Pydantic 校验异常，用来区分配置模型校验错误和业务层自定义校验错误。
from pydantic import ValidationError

# 逻辑注释：MemoryConfig 定义整体配置结构，MemoryItem 统一对外返回的记忆数据形态。
from mem0.configs.base import MemoryConfig, MemoryItem
# 逻辑注释：MemoryType 用枚举约束记忆类型，避免用散落的字符串判断业务分支。
from mem0.configs.enums import MemoryType
# 逻辑注释：这些 prompt 负责指导 LLM 从对话中抽取普通记忆或过程性记忆，是 infer 模式的核心输入。
from mem0.configs.prompts import (
    ADDITIVE_EXTRACTION_PROMPT,
    AGENT_CONTEXT_SUFFIX,
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
    generate_additive_extraction_prompt,
)
# 逻辑注释：使用 mem0 自己的 ValidationError 可以携带 error_code、details、suggestion，错误信息更适合 SDK 用户。
from mem0.exceptions import ValidationError as Mem0ValidationError
# 逻辑注释：MemoryBase 提供同步/异步 Memory 的共同抽象，让两套实现保持同一个接口风格。
from mem0.memory.base import MemoryBase
from mem0.memory.setup import mem0_dir, setup_config
# 逻辑注释：SQLiteManager 负责保存历史消息和记忆变更历史，向量库只负责语义检索。
from mem0.memory.storage import SQLiteManager
# 逻辑注释：遥测开关和事件采集用于观察 SDK 调用情况，同时后面会对敏感配置做脱敏。
from mem0.memory.telemetry import MEM0_TELEMETRY, capture_event
# 逻辑注释：这些工具函数处理消息解析、JSON 提取、代码块清理和遥测过滤，是 LLM 输出鲁棒性的辅助层。
from mem0.memory.utils import (
    extract_json,
    parse_messages,
    parse_vision_messages,
    process_telemetry_filters,
    remove_code_blocks,
)
# 逻辑注释：实体抽取用于建立 entity → memory 的反向索引，后续搜索时能做实体增强排序。
from mem0.utils.entity_extraction import extract_entities, extract_entities_batch
# 逻辑注释：Factory 根据配置动态创建 embedder、LLM、reranker、vector store，避免 Memory 直接绑定某个 provider。
from mem0.utils.factory import (
    EmbedderFactory,
    LlmFactory,
    RerankerFactory,
    VectorStoreFactory,
)
# 逻辑注释：词形归一化结果会进入 payload，供 BM25/关键词检索使用。
from mem0.utils.lemmatization import lemmatize_for_bm25
# 逻辑注释：检索阶段会融合语义分数、BM25 分数和实体增强分数，这里导入对应权重和排序函数。
from mem0.utils.scoring import (
    ENTITY_BOOST_WEIGHT,
    get_bm25_params,
    normalize_bm25,
    score_and_rank,
)

# Suppress SWIG deprecation warnings globally
# 逻辑注释：全局压制 SWIG 相关弃用告警，避免底层依赖的噪声影响 SDK 使用者日志。
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")

# Initialize logger early for util functions
# 逻辑注释：提前创建模块级 logger，后续 helper 和类方法都复用同一个日志入口。
logger = logging.getLogger(__name__)


# Fields that hold runtime auth/connection objects and must be preserved.
# These are non-serializable objects (e.g. AWSV4SignerAuth, RequestsHttpConnection)
# needed by clients like OpenSearch — not sensitive strings to redact.
# 逻辑注释：这些字段虽然可能叫 auth/connection，但实际是运行时连接对象，复制配置时必须保留，否则客户端会失效。
_RUNTIME_FIELDS = frozenset({
    "http_auth",
    "auth",
    "connection_class",
    "ssl_context",
})

# Fields that are known to contain sensitive secrets and must be redacted.
# 逻辑注释：这些字段名被视为确定的密钥/凭证，进入遥测或克隆兜底时应清空，避免泄露。
_SENSITIVE_FIELDS_EXACT = frozenset({
    "api_key",
    "secret_key",
    "private_key",
    "access_key",
    "password",
    "credentials",
    "credential",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "auth_token",
    "session_token",
    "client_secret",
    "auth_client_secret",
    "azure_client_secret",
    "service_account_json",
    "aws_session_token",
})

# Suffixes that indicate a field likely holds a secret value.
# 逻辑注释：后缀规则用来兜住 db_password、auth_token 这类不在精确列表里的敏感字段。
_SENSITIVE_SUFFIXES = (
    "_password",
    "_secret",
    "_token",
    "_credential",
    "_credentials",
)

# Entity parameters that must be passed via filters, not top-level kwargs
# 逻辑注释：实体作用域参数集中定义，后面用同一份集合做校验和 filters 构造，减少规则漂移。
ENTITY_PARAMS = frozenset({"user_id", "agent_id", "run_id"})


# 逻辑注释：统一拒绝旧式顶层 user_id/agent_id/run_id 参数，强制调用方走 filters，避免同一个 API 出现两套作用域入口。
def _reject_top_level_entity_params(kwargs: Dict[str, Any], method_name: str) -> None:
    """Reject top-level entity parameters - must use filters instead."""
    # 逻辑注释：取交集能一次找出所有误传到顶层的实体作用域参数，错误提示也更完整。
    invalid_keys = ENTITY_PARAMS & set(kwargs.keys())
    # 逻辑注释：只要发现旧式顶层作用域参数就立即拒绝，避免和 filters 里的作用域发生冲突。
    if invalid_keys:
        # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
        raise ValueError(
            f"Top-level entity parameters {invalid_keys} are not supported in {method_name}(). "
            f"Use filters={{'user_id': '...'}} instead."
        )


# 逻辑注释：先清洗再校验实体 ID，保证后续 metadata/filter 使用的是稳定、无空白歧义的作用域键。
def _validate_and_trim_entity_id(value: Optional[str], name: str) -> Optional[str]:
    """
    Validates and normalizes an entity ID.
    - Trims leading/trailing whitespace
    - Rejects empty or whitespace-only strings
    - Rejects strings containing internal whitespace

    Args:
        value: The entity ID value to validate
        name: The parameter name (for error messages)

    Returns:
        The trimmed entity ID, or None if input is None

    Raises:
        ValueError: If entity ID is invalid
    """
    # 逻辑注释：None 表示调用方没有传这个 ID，不参与后续作用域过滤。
    if value is None:
        # 逻辑注释：没有可用结果时显式返回 None，让调用方能区分“没找到”和异常。
        return None
    # 逻辑注释：先去掉首尾空白，既允许用户输入有轻微格式问题，也避免把空格算进 ID。
    trimmed = value.strip()
    # 逻辑注释：去空白后为空说明没有真正的标识符，继续存储会导致作用域不可控。
    if trimmed == "":
        # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
        raise ValueError(
            f"Invalid {name}: cannot be empty or whitespace-only. Provide a valid identifier."
        )
    # 逻辑注释：内部空白会让一个 ID 看起来像多个 token，因此直接禁止，减少过滤歧义。
    if any(c.isspace() for c in trimmed):
        # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
        raise ValueError(
            f"Invalid {name}: cannot contain whitespace. Provide a valid identifier without spaces."
        )
    return trimmed


# 逻辑注释：在真正查询前集中校验阈值和返回数量，避免非法参数传到向量库后才报更难定位的错误。
def _validate_search_params(threshold: Optional[float] = None, top_k: Optional[int] = None) -> None:
    """
    Validates search parameters.

    Args:
        threshold: Similarity threshold (must be between 0 and 1)
        top_k: Number of results to return (must be non-negative integer)

    Raises:
        ValueError: If threshold or top_k are invalid
    """
    # 逻辑注释：threshold 可选；只有调用方传了值才需要校验类型和范围。
    if threshold is not None:
        # 逻辑注释：阈值必须能参与数值比较，字符串等类型不能传进排序逻辑。
        if not isinstance(threshold, (int, float)):
            # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
            raise ValueError("threshold must be a valid number")
        # 逻辑注释：相似度阈值按归一化分数处理，所以合法范围固定在 0 到 1。
        if threshold < 0 or threshold > 1:
            # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
            raise ValueError(
                f"Invalid threshold: {threshold}. Must be between 0 and 1 (inclusive)."
            )
    # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
    if top_k is not None:
        # 逻辑注释：top_k 控制返回条数，必须是真正整数；bool 虽是 int 子类但语义不对，所以排除。
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
            raise ValueError("top_k must be a valid integer")
        # 逻辑注释：负数返回条数没有意义，提前拒绝能避免向量库实现差异。
        if top_k < 0:
            # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
            raise ValueError(
                f"Invalid top_k: {top_k}. Must be a non-negative integer."
            )


# 逻辑注释：按“运行时对象优先保留、敏感字段再脱敏”的顺序判断字段，兼顾可用性和遥测安全。
def _is_sensitive_field(field_name: str) -> bool:
    """Check if a field should be redacted for telemetry safety.

    Uses a layered approach:
    1. Runtime fields (allowlist) — always preserved, highest priority.
    2. Exact deny list — known secret field names.
    3. Suffix deny list — catches patterns like db_password, auth_secret, etc.
    """
    # 逻辑注释：字段名统一小写并去空白，保证敏感字段匹配不受大小写或格式影响。
    name = field_name.lower().strip()
    # 逻辑注释：运行时连接对象优先放行，即使名字里有 auth，也不能被误判为要脱敏的字符串密钥。
    if name in _RUNTIME_FIELDS:
        return False
    # 逻辑注释：精确命中的密钥字段直接判定为敏感，避免进入遥测或日志。
    if name in _SENSITIVE_FIELDS_EXACT:
        return True
    return any(name.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES)


# 逻辑注释：复制配置时兼容不可序列化对象：先尝试 deepcopy，失败后退化为字典重建，同时保留连接对象、脱敏密钥。
def _safe_deepcopy_config(config):
    """Safely deepcopy config, falling back to dict-based cloning for non-serializable objects."""
    try:
        return deepcopy(config)
    # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
    except Exception as e:
        logger.debug(f"Deepcopy failed, using dict-based cloning: {e}")

        # 逻辑注释：记录原配置类型，后面重建时尽量返回同类对象，而不是裸 dict。
        config_class = type(config)

        # 逻辑注释：Pydantic v2 模型优先用 model_dump 导出，字段处理比直接读 __dict__ 更规范。
        if hasattr(config, "model_dump"):
            try:
                clone_dict = config.model_dump()
            # 逻辑注释：这里进入兜底路径，通常会从批量操作退化为逐条处理，提升整体成功率。
            except Exception:
                # 逻辑注释：无法走模型导出时，退回到对象属性字典，至少保留可见配置字段。
                clone_dict = dict(config.__dict__)
        else:
            # 逻辑注释：无法走模型导出时，退回到对象属性字典，至少保留可见配置字段。
            clone_dict = dict(config.__dict__)

        # Restore runtime fields, redact sensitive ones
        # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
        for field_name in list(clone_dict.keys()):
            # 逻辑注释：运行时连接对象优先放行，即使名字里有 auth，也不能被误判为要脱敏的字符串密钥。
            if field_name in _RUNTIME_FIELDS and hasattr(config, field_name):
                clone_dict[field_name] = getattr(config, field_name)
            # 逻辑注释：这是对前面判断的补充分支，用来兼容另一种输入/配置形态。
            elif _is_sensitive_field(field_name):
                clone_dict[field_name] = None

        try:
            return config_class(**clone_dict)
        # 逻辑注释：这里进入兜底路径，通常会从批量操作退化为逐条处理，提升整体成功率。
        except Exception:
            logger.debug("Config reconstruction failed, returning shallow dict clone")
            return type("Config", (), clone_dict)()


# 逻辑注释：只把带时区的 ISO 时间统一成 UTC；没有时区或无法解析的字符串保持原样，避免误改调用方语义。
def _normalize_iso_timestamp_to_utc(timestamp: Optional[str]) -> Optional[str]:
    """Normalize timezone-aware ISO timestamps to UTC without rewriting naive values."""
    # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
    if not timestamp:
        return timestamp
    try:
        # 逻辑注释：只解析标准 ISO 字符串；解析失败说明输入可能是自定义格式，应保持原值。
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp
    # 逻辑注释：没有时区信息的时间不能安全换算成 UTC，所以这里不做假设。
    if parsed.tzinfo is None:
        return timestamp
    return parsed.astimezone(timezone.utc).isoformat()


# 逻辑注释：把一次调用里的作用域信息拆成“写入 metadata 模板”和“查询 filters”，这样新增和检索能使用一致的会话边界。
def _build_filters_and_metadata(
    *,  # Enforce keyword-only arguments
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    actor_id: Optional[str] = None,  # For query-time filtering
    input_metadata: Optional[Dict[str, Any]] = None,
    input_filters: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Constructs metadata for storage and filters for querying based on session and actor identifiers.

    This helper supports multiple session identifiers (`user_id`, `agent_id`, and/or `run_id`)
    for flexible session scoping and optionally narrows queries to a specific `actor_id`. It returns two dicts:

    1. `base_metadata_template`: Used as a template for metadata when storing new memories.
       It includes all provided session identifier(s) and any `input_metadata`.
    2. `effective_query_filters`: Used for querying existing memories. It includes all
       provided session identifier(s), any `input_filters`, and a resolved actor
       identifier for targeted filtering if specified by any actor-related inputs.

    Actor filtering precedence: explicit `actor_id` arg → `filters["actor_id"]`
    This resolved actor ID is used for querying but is not added to `base_metadata_template`,
    as the actor for storage is typically derived from message content at a later stage.

    Args:
        user_id (Optional[str]): User identifier, for session scoping.
        agent_id (Optional[str]): Agent identifier, for session scoping.
        run_id (Optional[str]): Run identifier, for session scoping.
        actor_id (Optional[str]): Explicit actor identifier, used as a potential source for
            actor-specific filtering. See actor resolution precedence in the main description.
        input_metadata (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session identifiers for the storage metadata template. Defaults to an empty dict.
        input_filters (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session and actor identifiers for query filters. Defaults to an empty dict.

    Returns:
        tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - base_metadata_template (Dict[str, Any]): Metadata template for storing memories,
              scoped to the provided session(s).
            - effective_query_filters (Dict[str, Any]): Filters for querying memories,
              scoped to the provided session(s) and potentially a resolved actor.
    """

    # 逻辑注释：复制外部 metadata，后面会追加作用域字段；复制能避免修改调用方传入的原对象。
    base_metadata_template = deepcopy(input_metadata) if input_metadata else {}
    # 逻辑注释：查询 filters 也复制一份，保证内部补充 actor/session 条件时不会污染调用方数据。
    effective_query_filters = deepcopy(input_filters) if input_filters else {}

    # ---------- validate and add all provided session ids ----------
    # 逻辑注释：记录实际提供了哪些 session id，用于最后判断是否有作用域边界。
    session_ids_provided = []

    # Validate and trim entity IDs
    # 逻辑注释：所有 session id 在进入 metadata/filter 前统一清洗，保证存储和查询使用同一种规范值。
    user_id = _validate_and_trim_entity_id(user_id, "user_id")
    agent_id = _validate_and_trim_entity_id(agent_id, "agent_id")
    # 逻辑注释：所有 session id 在进入 metadata/filter 前统一清洗，保证存储和查询使用同一种规范值。
    run_id = _validate_and_trim_entity_id(run_id, "run_id")

    # 逻辑注释：有 user_id 时同时写入 metadata 和 filters，新增记忆和查询旧记忆会落在同一个用户作用域。
    if user_id:
        base_metadata_template["user_id"] = user_id
        effective_query_filters["user_id"] = user_id
        session_ids_provided.append("user_id")

    # 逻辑注释：agent_id 也参与存储和过滤，支持按 agent 维度隔离记忆。
    if agent_id:
        base_metadata_template["agent_id"] = agent_id
        effective_query_filters["agent_id"] = agent_id
        session_ids_provided.append("agent_id")

    # 逻辑注释：run_id 用于一次运行/会话级别的隔离，适合临时任务或批处理场景。
    if run_id:
        base_metadata_template["run_id"] = run_id
        effective_query_filters["run_id"] = run_id
        session_ids_provided.append("run_id")

    # 逻辑注释：没有任何 session id 就没有记忆边界，直接拒绝，避免把所有用户/agent 的记忆混在一起。
    if not session_ids_provided:
        # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
        raise Mem0ValidationError(
            message="At least one of 'user_id', 'agent_id', or 'run_id' must be provided.",
            error_code="VALIDATION_001",
            details={"provided_ids": {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}},
            suggestion="Please provide at least one identifier to scope the memory operation."
        )

    # ---------- optional actor filter ----------
    # 逻辑注释：actor 过滤优先使用显式参数，其次沿用 filters 中已有 actor_id，保证调用方可以按说话人缩小查询。
    resolved_actor_id = actor_id or effective_query_filters.get("actor_id")
    # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
    if resolved_actor_id:
        # 逻辑注释：actor_id 只加入查询 filters，不加入写入模板，因为写入时 actor 通常来自具体 message。
        effective_query_filters["actor_id"] = resolved_actor_id

    return base_metadata_template, effective_query_filters


# 逻辑注释：把多个实体 ID 排序拼成稳定字符串，用作历史消息的会话键，避免字典顺序造成不同 key。
def _build_session_scope(filters):
    """Build deterministic session scope string from entity IDs."""
    parts = []
    # 逻辑注释：按固定顺序拼接作用域字段，使同一组 filters 永远得到同一个 session_scope。
    for key in sorted(["user_id", "agent_id", "run_id"]):
        val = filters.get(key)
        # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
        if val:
            # 逻辑注释：只把有值的实体 ID 放入 scope，避免 None/空值影响历史消息分组。
            parts.append(f"{key}={val}")
    return "&".join(parts)


# 逻辑注释：模块加载时先执行配置初始化，确保默认目录/配置文件等运行前置条件已经准备好。
setup_config()
# 逻辑注释：提前创建模块级 logger，后续 helper 和类方法都复用同一个日志入口。
logger = logging.getLogger(__name__)


# 逻辑注释：同步版 Memory 实现，对外暴露增删改查和搜索；内部负责 LLM 抽取、向量存储、历史记录和实体索引。
class Memory(MemoryBase):
    # 逻辑注释：初始化 Memory 实例需要把配置里的各类 provider 变成真实客户端，并准备向量库、LLM、SQLite 历史库和可选 reranker。
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        # 逻辑注释：把配置保存到实例上，后续所有 provider 初始化、路径和版本信息都从这里读取。
        self.config = config

        # 逻辑注释：根据配置创建 embedding 模型；Memory 不关心具体 provider，只依赖统一 embed 接口。
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        # 逻辑注释：创建向量存储后，记忆文本的向量和 payload 都会通过它进行插入、查询、更新和删除。
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        # 逻辑注释：创建 LLM 客户端，infer/procedural 模式会用它从对话中抽取或总结记忆。
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        # 逻辑注释：SQLite 用来保存消息上下文和变更历史，和向量库形成“语义索引 + 审计记录”的双存储结构。
        self.db = SQLiteManager(self.config.history_db_path)
        # 逻辑注释：保存主记忆 collection 名，实体库会基于这个名字派生出独立 collection。
        self.collection_name = self.config.vector_store.config.collection_name
        # 逻辑注释：保存 API 版本，遥测事件会带上它，便于区分不同版本的行为。
        self.api_version = self.config.version
        # 逻辑注释：全局自定义指令会在 LLM 抽取记忆时作为默认额外要求。
        self.custom_instructions = self.config.custom_instructions

        # Initialize reranker if configured
        # 逻辑注释：reranker 默认不启用；只有配置显式提供时才创建，避免额外依赖和成本。
        self.reranker = None
        # 逻辑注释：检测到 reranker 配置才初始化二次排序器，搜索时也会按开关选择是否使用。
        if config.reranker:
            # 逻辑注释：通过工厂创建 reranker，使不同重排模型可以用同一套 Memory 搜索逻辑接入。
            self.reranker = RerankerFactory.create(
                config.reranker.provider,
                config.reranker.config
            )

        # Entity store is initialized lazily on first use
        # 逻辑注释：实体库先置空，后面通过 property 懒加载，避免不使用实体能力时创建多余向量库。
        self._entity_store = None

        # 逻辑注释：只有遥测开关打开时才准备遥测专用向量库，普通运行不会产生额外存储开销。
        if MEM0_TELEMETRY:
            # Create telemetry config manually to avoid deepcopy issues with thread locks
            # 逻辑注释：先组装一份遥测用配置字典，后面会覆盖 collection/path 并避免携带原业务 collection。
            telemetry_config_dict = {}
            # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
            if hasattr(self.config.vector_store.config, 'model_dump'):
                # For pydantic models
                # 逻辑注释：Pydantic 配置可直接导出为 dict，便于安全地修改遥测 collection。
                telemetry_config_dict = self.config.vector_store.config.model_dump()
            else:
                # For other objects, manually copy common attributes
                # 逻辑注释：非 Pydantic 配置只复制常见连接字段，避免把整个复杂对象原样塞进遥测配置。
                for attr in ['host', 'port', 'path', 'api_key', 'index_name', 'dimension', 'metric']:
                    # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                    if hasattr(self.config.vector_store.config, attr):
                        telemetry_config_dict[attr] = getattr(self.config.vector_store.config, attr)

            # Override collection name for telemetry
            # 逻辑注释：遥测数据写入独立 collection，避免和用户真实记忆混在一起。
            telemetry_config_dict['collection_name'] = "mem0migrations"

            # Set path for file-based vector stores
            telemetry_config = _safe_deepcopy_config(self.config.vector_store.config)
            # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
            if self.config.vector_store.provider in ["faiss", "qdrant"]:
                # 逻辑注释：文件型向量库需要独立目录，按 provider 名构造迁移/遥测存储路径。
                provider_path = f"migrations_{self.config.vector_store.provider}"
                telemetry_config_dict['path'] = os.path.join(mem0_dir, provider_path)
                # 逻辑注释：目录不存在时提前创建，避免初始化本地向量库时因路径缺失失败。
                os.makedirs(telemetry_config_dict['path'], exist_ok=True)

            # Create the config object using the same class as the original
            telemetry_config = self.config.vector_store.config.__class__(**telemetry_config_dict)
            # 逻辑注释：创建遥测专用向量库客户端，后续 capture_event 可复用这个存储。
            self._telemetry_vector_store = VectorStoreFactory.create(
                self.config.vector_store.provider, telemetry_config
            )
        # 逻辑注释：初始化结束后记录一次 init 事件，并标明 sync/async，便于观测两种实现的使用情况。
        capture_event("mem0.init", self, {"sync_type": "sync"})

    # 逻辑注释：把这个方法暴露成只读属性，调用方访问时像字段一样自然，同时内部仍可做懒加载。
    @property
    # 逻辑注释：实体向量库采用懒加载：只有真正需要实体链接/增强检索时才创建，减少初始化成本和嵌入式向量库锁冲突。
    def entity_store(self):
        """Lazily initialize entity store on first use."""
        # 逻辑注释：第一次访问实体库才进入初始化，后续直接复用已经创建的实例。
        if self._entity_store is None:
            # 逻辑注释：实体库复用主向量库配置的副本，避免直接修改主记忆 collection 配置。
            entity_config = _safe_deepcopy_config(self.config.vector_store.config)
            # 逻辑注释：保存主记忆 collection 名，实体库会基于这个名字派生出独立 collection。
            entity_collection = f"{self.collection_name}_entities"
            # Set collection name on the cloned config
            # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
            if hasattr(entity_config, 'collection_name'):
                # 逻辑注释：把副本的 collection 改成实体 collection，后续实体向量不会写入主记忆库。
                entity_config.collection_name = entity_collection
            # 逻辑注释：这是对前面判断的补充分支，用来兼容另一种输入/配置形态。
            elif isinstance(entity_config, dict):
                # 逻辑注释：把副本的 collection 改成实体 collection，后续实体向量不会写入主记忆库。
                entity_config['collection_name'] = entity_collection
            # For Qdrant, share the existing client to avoid RocksDB lock contention
            # when using embedded mode (path=...). QdrantConfig.client takes precedence
            # over host/port/path.
            # 逻辑注释：Qdrant 嵌入式模式下共享已有 client，避免同一路径被多个 RocksDB 实例同时打开导致锁冲突。
            if self.config.vector_store.provider == "qdrant" and hasattr(self.vector_store, "client"):
                # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                if hasattr(entity_config, "client"):
                    # 逻辑注释：把主向量库的 client 注入实体配置，实体库和主库共享同一个底层连接。
                    entity_config.client = self.vector_store.client
                # 逻辑注释：这是对前面判断的补充分支，用来兼容另一种输入/配置形态。
                elif isinstance(entity_config, dict):
                    # 逻辑注释：把主向量库的 client 注入实体配置，实体库和主库共享同一个底层连接。
                    entity_config["client"] = self.vector_store.client
            self._entity_store = VectorStoreFactory.create(
                self.config.vector_store.provider, entity_config
            )
        return self._entity_store

    # 逻辑注释：把抽取出的实体写入实体库；相似实体复用并追加 memory_id，新实体才新建，形成实体到记忆的反向索引。
    def _upsert_entity(self, entity_text, entity_type, memory_id, filters):
        """Upsert an entity into the entity store, linking it to a memory."""
        try:
            # 逻辑注释：实体也需要单独向量化，才能在实体库里用相似度判断是否已有同一实体。
            entity_embedding = self.embedding_model.embed(entity_text, "add")
            # 逻辑注释：实体检索只使用 session 级作用域字段，保证实体链接不会跨用户/agent/run 串数据。
            search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}

            # 逻辑注释：先在实体库里找最相近的实体，命中足够高时复用节点而不是重复创建。
            existing = self.entity_store.search(
                query=entity_text,
                vectors=entity_embedding,
                top_k=1,
                filters=search_filters,
            )

            # 逻辑注释：0.95 作为近似同实体阈值，只有非常相近时才合并，降低误把不同实体合并的风险。
            if existing and existing[0].score >= 0.95:
                # Update existing entity's linked_memory_ids
                match = existing[0]
                payload = match.payload or {}
                # 逻辑注释：实体 payload 里维护反向链接列表，用来知道这个实体关联了哪些记忆。
                linked_ids = payload.get("linked_memory_ids", [])
                # 逻辑注释：只有新记忆 ID 不在列表里才追加，避免重复链接导致后续 boost 被放大。
                if memory_id not in linked_ids:
                    linked_ids.append(memory_id)
                    payload["linked_memory_ids"] = linked_ids
                    self.entity_store.update(
                        vector_id=match.id,
                        vector=None,
                        payload=payload,
                    )
            else:
                # Create new entity
                # 逻辑注释：新实体需要独立 ID，和 memory_id 分开管理，便于实体库单独增删改查。
                entity_id = str(uuid.uuid4())
                # 逻辑注释：实体 payload 同时保存实体文本、类型、关联记忆和 session 过滤字段，后续搜索/清理都依赖这些信息。
                entity_payload = {
                    "data": entity_text,
                    "entity_type": entity_type,
                    "linked_memory_ids": [memory_id],
                    **{k: v for k, v in search_filters.items()},
                }
                # 逻辑注释：把新实体向量和 payload 写入实体库，建立实体索引。
                self.entity_store.insert(
                    vectors=[entity_embedding],
                    ids=[entity_id],
                    payloads=[entity_payload],
                )
        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
        except Exception as e:
            # 逻辑注释：实体索引失败不应影响主记忆写入，所以这里只记录警告而不是抛出。
            logger.warning(f"Entity upsert failed for '{entity_text}': {e}")

    # 逻辑注释：删除或更新记忆后清理实体索引：从实体的 linked_memory_ids 中移除该 memory_id，孤立实体直接删除。
    def _remove_memory_from_entity_store(self, memory_id, filters):
        """Strip `memory_id` from every entity record scoped to `filters`.

        For each entity whose `linked_memory_ids` contains `memory_id`:
          - remove the id; if the list becomes empty, delete the entity record.
          - otherwise re-embed the entity text and update the payload
            (the vector store's update() requires a vector).

        No-op if the entity store has never been initialized in this process.
        Errors on individual entities are swallowed at debug level; outer
        failures are swallowed at warning level so the primary delete/update
        path is never broken by entity cleanup.
        """
        # 逻辑注释：第一次访问实体库才进入初始化，后续直接复用已经创建的实例。
        if self._entity_store is None:
            return
        # 逻辑注释：实体检索只使用 session 级作用域字段，保证实体链接不会跨用户/agent/run 串数据。
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        try:
            # 逻辑注释：清理时先列出当前作用域下的实体，再逐个检查是否链接了待删除/更新的记忆。
            listed = self.entity_store.list(filters=search_filters, top_k=10000)
            # 逻辑注释：不同向量库 list 返回格式不一致，这里兼容嵌套列表和扁平列表两种结构。
            rows = listed[0] if isinstance(listed, (list, tuple)) and listed and isinstance(listed[0], list) else listed
            # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
            for row in rows or []:
                try:
                    # 逻辑注释：实体行可能来自不同实现，统一用 getattr 安全取 payload。
                    payload = getattr(row, "payload", None) or {}
                    linked = payload.get("linked_memory_ids", [])
                    # 逻辑注释：linked_memory_ids 不是列表或不包含目标 memory_id 时，说明这条实体不需要处理。
                    if not isinstance(linked, list) or memory_id not in linked:
                        continue
                    # 逻辑注释：构造移除目标 memory_id 后的新链接列表，用于判断实体是否还被其他记忆引用。
                    remaining = [mid for mid in linked if mid != memory_id]
                    # 逻辑注释：没有任何记忆再引用该实体时，实体节点已经孤立，可以删除。
                    if not remaining:
                        try:
                            # 逻辑注释：删除孤立实体，避免实体库里留下无法增强任何记忆的脏数据。
                            self.entity_store.delete(vector_id=row.id)
                        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
                        except Exception as e:
                            logger.debug(f"Entity delete failed for id={row.id}: {e}")
                    else:
                        # 逻辑注释：实体仍被其他记忆引用时，需要取出实体文本重新生成向量以满足 update 接口要求。
                        entity_text = payload.get("data")
                        # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                        if not isinstance(entity_text, str) or not entity_text:
                            logger.debug(f"Entity id={row.id} missing 'data'; skipping update during cleanup")
                            continue
                        try:
                            # 逻辑注释：有些向量库 update 要求同时传 vector，所以这里即使只改 payload 也重新计算实体向量。
                            vec = self.embedding_model.embed(entity_text, "update")
                        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
                        except Exception as e:
                            logger.debug(f"Entity re-embed failed for '{entity_text}': {e}")
                            continue
                        # 逻辑注释：保留实体原有信息，只替换 linked_memory_ids，避免丢失 entity_type/session 等字段。
                        new_payload = {**payload, "linked_memory_ids": remaining}
                        try:
                            self.entity_store.update(
                                vector_id=row.id,
                                vector=vec,
                                payload=new_payload,
                            )
                        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
                        except Exception as e:
                            logger.debug(f"Entity update failed for id={row.id}: {e}")
                # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
                except Exception as e:
                    logger.debug(f"Entity cleanup error: {e}")
        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
        except Exception as e:
            logger.warning(f"Entity store cleanup failed for memory_id={memory_id}: {e}")

    # 逻辑注释：从单条记忆文本中抽取实体并建立链接，主要用于 update 后把新文本重新挂到实体索引上。
    def _link_entities_for_memory(self, memory_id, text, filters):
        """Extract entities from `text` and link them to `memory_id` in the
        entity store, scoped to `filters`. Simpler single-memory variant of
        Phase 7 in add(): per-entity search-then-update-or-insert via the
        existing `_upsert_entity` helper. Non-fatal on any failure.
        """
        try:
            # 逻辑注释：从记忆文本抽取实体，只有抽到实体才需要进入实体链接流程。
            entities = extract_entities(text)
            # 逻辑注释：没有实体时直接返回，避免空循环和不必要的向量库访问。
            if not entities:
                return
            # 逻辑注释：用集合在单条文本内去重，避免同一个实体重复 upsert。
            seen = set()
            # 逻辑注释：逐个处理抽取出的实体，把每个实体都链接到当前记忆。
            for entity_type, entity_text in entities:
                # 逻辑注释：实体去重用小写+去空白后的规范 key，降低大小写和首尾空格带来的重复。
                key = entity_text.strip().lower()
                # 逻辑注释：空实体或已处理实体都跳过，保持实体链接的唯一性。
                if not key or key in seen:
                    continue
                seen.add(key)
                try:
                    # 逻辑注释：每个有效实体交给 upsert，内部决定复用旧实体还是新建实体。
                    self._upsert_entity(entity_text, entity_type, memory_id, filters)
                # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
                except Exception as e:
                    logger.debug(f"Entity link failed for '{entity_text}': {e}")
        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
        except Exception as e:
            logger.warning(f"Entity linking failed for memory_id={memory_id}: {e}")

    # 逻辑注释：类方法不依赖已有实例，适合作为另一种构造入口。
    @classmethod
    # 逻辑注释：从普通字典创建配置对象，再交给构造函数；这里把外部配置入口和类初始化解耦。
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config_dict)
        # 逻辑注释：配置校验错误需要原样抛出，调用方才能看到 Pydantic 提供的具体字段问题。
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    # 逻辑注释：静态方法不依赖实例状态，这里用于纯配置处理/转换逻辑。
    @staticmethod
    # 逻辑注释：当前只是透传配置，保留这个钩子方便以后在构造 MemoryConfig 前做兼容性转换。
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return config_dict
        # 逻辑注释：配置校验错误需要原样抛出，调用方才能看到 Pydantic 提供的具体字段问题。
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    # 逻辑注释：根据是否有 agent_id 且消息里是否出现 assistant，决定记忆应偏向 agent 视角还是 user 视角。
    def _should_use_agent_memory_extraction(self, messages, metadata):
        """Determine whether to use agent memory extraction based on the logic:
        - If agent_id is present and messages contain assistant role -> True
        - Otherwise -> False

        Args:
            messages: List of message dictionaries
            metadata: Metadata containing user_id, agent_id, etc.

        Returns:
            bool: True if should use agent memory extraction, False for user memory extraction
        """
        # Check if agent_id is present in metadata
        has_agent_id = metadata.get("agent_id") is not None

        # Check if there are assistant role messages
        has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)

        # Use agent memory extraction if agent_id is present and there are assistant messages
        return has_agent_id and has_assistant_messages

    # 逻辑注释：新增记忆的公共入口：先确定作用域和输入格式，再按 procedural/raw/infer 三条路径分流。
    def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """
        Create a new memory.

        Adds new memories scoped to a single session id (e.g. `user_id`, `agent_id`, or `run_id`). One of those ids is required.

        Args:
            messages (str or List[Dict[str, str]]): The message content or list of messages
                (e.g., `[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]`)
                to be processed and stored.
            user_id (str, optional): ID of the user creating the memory. Defaults to None.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            infer (bool, optional): If True (default), an LLM is used to extract key facts from
                'messages' and decide whether to add, update, or delete related memories.
                If False, 'messages' are added as raw memories directly.
            memory_type (str, optional): Specifies the type of memory. Currently, only
                `MemoryType.PROCEDURAL.value` ("procedural_memory") is explicitly handled for
                creating procedural memories (typically requires 'agent_id'). Otherwise, memories
                are treated as general conversational/factual memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.


        Returns:
            dict: A dictionary containing the result of the memory addition operation, typically
                  including a list of memory items affected (added, updated) under a "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "event": "ADD"}]}`

        Raises:
            Mem0ValidationError: If input validation fails (invalid memory_type, messages format, etc.).
            VectorStoreError: If vector store operations fail.
            EmbeddingError: If embedding generation fails.
            LLMError: If LLM operations fail.
            DatabaseError: If database operations fail.
        """

        # 逻辑注释：新增记忆前先统一构造 metadata 和 filters，保证写入、检索旧记忆和历史上下文使用同一作用域。
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
        )

        # 逻辑注释：当前只额外支持 procedural_memory；其他 memory_type 会让调用方误以为有别的处理逻辑，因此拒绝。
        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
            raise Mem0ValidationError(
                message=f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories.",
                error_code="VALIDATION_002",
                details={"provided_type": memory_type, "valid_type": MemoryType.PROCEDURAL.value},
                suggestion=f"Use '{MemoryType.PROCEDURAL.value}' to create procedural memories."
            )

        # 逻辑注释：单字符串输入被包装成 user 消息，方便后续统一按消息列表处理。
        if isinstance(messages, str):
            # 逻辑注释：把简写输入转换成标准 role/content 结构，后面的解析和 LLM prompt 不需要再分支处理。
            messages = [{"role": "user", "content": messages}]

        # 逻辑注释：这是对前面判断的补充分支，用来兼容另一种输入/配置形态。
        elif isinstance(messages, dict):
            # 逻辑注释：这里保存该阶段的中间结果，供后续抽取、去重、批量写入或返回结果复用。
            messages = [messages]

        # 逻辑注释：这是对前面判断的补充分支，用来兼容另一种输入/配置形态。
        elif not isinstance(messages, list):
            # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
            raise Mem0ValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries."
            )

        # 逻辑注释：过程性记忆要求 agent 作用域，因为它描述的是 agent 的行为流程，而不是普通用户事实。
        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            # 逻辑注释：命中过程序记忆分支后直接生成并存储 procedural memory，不再走普通事实抽取流程。
            results = self._create_procedural_memory(messages, metadata=processed_metadata, prompt=prompt)
            return results

        # 逻辑注释：如果配置启用视觉能力，消息解析要保留/处理图像内容，让 LLM 能理解多模态输入。
        if self.config.llm.config.get("enable_vision"):
            # 逻辑注释：视觉消息先规范化为 LLM 可接收格式；未启用视觉时仍会做基础消息清洗。
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            # 逻辑注释：视觉消息先规范化为 LLM 可接收格式；未启用视觉时仍会做基础消息清洗。
            messages = parse_vision_messages(messages)

        # 逻辑注释：普通记忆最终交给向量写入流程处理，add 只负责入口校验和分流。
        vector_store_result = self._add_to_vector_store(messages, processed_metadata, effective_filters, infer, prompt=prompt)
        # 逻辑注释：对外统一用 results 包一层，保持 add/search/get_all 等接口返回结构一致。
        return {"results": vector_store_result}

    # 逻辑注释：真正写入向量库的核心流程：raw 模式直接存，infer 模式走“检索旧记忆→LLM 抽取→去重→批量入库→实体链接”。
    def _add_to_vector_store(self, messages, metadata, filters, infer, prompt=None):
        # 逻辑注释：关闭 infer 时不调用 LLM 抽取事实，而是把非 system 消息原样作为记忆写入。
        if not infer:
            # 逻辑注释：收集本次成功写入的记忆，最后按 API 约定返回给调用方。
            returned_memories = []
            # 逻辑注释：raw 模式逐条处理消息，每条有效消息都会成为一条独立记忆。
            for message_dict in messages:
                # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                if (
                    # 逻辑注释：raw 模式仍要校验每条消息至少有 role/content，否则无法形成可解释的记忆记录。
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    # 逻辑注释：单条消息格式错误只跳过并告警，不让一个坏消息导致整批写入失败。
                    logger.warning(f"Skipping invalid message format: {message_dict}")
                    continue

                # 逻辑注释：system 消息通常是指令/上下文，不应作为用户事实或对话记忆存储。
                if message_dict["role"] == "system":
                    continue

                # 逻辑注释：每条消息复制一份 metadata，避免给其中一条消息加 role/actor 时影响其他消息。
                per_msg_meta = deepcopy(metadata)
                # 逻辑注释：把原消息角色写入 metadata，后续读取/搜索时可以知道记忆来自 user 还是 assistant。
                per_msg_meta["role"] = message_dict["role"]

                # 逻辑注释：如果消息带 name，就把它作为 actor_id，支持多人/多角色对话里的说话人过滤。
                actor_name = message_dict.get("name")
                # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                if actor_name:
                    # 逻辑注释：actor_id 写入 metadata 后，后续可以按具体说话人查找或清理记忆。
                    per_msg_meta["actor_id"] = actor_name

                # 逻辑注释：raw 模式把原始 content 当成记忆正文，不做 LLM 改写。
                msg_content = message_dict["content"]
                # 逻辑注释：写向量库前先把文本转成 embedding，后续语义搜索才能召回这条记忆。
                msg_embeddings = self.embedding_model.embed(msg_content, "add")
                # 逻辑注释：统一通过 _create_memory 写入向量库和历史表，避免 raw 模式遗漏审计记录。
                mem_id = self._create_memory(msg_content, {msg_content: msg_embeddings}, per_msg_meta)

                # 逻辑注释：把对外需要的 id/memory/event/role 等信息记录下来，作为本次 add 的返回值。
                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )
            # 逻辑注释：返回本次实际新增的记忆列表，前面被跳过/去重的内容不会出现在结果里。
            return returned_memories

        # === V3 PHASED BATCH PIPELINE ===

        # Phase 0: Context gathering
        # 逻辑注释：把 filters 转成稳定会话 key，用来读取和保存最近消息上下文。
        session_scope = _build_session_scope(filters)
        # 逻辑注释：取最近对话作为 LLM 抽取记忆的上下文，帮助判断新信息是否真的值得写入。
        last_messages = self.db.get_last_messages(session_scope, limit=10)
        # 逻辑注释：把 role/content 消息列表整理成 prompt 可读的文本，供 embedding 和 LLM 使用。
        parsed_messages = parse_messages(messages)

        # Phase 1: Existing memory retrieval
        # 逻辑注释：实体检索只使用 session 级作用域字段，保证实体链接不会跨用户/agent/run 串数据。
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        # 逻辑注释：用当前消息整体作为查询生成 embedding，先召回可能相关的旧记忆。
        query_embedding = self.embedding_model.embed(parsed_messages, "search")
        # 逻辑注释：检索旧记忆的目的是给 LLM 对照：新消息是新增事实、重复事实，还是应更新已有事实。
        existing_results = self.vector_store.search(
            query=parsed_messages,
            vectors=query_embedding,
            top_k=10,
            filters=search_filters,
        )

        # Map UUIDs to integers (anti-hallucination)
        # 逻辑注释：只把必要的旧记忆文本传给 LLM，减少 prompt 体积和泄漏无关 payload 的风险。
        existing_memories = []
        # 逻辑注释：真实 UUID 不直接暴露给 LLM，而是映射成短编号，降低模型编造或误改 ID 的概率。
        uuid_mapping = {}
        # 逻辑注释：按召回顺序给旧记忆编号，后面 prompt 中使用这些短 ID 引用旧记忆。
        for idx, mem in enumerate(existing_results):
            # 逻辑注释：保存短编号到真实 UUID 的映射，必要时可以把 LLM 的引用还原成真实记忆 ID。
            uuid_mapping[str(idx)] = mem.id
            # 逻辑注释：构造给 LLM 的旧记忆摘要，只保留短 ID 和正文，避免 prompt 复杂化。
            existing_memories.append({"id": str(idx), "text": mem.payload.get("data", "")})

        # Phase 2: LLM extraction (single call)
        # 逻辑注释：只有 agent_id 且没有 user_id 时视为纯 agent 作用域，prompt 会额外强调 agent 语境。
        is_agent_scoped = bool(filters.get("agent_id")) and not filters.get("user_id")
        # 逻辑注释：使用增量抽取系统提示，目标是只抽取值得长期保存的新事实。
        system_prompt = ADDITIVE_EXTRACTION_PROMPT
        # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
        if is_agent_scoped:
            # 逻辑注释：agent 作用域下追加语境后缀，让 LLM 按 agent 记忆而不是用户画像来理解对话。
            system_prompt += AGENT_CONTEXT_SUFFIX

        # 逻辑注释：全局自定义指令会在 LLM 抽取记忆时作为默认额外要求。
        custom_instr = prompt or self.custom_instructions

        # 逻辑注释：把旧记忆、新消息、最近上下文和自定义规则合成一个用户 prompt，供 LLM 一次性判断。
        user_prompt = generate_additive_extraction_prompt(
            existing_memories=existing_memories,
            new_messages=parsed_messages,
            last_k_messages=last_messages,
            custom_instructions=custom_instr,
        )

        try:
            # 逻辑注释：LLM 只调用一次并要求 JSON 输出，减少多轮抽取带来的延迟和不一致。
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
        except Exception as e:
            # 逻辑注释：抽取失败时记录错误并返回空结果，避免把未理解的文本错误写成记忆。
            logger.error(f"LLM extraction failed: {e}")
            # 逻辑注释：该分支没有产生可写入/可返回的记忆，返回空列表而不是报错。
            return []

        # Parse response
        try:
            # 逻辑注释：先去掉 ```json 这类代码块包裹，提升 json.loads 成功率。
            response = remove_code_blocks(response)
            # 逻辑注释：空响应说明 LLM 没有给出可解析内容，直接视为没有抽到记忆。
            if not response or not response.strip():
                # 逻辑注释：这里保存该阶段的中间结果，供后续抽取、去重、批量写入或返回结果复用。
                extracted_memories = []
            else:
                try:
                    # 逻辑注释：按约定读取 JSON 里的 memory 数组，后续每个元素应包含待写入文本。
                    extracted_memories = json.loads(response, strict=False).get("memory", [])
                # 逻辑注释：LLM 有时不会返回严格 JSON；第一次解析失败后，再尝试从文本中抽取 JSON 片段。
                except json.JSONDecodeError:
                    # 逻辑注释：如果整段不是合法 JSON，就从文本里尽量截取 JSON 片段再解析。
                    extracted_json = extract_json(response)
                    # 逻辑注释：这里保存该阶段的中间结果，供后续抽取、去重、批量写入或返回结果复用。
                    extracted_memories = json.loads(extracted_json, strict=False).get("memory", [])
        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
        except Exception as e:
            # 逻辑注释：解析错误只影响本次抽取结果，不让异常继续破坏调用方流程。
            logger.error(f"Error parsing extraction response: {e}")
            # 逻辑注释：这里保存该阶段的中间结果，供后续抽取、去重、批量写入或返回结果复用。
            extracted_memories = []

        # 逻辑注释：LLM 没抽到长期记忆时仍保存原消息上下文，方便下一次抽取时参考最近对话。
        if not extracted_memories:
            # Save messages even if nothing extracted
            # 逻辑注释：保存原始消息到历史上下文库，后续 add 可以利用 last_messages 判断记忆变化。
            self.db.save_messages(messages, session_scope)
            # 逻辑注释：该分支没有产生可写入/可返回的记忆，返回空列表而不是报错。
            return []

        # Phase 3: Batch embed all extracted memory texts
        # 逻辑注释：只对非空文本生成 embedding；空文本不会成为记忆，也不浪费 embedding 调用。
        mem_texts = [m.get("text", "") for m in extracted_memories if m.get("text")]
        try:
            # 逻辑注释：批量 embedding 能减少 provider 调用次数，是批处理新增记忆的主要性能优化。
            mem_embeddings_list = self.embedding_model.embed_batch(mem_texts, "add")
            # 逻辑注释：把文本和 embedding 建成映射，后面构造记录时可以 O(1) 取向量。
            embed_map = dict(zip(mem_texts, mem_embeddings_list))
        # 逻辑注释：这里进入兜底路径，通常会从批量操作退化为逐条处理，提升整体成功率。
        except Exception:
            # Fallback: embed individually
            # 逻辑注释：批量 embedding 失败后准备逐条兜底，尽量让部分可处理记忆仍能写入。
            embed_map = {}
            # 逻辑注释：逐条 embedding 作为降级路径，牺牲性能换取更高的成功率。
            for text in mem_texts:
                try:
                    # 逻辑注释：这里保存该阶段的中间结果，供后续抽取、去重、批量写入或返回结果复用。
                    embed_map[text] = self.embedding_model.embed(text, "add")
                # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
                except Exception as e:
                    # 逻辑注释：单条文本 embedding 失败只跳过该条，避免整批新增失败。
                    logger.warning(f"Failed to embed memory text: {e}")

        # Phase 4: Per-memory CPU processing + Phase 5: Hash dedup
        # Build set of existing hashes for dedup
        # 逻辑注释：收集旧记忆的内容哈希，后面用它快速判断是否已经存过完全相同的文本。
        existing_hashes = set()
        # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
        for mem in existing_results:
            # 逻辑注释：旧记忆 payload 里的 hash 是去重依据，比直接比较所有文本更稳定高效。
            h = mem.payload.get("hash") if hasattr(mem, "payload") and mem.payload else None
            # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
            if h:
                existing_hashes.add(h)

        # 逻辑注释：records 是批量写入的中间结构，集中保存 id、文本、向量和 payload。
        records = []  # (memory_id, text, embedding, payload)
        # 逻辑注释：本批次内部也要去重，避免 LLM 在一次响应里重复抽取同一事实。
        seen_hashes = set()  # dedup within the current batch
        # 逻辑注释：逐条处理 LLM 抽取出的候选记忆，只有通过非空、可 embedding、非重复校验的才会入库。
        for mem in extracted_memories:
            # 逻辑注释：候选记忆以 text 字段为正文；缺失 text 的条目不具备可存储内容。
            text = mem.get("text")
            # 逻辑注释：没有正文或没有成功生成 embedding 的候选都会被跳过，保证后面 records 完整可写。
            if not text or text not in embed_map:
                continue

            # 逻辑注释：使用文本 MD5 作为内容指纹，用于跨批次和批次内的精确重复检测。
            mem_hash = hashlib.md5(text.encode()).hexdigest()
            # 逻辑注释：如果内容哈希已经出现过，就说明是精确重复记忆，跳过以保持记忆库简洁。
            if mem_hash in existing_hashes or mem_hash in seen_hashes:
                logger.debug(f"Skipping duplicate memory (hash match): {text[:50]}")
                continue
            seen_hashes.add(mem_hash)

            # 逻辑注释：提前保存词形归一化文本，后续关键词检索无需每次重新处理存量记忆。
            text_lemmatized = lemmatize_for_bm25(text)

            # 逻辑注释：每条记忆用 UUID 作为向量库 ID，保证跨批次新增也不会冲突。
            memory_id = str(uuid.uuid4())
            # 逻辑注释：每条候选记忆都复制一份基础 metadata，再补充该记忆自己的 data/hash/time 等字段。
            mem_metadata = deepcopy(metadata)
            # 逻辑注释：记忆正文放入 payload 的 data 字段，读取和搜索结果格式化都从这里取文本。
            mem_metadata["data"] = text
            # 逻辑注释：把 BM25 用的归一化文本一起存进 payload，服务混合检索。
            mem_metadata["text_lemmatized"] = text_lemmatized
            # 逻辑注释：hash 存入 payload，后续新增时能用旧 hash 快速去重。
            mem_metadata["hash"] = mem_hash
            # 逻辑注释：调用方没有提供创建时间时，使用当前 UTC 时间作为记忆创建时间。
            if "created_at" not in mem_metadata:
                # 逻辑注释：这里保存该阶段的中间结果，供后续抽取、去重、批量写入或返回结果复用。
                mem_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
            # 逻辑注释：新增时更新时间等于创建时间，后续 update 才会改变 updated_at。
            mem_metadata["updated_at"] = mem_metadata["created_at"]
            # 逻辑注释：LLM 如果标出事实归属，就把 attributed_to 写入 payload，便于区分事实属于谁。
            if mem.get("attributed_to"):
                # 逻辑注释：这里保存该阶段的中间结果，供后续抽取、去重、批量写入或返回结果复用。
                mem_metadata["attributed_to"] = mem["attributed_to"]

            # 逻辑注释：通过 records 聚合写入所需的四元组，后续向量插入、历史记录、实体链接都复用它。
            records.append((memory_id, text, embed_map[text], mem_metadata))

        # 逻辑注释：所有候选都被过滤/去重后，只保存上下文消息，不向向量库写任何新记忆。
        if not records:
            # 逻辑注释：保存原始消息到历史上下文库，后续 add 可以利用 last_messages 判断记忆变化。
            self.db.save_messages(messages, session_scope)
            # 逻辑注释：该分支没有产生可写入/可返回的记忆，返回空列表而不是报错。
            return []

        # Phase 6: Batch persist
        # 逻辑注释：从 records 拆出向量列表，供向量库批量 insert。
        all_vectors = [r[2] for r in records]
        # 逻辑注释：从 records 拆出 ID 列表，和向量列表一一对应。
        all_ids = [r[0] for r in records]
        # 逻辑注释：从 records 拆出 payload 列表，写入后读取/过滤/关键词检索都依赖这些字段。
        all_payloads = [r[3] for r in records]

        try:
            self.vector_store.insert(
                vectors=all_vectors,
                ids=all_ids,
                payloads=all_payloads,
            )
        # 逻辑注释：这里进入兜底路径，通常会从批量操作退化为逐条处理，提升整体成功率。
        except Exception:
            # Fallback: insert one by one
            # 逻辑注释：批量插入失败后逐条重试，让部分记忆仍有机会写入成功。
            for mid, vec, pay in zip(all_ids, all_vectors, all_payloads):
                try:
                    # 逻辑注释：这里保存该阶段的中间结果，供后续抽取、去重、批量写入或返回结果复用。
                    self.vector_store.insert(vectors=[vec], ids=[mid], payloads=[pay])
                # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
                except Exception as e:
                    # 逻辑注释：逐条插入失败才记录错误；这个错误只影响对应 memory_id。
                    logger.error(f"Failed to insert memory {mid}: {e}")

        # Batch history
        # 逻辑注释：为每条新增记忆准备历史记录，保证向量库写入后也有可审计的 ADD 事件。
        history_records = [
            {
                "memory_id": r[0],
                "old_memory": None,
                "new_memory": r[1],
                "event": "ADD",
                "created_at": r[3].get("created_at"),
                "is_deleted": 0,
            }
            # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
            for r in records
        ]
        try:
            # 逻辑注释：优先批量写历史，和批量插入一样减少数据库调用。
            self.db.batch_add_history(history_records)
        # 逻辑注释：这里进入兜底路径，通常会从批量操作退化为逐条处理，提升整体成功率。
        except Exception:
            # Fallback: add one by one
            # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
            for hr in history_records:
                try:
                    # 逻辑注释：批量写历史失败后逐条补写，避免完全丢失审计记录。
                    self.db.add_history(hr["memory_id"], None, hr["new_memory"], "ADD", created_at=hr.get("created_at"))
                # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
                except Exception as e:
                    logger.error(f"Failed to add history for {hr['memory_id']}: {e}")

        # Phase 7: Batch entity linking
        try:
            # 逻辑注释：实体抽取只需要记忆文本，因此从 records 中取出所有文本做批处理。
            all_texts = [r[1] for r in records]
            # 逻辑注释：批量抽实体减少重复调用，后面用实体索引增强检索排序。
            all_entities = extract_entities_batch(all_texts)

            # 7a: Global dedup — collect unique entities across all memories
            # 逻辑注释：全局实体表把同批次重复实体合并，并记录它关联的所有 memory_id。
            global_entities = {}  # normalized_key -> (entity_type, entity_text, set of memory_ids)
            # 逻辑注释：按 records 顺序把每条记忆和对应实体列表对齐，建立实体到记忆的关系。
            for idx, (memory_id, text, embedding, payload) in enumerate(records):
                # 逻辑注释：如果批量抽取结果长度不完全匹配，就给缺失项空实体列表，避免越界。
                entities = all_entities[idx] if idx < len(all_entities) else []
                # 逻辑注释：逐个处理抽取出的实体，把每个实体都链接到当前记忆。
                for entity_type, entity_text in entities:
                    # 逻辑注释：实体去重用小写+去空白后的规范 key，降低大小写和首尾空格带来的重复。
                    key = entity_text.strip().lower()
                    # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                    if key in global_entities:
                        # 逻辑注释：同一实体已出现时只追加新的 memory_id，不重复保存实体文本。
                        global_entities[key][2].add(memory_id)
                    else:
                        # 逻辑注释：首次遇到实体时保存类型、原文和关联 memory_id 集合，后续用于批量查重/插入。
                        global_entities[key] = [entity_type, entity_text, {memory_id}]

            # 逻辑注释：只有抽到至少一个实体时才进入实体库流程，避免无意义的 embedding/search。
            if global_entities:
                # 逻辑注释：固定实体处理顺序，方便 entity_texts、embeddings 和后续结果按索引对齐。
                ordered_keys = list(global_entities.keys())
                # 逻辑注释：只把实体原文送去 embedding，类型和关联记忆保留在 global_entities 里。
                entity_texts = [global_entities[k][1] for k in ordered_keys]

                # 7b: Single batch embed for all unique entities
                try:
                    # 逻辑注释：唯一实体批量 embedding，避免同一个实体在一批记忆中重复计算。
                    entity_embeddings = self.embedding_model.embed_batch(entity_texts, "add")
                # 逻辑注释：这里进入兜底路径，通常会从批量操作退化为逐条处理，提升整体成功率。
                except Exception:
                    # Fallback: embed individually, use None for failures
                    # 逻辑注释：这里保存该阶段的中间结果，供后续抽取、去重、批量写入或返回结果复用。
                    entity_embeddings = []
                    # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
                    for t in entity_texts:
                        try:
                            entity_embeddings.append(self.embedding_model.embed(t, "add"))
                        # 逻辑注释：这里进入兜底路径，通常会从批量操作退化为逐条处理，提升整体成功率。
                        except Exception:
                            # 逻辑注释：单个实体 embedding 失败时用 None 占位，保持索引对齐并在后面过滤掉。
                            entity_embeddings.append(None)

                # Filter out entities with failed embeddings
                # 逻辑注释：过滤掉 embedding 失败的实体，只对有向量的实体做实体库检索。
                valid = [(i, k) for i, k in enumerate(ordered_keys) if entity_embeddings[i] is not None]
                # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                if valid:
                    # 逻辑注释：拆出有效实体的原始索引和 key，便于同时访问 embedding 和实体元数据。
                    valid_indices, valid_keys = zip(*valid)
                    # 逻辑注释：有效实体向量按 valid_keys 顺序排列，后续 search_batch 的返回也按这个顺序对齐。
                    valid_vectors = [entity_embeddings[i] for i in valid_indices]

                    # 7c: Batch search for existing entities
                    # 逻辑注释：有效实体文本和向量一起传给批量搜索，用于判断实体是否已存在。
                    valid_texts = [global_entities[k][1] for k in valid_keys]
                    # 逻辑注释：批量搜索已有实体，减少逐实体查询的网络/存储开销。
                    existing_matches = self.entity_store.search_batch(
                        queries=valid_texts,
                        vectors_list=valid_vectors,
                        top_k=1,
                        filters=search_filters,
                    )

                    # 7d: Separate into inserts vs updates
                    # 逻辑注释：把需要新建的实体先暂存在列表里，最后统一批量 insert。
                    to_insert_vectors, to_insert_ids, to_insert_payloads = [], [], []
                    # 逻辑注释：逐个有效实体根据搜索结果决定更新已有实体还是加入新建列表。
                    for j, key in enumerate(valid_keys):
                        # 逻辑注释：这里保存该阶段的中间结果，供后续抽取、去重、批量写入或返回结果复用。
                        entity_type, entity_text, memory_ids = global_entities[key]
                        # 逻辑注释：搜索结果按实体顺序对齐；缺失时按空列表处理，表示没有匹配实体。
                        matches = existing_matches[j] if j < len(existing_matches) else []

                        # 逻辑注释：高度相似才认为是同一实体，避免实体索引过度合并。
                        if matches and matches[0].score >= 0.95:
                            # Update existing entity
                            # 逻辑注释：这里保存该阶段的中间结果，供后续抽取、去重、批量写入或返回结果复用。
                            match = matches[0]
                            payload = match.payload or {}
                            # 逻辑注释：用 set 合并已有链接和本批新链接，天然去重。
                            linked = set(payload.get("linked_memory_ids", []))
                            # 逻辑注释：把本批中关联该实体的所有 memory_id 合并进已有实体链接。
                            linked |= memory_ids
                            # 逻辑注释：排序后写回 payload，让结果稳定，也便于调试比较。
                            payload["linked_memory_ids"] = sorted(linked)
                            try:
                                self.entity_store.update(
                                    vector_id=match.id,
                                    vector=None,
                                    payload=payload,
                                )
                            # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
                            except Exception as e:
                                logger.debug(f"Entity update failed for '{entity_text}': {e}")
                        else:
                            # New entity — collect for batch insert
                            # 逻辑注释：没有匹配实体时，把该实体加入待插入集合，稍后统一写入。
                            to_insert_vectors.append(valid_vectors[j])
                            to_insert_ids.append(str(uuid.uuid4()))
                            # 逻辑注释：新实体 payload 带上实体信息、关联记忆和 session filters，支持后续增强检索和清理。
                            to_insert_payloads.append({
                                "data": entity_text,
                                "entity_type": entity_type,
                                "linked_memory_ids": sorted(memory_ids),
                                **search_filters,
                            })

                    # 7e: Single batch insert for all new entities
                    # 逻辑注释：只有存在新实体时才调用 insert，避免空批次触发某些向量库异常。
                    if to_insert_vectors:
                        try:
                            # 逻辑注释：把新实体向量和 payload 写入实体库，建立实体索引。
                            self.entity_store.insert(
                                vectors=to_insert_vectors,
                                ids=to_insert_ids,
                                payloads=to_insert_payloads,
                            )
                        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
                        except Exception as e:
                            # 逻辑注释：批量实体插入失败不影响已写入的记忆，只记录警告供排查。
                            logger.warning(f"Batch entity insert failed: {e}")
        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
        except Exception as e:
            # 逻辑注释：实体链接属于增强能力，失败时不回滚主记忆写入。
            logger.warning(f"Batch entity linking failed: {e}")

        # Phase 8: Save messages + return
        # 逻辑注释：保存原始消息到历史上下文库，后续 add 可以利用 last_messages 判断记忆变化。
        self.db.save_messages(messages, session_scope)

        returned_memories = [
            {"id": r[0], "memory": r[1], "event": "ADD"}
            # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
            for r in records
        ]

        # 逻辑注释：遥测前对 filters 做脱敏/编码，只上报维度信息而不是原始实体 ID。
        keys, encoded_ids = process_telemetry_filters(filters)
        # 逻辑注释：记录当前操作的遥测事件，用于观察 API 使用情况和排查性能/行为问题。
        capture_event(
            "mem0.add",
            self,
            # 逻辑注释：保存 API 版本，遥测事件会带上它，便于区分不同版本的行为。
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"},
        )
        # 逻辑注释：返回本次实际新增的记忆列表，前面被跳过/去重的内容不会出现在结果里。
        return returned_memories

    # 逻辑注释：按 memory_id 读取单条记忆，并把系统字段和自定义 metadata 整理成对外稳定的返回结构。
    def get(self, memory_id):
        """
        Retrieve a memory by ID.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        # 逻辑注释：记录当前操作的遥测事件，用于观察 API 使用情况和排查性能/行为问题。
        capture_event("mem0.get", self, {"memory_id": memory_id, "sync_type": "sync"})
        # 逻辑注释：通过向量库 ID 直接取 payload，这是 get/update/delete 的基础读取路径。
        memory = self.vector_store.get(vector_id=memory_id)
        # 逻辑注释：向量库没有返回记录时表示 memory_id 不存在，get 用 None 表达未找到。
        if not memory:
            # 逻辑注释：没有可用结果时显式返回 None，让调用方能区分“没找到”和异常。
            return None

        # 逻辑注释：这些 payload 字段是常用作用域/来源信息，返回时提升到顶层，调用方读取更方便。
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        # 逻辑注释：核心字段和已提升字段不再放进 metadata，避免结果里重复出现同一信息。
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        # 逻辑注释：用 MemoryItem 统一字段名和序列化形态，屏蔽不同向量库返回对象的差异。
        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload.get("data", ""),
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump()

        # 逻辑注释：遍历可提升字段，只有 payload 里真的存在时才加入返回结果。
        for key in promoted_payload_keys:
            # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
            if key in memory.payload:
                # 逻辑注释：把作用域/角色字段放到结果顶层，方便用户直接过滤或展示。
                result_item[key] = memory.payload[key]

        # 逻辑注释：除系统字段外的 payload 都视为用户自定义 metadata，保留在 metadata 子对象里。
        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        # 逻辑注释：只有存在额外 metadata 时才添加 metadata 字段，保持返回结构简洁。
        if additional_metadata:
            # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
            result_item["metadata"] = additional_metadata

        # 逻辑注释：返回已经格式化过的结果，调用方无需理解向量库原始 payload 结构。
        return result_item

    # 逻辑注释：列出某个作用域下的记忆；先校验 filters/top_k，再委托向量库 list 并格式化结果。
    def get_all(
        self,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        **kwargs,
    ):
        """
        List all memories.

        Args:
            filters (dict): Filter dict containing entity IDs and optional metadata filters.
                Must contain at least one of: user_id, agent_id, run_id.
                Example: filters={"user_id": "u1", "agent_id": "a1"}
            top_k (int, optional): The maximum number of memories to return. Defaults to 20.

        Returns:
            dict: A dictionary containing a list of memories under the "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", ...}]}`

        Raises:
            ValueError: If filters doesn't contain at least one of user_id, agent_id, run_id,
                or if top_k is invalid.
        """
        # Reject top-level entity params - must use filters instead
        # 逻辑注释：兼容性层面拒绝 user_id 等顶层参数，统一要求调用方通过 filters 指定作用域。
        _reject_top_level_entity_params(kwargs, "get_all")

        # Validate top_k
        # 逻辑注释：在触达向量库前校验 top_k/threshold，错误更早、更清晰。
        _validate_search_params(top_k=top_k)

        # Validate and trim entity IDs in filters
        # 逻辑注释：复制 filters 后再修改，避免 trim 或高级过滤转换影响调用方原对象。
        effective_filters = dict(filters) if filters else {}
        # 逻辑注释：如果 filters 里包含实体 ID，就先做同样的清洗校验，保证查询作用域格式正确。
        if "user_id" in effective_filters:
            # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        # 逻辑注释：如果 filters 里包含实体 ID，就先做同样的清洗校验，保证查询作用域格式正确。
        if "agent_id" in effective_filters:
            # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        # 逻辑注释：如果 filters 里包含实体 ID，就先做同样的清洗校验，保证查询作用域格式正确。
        if "run_id" in effective_filters:
            # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )

        # Validate filters contains at least one entity ID
        # 逻辑注释：读取/搜索必须至少限定一个实体作用域，防止默认扫描整个记忆库。
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
                "Example: filters={'user_id': 'u1'}"
            )

        # 逻辑注释：内部统一用 limit 表示最终返回条数，和向量库参数命名保持一致。
        limit = top_k

        # 逻辑注释：遥测前对 filters 做脱敏/编码，只上报维度信息而不是原始实体 ID。
        keys, encoded_ids = process_telemetry_filters(effective_filters)
        # 逻辑注释：记录当前操作的遥测事件，用于观察 API 使用情况和排查性能/行为问题。
        capture_event(
            "mem0.get_all", self, {"limit": limit, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"}
        )

        # 逻辑注释：实际 list 和格式化下沉到 helper，get_all 本身只处理校验和返回包装。
        all_memories_result = self._get_all_from_vector_store(effective_filters, limit)

        # 逻辑注释：对外统一用 results 包一层，保持 add/search/get_all 等接口返回结构一致。
        return {"results": all_memories_result}

    # 逻辑注释：兼容不同向量库 list 返回结构，统一展开为 MemoryItem 列表，同时保留额外 metadata。
    def _get_all_from_vector_store(self, filters, limit):
        # 逻辑注释：按 filters 从向量库列出记忆，top_k/limit 控制最多返回多少条。
        memories_result = self.vector_store.list(filters=filters, top_k=limit)

        # Handle different vector store return formats by inspecting first element
        # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
        if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0:
            # 逻辑注释：检查第一个元素的类型，用来判断向量库返回的是嵌套列表还是扁平列表。
            first_element = memories_result[0]

            # If first element is a container, unwrap one level
            # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
            if isinstance(first_element, (list, tuple)):
                # 逻辑注释：如果第一层包了一层列表，就展开一层得到真正的记忆对象列表。
                actual_memories = first_element
            else:
                # First element is a memory object, structure is already flat
                # 逻辑注释：如果返回已经是扁平结构，就直接使用，不做额外变换。
                actual_memories = memories_result
        else:
            # 逻辑注释：如果返回已经是扁平结构，就直接使用，不做额外变换。
            actual_memories = memories_result

        # 逻辑注释：这些 payload 字段是常用作用域/来源信息，返回时提升到顶层，调用方读取更方便。
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        # 逻辑注释：核心字段和已提升字段不再放进 metadata，避免结果里重复出现同一信息。
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        # 逻辑注释：统一把向量库对象转换成 SDK 对外返回的字典列表。
        formatted_memories = []
        # 逻辑注释：逐条格式化记忆对象，处理字段提升和额外 metadata。
        for mem in actual_memories:
            # 逻辑注释：用 MemoryItem 统一字段名和序列化形态，屏蔽不同向量库返回对象的差异。
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
            ).model_dump(exclude={"score"})

            # 逻辑注释：遍历可提升字段，只有 payload 里真的存在时才加入返回结果。
            for key in promoted_payload_keys:
                # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                if key in mem.payload:
                    # 逻辑注释：把作用域/角色字段放到结果顶层，方便用户直接过滤或展示。
                    memory_item_dict[key] = mem.payload[key]

            # 逻辑注释：除系统字段外的 payload 都视为用户自定义 metadata，保留在 metadata 子对象里。
            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            # 逻辑注释：只有存在额外 metadata 时才添加 metadata 字段，保持返回结构简洁。
            if additional_metadata:
                # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        # 逻辑注释：返回已经格式化过的结果，调用方无需理解向量库原始 payload 结构。
        return formatted_memories

    # 逻辑注释：搜索入口负责校验和预处理 filters，再调用混合检索；可选 rerank 会在初排结果上二次排序。
    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        threshold: float = 0.1,
        rerank: bool = False,
        **kwargs,
    ):
        """
        Searches for memories based on a query.

        Args:
            query (str): Query to search for.
            top_k (int, optional): Maximum number of results to return. Defaults to 20.
            filters (dict): Filter dict containing entity IDs and optional metadata filters.
                Must contain at least one of: user_id, agent_id, run_id.
                Example: filters={"user_id": "u1", "agent_id": "a1"}

                Enhanced metadata filtering with operators:
                - {"key": "value"} - exact match
                - {"key": {"eq": "value"}} - equals
                - {"key": {"ne": "value"}} - not equals
                - {"key": {"in": ["val1", "val2"]}} - in list
                - {"key": {"nin": ["val1", "val2"]}} - not in list
                - {"key": {"gt": 10}} - greater than
                - {"key": {"gte": 10}} - greater than or equal
                - {"key": {"lt": 10}} - less than
                - {"key": {"lte": 10}} - less than or equal
                - {"key": {"contains": "text"}} - contains text
                - {"key": {"icontains": "text"}} - case-insensitive contains
                - {"key": "*"} - wildcard match (any value)
                - {"AND": [filter1, filter2]} - logical AND
                - {"OR": [filter1, filter2]} - logical OR
                - {"NOT": [filter1]} - logical NOT
            threshold (float, optional): Minimum score for a memory to be included. Defaults to 0.1.
            rerank (bool, optional): Whether to rerank results. Defaults to False.

        Returns:
            dict: A dictionary containing the search results under a "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`

        Raises:
            ValueError: If filters doesn't contain at least one of user_id, agent_id, run_id,
                or if threshold/top_k values are invalid.
        """
        # Reject top-level entity params - must use filters instead
        # 逻辑注释：兼容性层面拒绝 user_id 等顶层参数，统一要求调用方通过 filters 指定作用域。
        _reject_top_level_entity_params(kwargs, "search")

        # Validate search parameters (before applying defaults)
        # 逻辑注释：在触达向量库前校验 top_k/threshold，错误更早、更清晰。
        _validate_search_params(threshold=threshold, top_k=top_k)

        # Validate and trim entity IDs in filters
        # 逻辑注释：复制 filters 后再修改，避免 trim 或高级过滤转换影响调用方原对象。
        effective_filters = filters.copy() if filters else {}
        # 逻辑注释：如果 filters 里包含实体 ID，就先做同样的清洗校验，保证查询作用域格式正确。
        if "user_id" in effective_filters:
            # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        # 逻辑注释：如果 filters 里包含实体 ID，就先做同样的清洗校验，保证查询作用域格式正确。
        if "agent_id" in effective_filters:
            # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        # 逻辑注释：如果 filters 里包含实体 ID，就先做同样的清洗校验，保证查询作用域格式正确。
        if "run_id" in effective_filters:
            # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )
        # 逻辑注释：读取/搜索必须至少限定一个实体作用域，防止默认扫描整个记忆库。
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
                "Example: filters={'user_id': 'u1'}"
            )

        # 逻辑注释：内部统一用 limit 表示最终返回条数，和向量库参数命名保持一致。
        limit = top_k

        # Apply enhanced metadata filtering if advanced operators are detected
        # 逻辑注释：检测到高级过滤语法时先转换成向量库兼容格式，否则简单 filters 直接透传。
        if self._has_advanced_operators(effective_filters):
            # 逻辑注释：把 AND/OR/NOT、比较操作符等高级语义转换成内部统一表达。
            processed_filters = self._process_metadata_filters(effective_filters)
            # Remove logical/operator keys that have been reprocessed
            # 逻辑注释：转换后移除原始逻辑操作符，避免同一个条件被同时以新旧两种格式传给向量库。
            for logical_key in ("AND", "OR", "NOT"):
                # 逻辑注释：已被转换的复杂字段从原 filters 删除，保持最终 filters 只有向量库能理解的结构。
                effective_filters.pop(logical_key, None)
            # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
            for fk in list(effective_filters.keys()):
                # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                if fk not in ("AND", "OR", "NOT", "user_id", "agent_id", "run_id") and isinstance(effective_filters.get(fk), dict):
                    # 逻辑注释：已被转换的复杂字段从原 filters 删除，保持最终 filters 只有向量库能理解的结构。
                    effective_filters.pop(fk, None)
            # 逻辑注释：把转换后的高级过滤条件合并回有效 filters，后续检索统一使用这份结果。
            effective_filters.update(processed_filters)

        # 逻辑注释：遥测前对 filters 做脱敏/编码，只上报维度信息而不是原始实体 ID。
        keys, encoded_ids = process_telemetry_filters(effective_filters)
        # 逻辑注释：记录当前操作的遥测事件，用于观察 API 使用情况和排查性能/行为问题。
        capture_event(
            "mem0.search",
            self,
            {
                "limit": limit,
                # 逻辑注释：保存 API 版本，遥测事件会带上它，便于区分不同版本的行为。
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "sync",
                "threshold": threshold,
                "advanced_filters": bool(filters and self._has_advanced_operators(filters)),
            },
        )

        # 逻辑注释：底层搜索会完成语义、关键词、实体增强的融合排序，search 入口只负责调用。
        original_memories = self._search_vector_store(query, effective_filters, limit, threshold)

        # Apply reranking if enabled and reranker is available
        # 逻辑注释：只有用户开启 rerank、实例也配置了 reranker 且已有初排结果时才做二次排序。
        if rerank and self.reranker and original_memories:
            try:
                # 逻辑注释：reranker 根据原始 query 对候选记忆重新排序，通常能提升相关性但会增加成本。
                reranked_memories = self.reranker.rerank(query, original_memories, limit)
                # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
                original_memories = reranked_memories
            # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
            except Exception as e:
                # 逻辑注释：重排失败不影响搜索可用性，直接退回初排结果。
                logger.warning(f"Reranking failed, using original results: {e}")

        # 逻辑注释：对外统一用 results 包一层，保持 add/search/get_all 等接口返回结构一致。
        return {"results": original_memories}

    # 逻辑注释：把平台层的增强过滤语法转换成向量库更容易消费的格式，并支持 AND/OR/NOT 组合。
    def _process_metadata_filters(self, metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process enhanced metadata filters and convert them to vector store compatible format.

        Args:
            metadata_filters: Enhanced metadata filters with operators

        Returns:
            Dict of processed filters compatible with vector store
        """
        # 逻辑注释：转换结果单独累积，最后再替换/合并到有效 filters 里。
        processed_filters = {}

        # 逻辑注释：定义 process_condition，封装这段业务逻辑，减少外部调用方理解内部细节的成本。
        def process_condition(key: str, condition: Any) -> Dict[str, Any]:
            # 逻辑注释：非 dict 条件代表简单等值匹配，是最常见、最直接的过滤形式。
            if not isinstance(condition, dict):
                # Simple equality: {"key": "value"}
                # 逻辑注释：星号表示通配字段，具体如何匹配由底层向量库适配层处理。
                if condition == "*":
                    # Wildcard: match everything for this field (implementation depends on vector store)
                    return {key: "*"}
                return {key: condition}

            result = {}
            # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
            for operator, value in condition.items():
                # Map platform operators to universal format that can be translated by each vector store
                # 逻辑注释：建立平台操作符到内部操作符的映射，当前两边同名，但保留了适配空间。
                operator_map = {
                    "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte",
                    "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                    "contains": "contains", "icontains": "icontains"
                }

                # 逻辑注释：只允许白名单里的操作符，避免未知过滤语法被静默传到向量库。
                if operator in operator_map:
                    # 逻辑注释：同一个字段可能有多个比较条件，用嵌套 dict 合并到同一字段下。
                    result.setdefault(key, {})[operator_map[operator]] = value
                else:
                    # 逻辑注释：遇到不支持的操作符立即报错，避免用户以为过滤生效但实际被忽略。
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
            return result

        # 逻辑注释：定义 merge_filters，封装这段业务逻辑，减少外部调用方理解内部细节的成本。
        def merge_filters(target: Dict[str, Any], source: Dict[str, Any]) -> None:
            """Merge source into target, deep-merging nested operator dicts for the same key."""
            # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
            for key, value in source.items():
                # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    # 逻辑注释：同一个字段的多个操作符合并到一起，例如 gte 和 lte 可以同时存在。
                    target[key].update(value)
                else:
                    # 逻辑注释：字段不存在或不是同类嵌套结构时，直接写入目标 filters。
                    target[key] = value

        # 逻辑注释：逐个处理原始 filters 条目，普通字段和逻辑操作符分开转换。
        for key, value in metadata_filters.items():
            # 逻辑注释：AND 语义是所有子条件同时成立，所以可以直接合并到同一个 filters 对象里。
            if key == "AND":
                # Logical AND: combine multiple conditions
                # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                if not isinstance(value, list):
                    # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
                    raise ValueError("AND operator requires a list of conditions")
                # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
                for condition in value:
                    for sub_key, sub_value in condition.items():
                        merge_filters(processed_filters, process_condition(sub_key, sub_value))
            # 逻辑注释：这是对前面判断的补充分支，用来兼容另一种输入/配置形态。
            elif key == "OR":
                # Logical OR: Pass through to vector store for implementation-specific handling
                # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                if not isinstance(value, list) or not value:
                    # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
                    raise ValueError("OR operator requires a non-empty list of conditions")
                # Store OR conditions in a way that vector stores can interpret
                # 逻辑注释：$or 保存多个备选条件，每个条件内部仍按普通字段规则转换。
                processed_filters["$or"] = []
                # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
                for condition in value:
                    or_condition = {}
                    # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
                    for sub_key, sub_value in condition.items():
                        merge_filters(or_condition, process_condition(sub_key, sub_value))
                    # 逻辑注释：$or 保存多个备选条件，每个条件内部仍按普通字段规则转换。
                    processed_filters["$or"].append(or_condition)
            # 逻辑注释：这是对前面判断的补充分支，用来兼容另一种输入/配置形态。
            elif key == "NOT":
                # Logical NOT: Pass through to vector store for implementation-specific handling
                # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                if not isinstance(value, list) or not value:
                    # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
                    raise ValueError("NOT operator requires a non-empty list of conditions")
                # 逻辑注释：$not 保存需要排除的条件集合，交给底层适配层处理。
                processed_filters["$not"] = []
                # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
                for condition in value:
                    not_condition = {}
                    # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
                    for sub_key, sub_value in condition.items():
                        merge_filters(not_condition, process_condition(sub_key, sub_value))
                    # 逻辑注释：$not 保存需要排除的条件集合，交给底层适配层处理。
                    processed_filters["$not"].append(not_condition)
            else:
                merge_filters(processed_filters, process_condition(key, value))

        return processed_filters

    # 逻辑注释：轻量判断 filters 是否包含高级操作符，用来决定是否需要进入转换流程。
    def _has_advanced_operators(self, filters: Dict[str, Any]) -> bool:
        """
        Check if filters contain advanced operators that need special processing.
        
        Args:
            filters: Dictionary of filters to check
            
        Returns:
            bool: True if advanced operators are detected
        """
        # 逻辑注释：非 dict filters 不可能包含高级过滤语法，直接返回 False。
        if not isinstance(filters, dict):
            return False
            
        # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
        for key, value in filters.items():
            # Check for platform-style logical operators
            # 逻辑注释：出现逻辑操作符就说明需要高级过滤转换。
            if key in ["AND", "OR", "NOT"]:
                return True
            # Check for comparison operators (without $ prefix for universal compatibility)
            # 逻辑注释：字段值是 dict 时可能包含 eq/gt/in 等比较操作符，需要继续检查。
            if isinstance(value, dict):
                # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
                for op in value.keys():
                    # 逻辑注释：命中任意比较/包含操作符，就判定 filters 使用了高级语法。
                    if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                        return True
            # Check for wildcard values
            # 逻辑注释：通配符也属于增强过滤语义，需要走转换逻辑。
            if value == "*":
                return True
        return False

    # 逻辑注释：底层混合检索：语义向量召回、关键词 BM25、实体增强一起打分，再统一排序和格式化。
    def _search_vector_store(self, query, filters, limit, threshold=0.1):
        # Guard against None threshold (backward compat)
        # 逻辑注释：兼容旧调用可能传 None 的情况，统一回落到默认阈值 0.1。
        if threshold is None:
            # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
            threshold = 0.1

        # Step 1: Preprocess query
        # 逻辑注释：查询文本也做词形归一化，保证和写入时保存的 text_lemmatized 在同一空间比较。
        query_lemmatized = lemmatize_for_bm25(query)
        # 逻辑注释：从查询中抽实体，后面可以通过实体库给相关记忆额外加分。
        query_entities = extract_entities(query)

        # Step 2: Embed query
        # 逻辑注释：查询向量用于语义召回，能找到表述不同但含义相近的记忆。
        embeddings = self.embedding_model.embed(query, "search")

        # Step 3: Semantic search (over-fetch for scoring pool)
        # 逻辑注释：先多召回一些候选，再融合 BM25/实体分数排序，避免早期截断错过好结果。
        internal_limit = max(limit * 4, 60)
        # 逻辑注释：语义检索提供候选池的主体，filters 保证只查当前作用域内的记忆。
        semantic_results = self.vector_store.search(
            query=query, vectors=embeddings, top_k=internal_limit, filters=filters
        )

        # Step 4: Keyword search (if store supports it)
        # 逻辑注释：如果向量库支持关键词检索，就额外召回词面匹配强的结果，用于混合排序。
        keyword_results = self.vector_store.keyword_search(
            query=query_lemmatized, top_k=internal_limit, filters=filters
        )

        # Step 5: Compute BM25 scores from keyword results
        # 逻辑注释：BM25 分数单独按 memory_id 保存，后面和语义分数融合。
        bm25_scores = {}
        # 逻辑注释：有些向量库可能不支持 keyword_search；None 表示跳过关键词分支。
        if keyword_results is not None:
            # 逻辑注释：根据查询长度/形态选择归一化参数，把 BM25 原始分数压到可融合区间。
            midpoint, steepness = get_bm25_params(query, lemmatized=query_lemmatized)
            # 逻辑注释：逐条读取关键词检索结果，将不同返回对象格式统一成 memory_id 和 raw_score。
            for mem in keyword_results:
                # 逻辑注释：兼容对象式和 dict 式结果，统一转成字符串 ID 作为打分 key。
                mem_id = str(mem.id) if hasattr(mem, 'id') else str(mem.get('id', ''))
                # 逻辑注释：同样兼容对象式/dict 式 score 字段，避免绑定某一种向量库返回类型。
                raw_score = mem.score if hasattr(mem, 'score') else mem.get('score', 0)
                # 逻辑注释：只有正向关键词匹配分才参与融合，零分或空值不会影响排序。
                if raw_score and raw_score > 0:
                    # 逻辑注释：把 BM25 原始分归一化，和语义/实体分数处在可比较尺度上。
                    bm25_scores[mem_id] = normalize_bm25(raw_score, midpoint, steepness)

        # Step 6: Compute entity boosts
        # 逻辑注释：实体增强默认为空；没有抽到查询实体时，最终排序不会受到实体分支影响。
        entity_boosts = {}
        # 逻辑注释：只有查询里有实体时才访问实体库，减少普通搜索的额外开销。
        if query_entities:
            # 逻辑注释：实体增强会把命中实体关联的记忆额外加分，让精确实体相关结果更靠前。
            entity_boosts = self._compute_entity_boosts(query_entities, filters)

        # Step 7: Build candidate set from semantic results
        # 逻辑注释：把语义召回结果转换成统一候选结构，供 score_and_rank 融合排序。
        candidates = []
        # 逻辑注释：遍历语义候选，保留 id、语义分和 payload，后续格式化也依赖 payload。
        for mem in semantic_results:
            # 逻辑注释：兼容对象式和 dict 式结果，统一转成字符串 ID 作为打分 key。
            mem_id = str(mem.id)
            candidates.append({
                "id": mem_id,
                "score": mem.score,
                # 逻辑注释：payload 里包含记忆正文、metadata、hash 等返回所需信息。
                "payload": mem.payload if hasattr(mem, 'payload') else {},
            })

        # Step 8: Score and rank
        # 逻辑注释：统一融合语义分、BM25 分和实体 boost，并按阈值/top_k 截断。
        scored_results = score_and_rank(
            semantic_results=candidates,
            bm25_scores=bm25_scores,
            entity_boosts=entity_boosts,
            threshold=threshold,
            top_k=limit,
        )

        # Step 9: Format results
        # 逻辑注释：这些 payload 字段是常用作用域/来源信息，返回时提升到顶层，调用方读取更方便。
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        # 逻辑注释：核心字段和已提升字段不再放进 metadata，避免结果里重复出现同一信息。
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
        original_memories = []
        # 逻辑注释：只格式化融合排序后的最终结果，而不是所有召回候选。
        for scored in scored_results:
            # 逻辑注释：从候选中安全取 payload；缺失时用空 dict 防止字段访问异常。
            payload = scored.get("payload") or {}

            # 逻辑注释：没有 data 的候选不是有效记忆文本，跳过避免返回空 memory。
            if not payload.get("data"):
                # 逻辑注释：当前项不满足处理条件，跳过它并继续处理下一项，保证整批流程不中断。
                continue  # Skip candidates with no payload data

            # 逻辑注释：用 MemoryItem 统一字段名和序列化形态，屏蔽不同向量库返回对象的差异。
            memory_item_dict = MemoryItem(
                id=scored["id"],
                memory=payload.get("data", ""),
                hash=payload.get("hash"),
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at"),
                score=scored["score"],
            ).model_dump()

            # 逻辑注释：遍历可提升字段，只有 payload 里真的存在时才加入返回结果。
            for key in promoted_payload_keys:
                # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                if key in payload:
                    # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
                    memory_item_dict[key] = payload[key]

            # 逻辑注释：除系统字段外的 payload 都视为用户自定义 metadata，保留在 metadata 子对象里。
            additional_metadata = {k: v for k, v in payload.items() if k not in core_and_promoted_keys}
            # 逻辑注释：只有存在额外 metadata 时才添加 metadata 字段，保持返回结构简洁。
            if additional_metadata:
                # 逻辑注释：向量库没有返回记录时表示 memory_id 不存在，get 用 None 表达未找到。
                if not memory_item_dict.get("metadata"):
                    # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
                    memory_item_dict["metadata"] = {}
                # 逻辑注释：把额外 metadata 合并到结果对象，既保留系统字段，又不丢调用方自定义字段。
                memory_item_dict["metadata"].update(additional_metadata)

            original_memories.append(memory_item_dict)

        # 逻辑注释：返回已经格式化过的结果，调用方无需理解向量库原始 payload 结构。
        return original_memories

    # 逻辑注释：根据查询实体去实体库找相关记忆，并给命中的 memory_id 加权，提升实体精确匹配的排序位置。
    def _compute_entity_boosts(self, query_entities, filters):
        """Compute per-memory entity boosts from entity store search.

        For each extracted entity from the query:
        1. Embed the entity text
        2. Search the entity store (threshold >= 0.5)
        3. For each matched entity, boost its linked memories

        Returns:
            Dict mapping memory_id (str) -> max entity boost [0, 0.5].
        """
        # Deduplicate entities (max 8)
        # 逻辑注释：用集合在单条文本内去重，避免同一个实体重复 upsert。
        seen = set()
        # 逻辑注释：实体增强前先准备去重后的实体列表，避免重复查询同一实体。
        deduped = []
        # 逻辑注释：最多处理前 8 个实体，防止复杂查询触发过多实体库查询。
        for entity_type, entity_text in query_entities[:8]:
            # 逻辑注释：实体去重用小写+去空白后的规范 key，降低大小写和首尾空格带来的重复。
            key = entity_text.strip().lower()
            # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
            if key and key not in seen:
                seen.add(key)
                # 逻辑注释：只把非空且未见过的实体加入待查询列表。
                deduped.append((entity_type, entity_text))

        # 逻辑注释：去重后没有实体时无需访问实体库，直接返回空 boost。
        if not deduped:
            # 逻辑注释：没有实体增强可用时返回空映射，后续融合打分自然退化为普通检索。
            return {}

        # 逻辑注释：实体检索只使用 session 级作用域字段，保证实体链接不会跨用户/agent/run 串数据。
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        # 逻辑注释：最终按 memory_id 保存 boost，多个实体命中同一记忆时取最大值。
        memory_boosts = {}

        try:
            # 逻辑注释：逐个查询实体库，每个实体都可能为一批关联记忆提供加分。
            for _, entity_text in deduped:
                # 逻辑注释：实体也需要单独向量化，才能在实体库里用相似度判断是否已有同一实体。
                entity_embedding = self.embedding_model.embed(entity_text, "search")
                # 逻辑注释：在实体库中找和查询实体相近的实体节点，再通过 linked_memory_ids 找到关联记忆。
                matches = self.entity_store.search(
                    query=entity_text,
                    vectors=entity_embedding,
                    top_k=500,
                    filters=search_filters,
                )

                # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
                for match in matches:
                    similarity = match.score if hasattr(match, 'score') else 0.0
                    # 逻辑注释：实体匹配太弱时不加分，避免噪声实体影响搜索排序。
                    if similarity < 0.5:
                        continue

                    payload = match.payload if hasattr(match, 'payload') else {}
                    # 逻辑注释：实体节点的反向链接列表告诉我们哪些记忆与该实体有关。
                    linked_memory_ids = payload.get("linked_memory_ids", [])
                    # 逻辑注释：链接字段异常时跳过该实体，避免坏 payload 影响搜索。
                    if not isinstance(linked_memory_ids, list):
                        continue

                    # Spread-attenuated boost: entities linking to many memories get attenuated
                    # 逻辑注释：实体关联的记忆越多，越可能是泛化实体，需要降低单条记忆的 boost。
                    num_linked = max(len(linked_memory_ids), 1)
                    # 逻辑注释：用扩散衰减权重抑制“高频实体”造成的过度加分。
                    memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
                    # 逻辑注释：最终实体 boost 同时考虑实体相似度、全局权重和扩散衰减。
                    boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight

                    # 逻辑注释：把同一实体带来的 boost 分发到它关联的每条记忆上。
                    for memory_id in linked_memory_ids:
                        # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                        if memory_id:
                            memory_key = str(memory_id)
                            # 逻辑注释：同一记忆被多个实体命中时取最大 boost，避免简单累加导致多实体查询过度放大。
                            memory_boosts[memory_key] = max(memory_boosts.get(memory_key, 0.0), boost)

        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
        except Exception as e:
            # 逻辑注释：实体增强失败时保留普通混合检索结果，搜索功能不中断。
            logger.warning(f"Entity boost computation failed: {e}")

        return memory_boosts

    # 逻辑注释：更新入口先生成新文本 embedding，再交给内部方法处理向量、metadata、历史和实体索引同步。
    def update(self, memory_id, data, metadata: Optional[Dict[str, Any]] = None):
        """
        Update a memory by ID.

        Args:
            memory_id (str): ID of the memory to update.
            data (str): New content to update the memory with.
            metadata (dict, optional): Metadata to update with the memory. Defaults to None.

        Returns:
            dict: Success message indicating the memory was updated.

        Example:
            >>> m.update(memory_id="mem_123", data="Likes to play tennis on weekends")
            {'message': 'Memory updated successfully!'}
        """
        # 逻辑注释：记录当前操作的遥测事件，用于观察 API 使用情况和排查性能/行为问题。
        capture_event("mem0.update", self, {"memory_id": memory_id, "sync_type": "sync"})

        # 逻辑注释：提前计算新文本 embedding，并用 dict 传给内部更新方法，避免重复计算。
        existing_embeddings = {data: self.embedding_model.embed(data, "update")}

        # 逻辑注释：内部更新方法负责真正修改向量库、写历史并同步实体索引。
        self._update_memory(memory_id, data, existing_embeddings, metadata)
        return {"message": "Memory updated successfully!"}

    # 逻辑注释：删除入口先确认 memory_id 存在，再删除向量记录并写入删除历史。
    def delete(self, memory_id):
        """
        Delete a memory by ID.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        # 逻辑注释：记录当前操作的遥测事件，用于观察 API 使用情况和排查性能/行为问题。
        capture_event("mem0.delete", self, {"memory_id": memory_id, "sync_type": "sync"})

        # 逻辑注释：通过向量库 ID 直接取 payload，这是 get/update/delete 的基础读取路径。
        existing_memory = self.vector_store.get(vector_id=memory_id)
        # 逻辑注释：找不到旧记忆时不能继续更新/删除，必须向调用方报告无效 memory_id。
        if existing_memory is None:
            # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
            raise ValueError(f"Memory with id {memory_id} not found")

        # 逻辑注释：内部删除方法统一处理向量库删除、历史记录和实体索引清理。
        self._delete_memory(memory_id, existing_memory)
        return {"message": "Memory deleted successfully!"}

    # 逻辑注释：按作用域批量删除记忆；要求至少一个实体过滤条件，避免误删整个库。
    def delete_all(self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None):
        """
        Delete all memories.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters: Dict[str, Any] = {}
        # 逻辑注释：有 user_id 时同时写入 metadata 和 filters，新增记忆和查询旧记忆会落在同一个用户作用域。
        if user_id:
            filters["user_id"] = user_id
        # 逻辑注释：agent_id 也参与存储和过滤，支持按 agent 维度隔离记忆。
        if agent_id:
            filters["agent_id"] = agent_id
        # 逻辑注释：run_id 用于一次运行/会话级别的隔离，适合临时任务或批处理场景。
        if run_id:
            filters["run_id"] = run_id

        # 逻辑注释：没有任何过滤条件时拒绝批量删除，避免误删所有记忆；全量清空必须显式调用 reset。
        if not filters:
            # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        # 逻辑注释：遥测前对 filters 做脱敏/编码，只上报维度信息而不是原始实体 ID。
        keys, encoded_ids = process_telemetry_filters(filters)
        # 逻辑注释：记录当前操作的遥测事件，用于观察 API 使用情况和排查性能/行为问题。
        capture_event("mem0.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"})
        # delete all vector memories and reset the collections
        # 逻辑注释：先列出当前作用域下所有记忆，再逐条走统一删除逻辑，确保历史和实体清理不遗漏。
        memories = self.vector_store.list(filters=filters)[0]
        # 逻辑注释：逐条删除可以复用 _delete_memory 的审计和实体清理流程。
        for memory in memories:
            # 逻辑注释：内部删除方法统一处理向量库删除、历史记录和实体索引清理。
            self._delete_memory(memory.id)

        # 逻辑注释：批量删除结束后记录删除数量，便于排查 filters 是否符合预期。
        logger.info(f"Deleted {len(memories)} memories")

        return {"message": "Memories deleted successfully!"}

    # 逻辑注释：读取某条记忆的变更历史，方便审计 ADD/UPDATE/DELETE 过程。
    def history(self, memory_id):
        """
        Get the history of changes for a memory by ID.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        # 逻辑注释：记录当前操作的遥测事件，用于观察 API 使用情况和排查性能/行为问题。
        capture_event("mem0.history", self, {"memory_id": memory_id, "sync_type": "sync"})
        # 逻辑注释：返回 memory_id 让上层可以继续记录、链接实体或给用户展示操作结果。
        return self.db.get_history(memory_id)

    # 逻辑注释：创建单条记忆的通用 helper：生成 ID、补齐 metadata/hash/time、写向量库并记录历史。
    def _create_memory(self, data, existing_embeddings, metadata=None):
        # 逻辑注释：创建前打 debug 日志，调试时可看到即将写入的记忆正文。
        logger.debug(f"Creating memory with {data=}")
        # 逻辑注释：如果上层已传入新文本 embedding，就直接复用，避免二次 embedding 调用。
        if data in existing_embeddings:
            # 逻辑注释：复用调用方已经计算好的 embedding，减少重复计算和 provider 成本。
            embeddings = existing_embeddings[data]
        else:
            # 逻辑注释：查询向量用于语义召回，能找到表述不同但含义相近的记忆。
            embeddings = self.embedding_model.embed(data, memory_action="add")
        # 逻辑注释：每条记忆用 UUID 作为向量库 ID，保证跨批次新增也不会冲突。
        memory_id = str(uuid.uuid4())
        # 逻辑注释：更新时先从调用方新 metadata 开始，再补齐系统字段和旧作用域字段。
        new_metadata = deepcopy(metadata) if metadata is not None else {}
        # 逻辑注释：更新后的正文写回 data 字段，读取和搜索都会看到新文本。
        new_metadata["data"] = data
        # 逻辑注释：更新后重新计算内容 hash，保证后续去重依据和新文本一致。
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
        if "created_at" not in new_metadata:
            # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
            new_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        new_metadata["updated_at"] = new_metadata["created_at"]
        # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data)

        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[new_metadata],
        )
        self.db.add_history(
            memory_id,
            None,
            data,
            "ADD",
            created_at=new_metadata.get("created_at"),
            updated_at=new_metadata.get("updated_at"),
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )
        # 逻辑注释：返回 memory_id 让上层可以继续记录、链接实体或给用户展示操作结果。
        return memory_id

    # 逻辑注释：把一段对话压缩成“过程性记忆”再存储，适合记录 agent 的长期操作流程。
    def _create_procedural_memory(self, messages, metadata=None, prompt=None):
        """
        Create a procedural memory

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        # 逻辑注释：过程性记忆生成前记录日志，因为它会调用 LLM 做总结，成本和普通写入不同。
        logger.info("Creating procedural memory")

        # 逻辑注释：构造用于过程性记忆的消息序列：系统提示、原对话、最后的总结指令。
        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {
                "role": "user",
                "content": "Create procedural memory of the above conversation.",
            },
        ]

        try:
            # 逻辑注释：调用 LLM 把整段对话总结成可长期保存的流程/操作记忆。
            procedural_memory = self.llm.generate_response(messages=parsed_messages)
            # 逻辑注释：去掉 LLM 可能包上的代码块标记，存储时只保留纯文本记忆。
            procedural_memory = remove_code_blocks(procedural_memory)
        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        # 逻辑注释：过程性记忆必须有 metadata/作用域，否则总结出来的流程无法归属到具体 agent/run。
        if metadata is None:
            # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
            raise ValueError("Metadata cannot be done for procedural memory.")

        # 逻辑注释：在原 metadata 基础上标记 memory_type，后续可区分普通事实记忆和过程性记忆。
        metadata = {**metadata, "memory_type": MemoryType.PROCEDURAL.value}
        # 逻辑注释：查询向量用于语义召回，能找到表述不同但含义相近的记忆。
        embeddings = self.embedding_model.embed(procedural_memory, memory_action="add")
        # 逻辑注释：过程性记忆最终仍按普通记忆写入向量库和历史表，只是正文来自 LLM 总结。
        memory_id = self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        # 逻辑注释：记录当前操作的遥测事件，用于观察 API 使用情况和排查性能/行为问题。
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "sync"})

        # 逻辑注释：按 add 接口的返回格式包装过程性记忆创建结果。
        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    # 逻辑注释：内部更新流程不仅改向量和 payload，还保留创建时间/作用域，记录历史，并重建相关实体链接。
    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
        logger.info(f"Updating memory with {data=}")

        try:
            # 逻辑注释：通过向量库 ID 直接取 payload，这是 get/update/delete 的基础读取路径。
            existing_memory = self.vector_store.get(vector_id=memory_id)
        # 逻辑注释：这里进入兜底路径，通常会从批量操作退化为逐条处理，提升整体成功率。
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        # 逻辑注释：找不到旧记忆时不能继续更新/删除，必须向调用方报告无效 memory_id。
        if existing_memory is None:
            # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
            raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")

        # 逻辑注释：保存旧文本，后面写历史记录时能形成 old_memory → new_memory 的变更链。
        prev_value = existing_memory.payload.get("data")

        # 逻辑注释：更新时先从调用方新 metadata 开始，再补齐系统字段和旧作用域字段。
        new_metadata = deepcopy(metadata) if metadata is not None else {}

        # 逻辑注释：更新后的正文写回 data 字段，读取和搜索都会看到新文本。
        new_metadata["data"] = data
        # 逻辑注释：更新后重新计算内容 hash，保证后续去重依据和新文本一致。
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data)
        # 逻辑注释：更新不能改变原创建时间，因此从旧 payload 继承 created_at。
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        # 逻辑注释：更新时间使用当前 UTC 时间，表示这次 update 的发生时间。
        new_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Preserve session identifiers from existing memory only if not provided in new metadata
        # 逻辑注释：如果调用方没显式覆盖 user_id，就沿用旧记忆的 user_id，避免更新后丢失作用域。
        if "user_id" not in new_metadata and "user_id" in existing_memory.payload:
            # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        # 逻辑注释：agent_id 同样默认继承旧值，保持记忆仍在原 agent 作用域内。
        if "agent_id" not in new_metadata and "agent_id" in existing_memory.payload:
            # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        # 逻辑注释：run_id 默认继承旧值，避免单条更新把记忆移出原运行范围。
        if "run_id" not in new_metadata and "run_id" in existing_memory.payload:
            # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
            new_metadata["run_id"] = existing_memory.payload["run_id"]
        # 逻辑注释：actor_id 来自原消息说话人，更新时继续保留，除非业务另行处理。
        if "actor_id" in existing_memory.payload:
            # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        # 逻辑注释：role 默认继承旧值，让更新后的记忆仍知道原始消息角色。
        if "role" not in new_metadata and "role" in existing_memory.payload:
            # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
            new_metadata["role"] = existing_memory.payload["role"]

        # 逻辑注释：如果上层已传入新文本 embedding，就直接复用，避免二次 embedding 调用。
        if data in existing_embeddings:
            # 逻辑注释：复用调用方已经计算好的 embedding，减少重复计算和 provider 成本。
            embeddings = existing_embeddings[data]
        else:
            # 逻辑注释：查询向量用于语义召回，能找到表述不同但含义相近的记忆。
            embeddings = self.embedding_model.embed(data, "update")

        # 逻辑注释：向量和 payload 一起更新，保证语义检索和返回内容同步变成新文本。
        self.vector_store.update(
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        self.db.add_history(
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )

        # Entity-store cleanup: strip this memory's id from old-text entities,
        # then re-extract entities from the new text and link them back.
        # 逻辑注释：从更新后的 metadata 提取 session filters，用于限定实体清理/重建的作用域。
        session_filters = {k: new_metadata[k] for k in ("user_id", "agent_id", "run_id") if new_metadata.get(k)}
        # 逻辑注释：先把该 memory_id 从旧实体链接里移除，避免旧文本实体继续影响搜索。
        self._remove_memory_from_entity_store(memory_id, session_filters)
        # 逻辑注释：再按新文本重新抽实体并链接，让实体索引和更新后的记忆保持一致。
        self._link_entities_for_memory(memory_id, data, session_filters)

        # 逻辑注释：返回 memory_id 让上层可以继续记录、链接实体或给用户展示操作结果。
        return memory_id

    # 逻辑注释：内部删除流程把向量库删除和历史审计打包在一起，并同步清理实体反向索引。
    def _delete_memory(self, memory_id, existing_memory=None):
        # 逻辑注释：删除前记录目标 ID，方便调试删除链路。
        logger.info(f"Deleting memory with {memory_id=}")
        # 逻辑注释：找不到旧记忆时不能继续更新/删除，必须向调用方报告无效 memory_id。
        if existing_memory is None:
            # 逻辑注释：通过向量库 ID 直接取 payload，这是 get/update/delete 的基础读取路径。
            existing_memory = self.vector_store.get(vector_id=memory_id)
            # 逻辑注释：找不到旧记忆时不能继续更新/删除，必须向调用方报告无效 memory_id。
            if existing_memory is None:
                # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
                raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")
        # 逻辑注释：删除历史需要保留被删除前的文本，因此先从 payload 取出旧值。
        prev_value = existing_memory.payload.get("data", "")
        # 逻辑注释：删除历史里保留原创建时间，并统一带时区时间到 UTC，方便审计排序。
        created_at = _normalize_iso_timestamp_to_utc(existing_memory.payload.get("created_at"))
        # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
        updated_at = datetime.now(timezone.utc).isoformat()
        # 逻辑注释：旧 payload 可能为空，用空 dict 兜底以便安全提取 session filters。
        payload = existing_memory.payload or {}
        # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
        session_filters = {k: payload[k] for k in ("user_id", "agent_id", "run_id") if payload.get(k)}
        # 逻辑注释：先从向量库删除主记忆，后面再写 DELETE 历史记录。
        self.vector_store.delete(vector_id=memory_id)
        self.db.add_history(
            memory_id,
            prev_value,
            None,
            "DELETE",
            created_at=created_at,
            updated_at=updated_at,
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )

        # Entity-store cleanup: strip this memory's id from any entity records
        # that linked to it. Non-fatal — the helper swallows errors.
        # 逻辑注释：先把该 memory_id 从旧实体链接里移除，避免旧文本实体继续影响搜索。
        self._remove_memory_from_entity_store(memory_id, session_filters)

        # 逻辑注释：返回 memory_id 让上层可以继续记录、链接实体或给用户展示操作结果。
        return memory_id

    # 逻辑注释：重置整个记忆系统：清理历史表、重建向量库，并在实体库已初始化时一并重置。
    def reset(self):
        """
        Reset the memory store by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        # 逻辑注释：reset 是破坏性操作，先用 warning 日志提示会清空所有记忆。
        logger.warning("Resetting all memories")

        # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
        if hasattr(self.db, "connection") and self.db.connection:
            # 逻辑注释：先删历史表，确保 reset 后历史状态和向量库状态一致地从空开始。
            self.db.connection.execute("DROP TABLE IF EXISTS history")
            # 逻辑注释：删除表后关闭旧 SQLite 连接，避免后续继续使用失效连接。
            self.db.connection.close()

        # 逻辑注释：SQLite 用来保存消息上下文和变更历史，和向量库形成“语义索引 + 审计记录”的双存储结构。
        self.db = SQLiteManager(self.config.history_db_path)

        # 逻辑注释：优先使用向量库自身 reset 能力，适配支持原地清空的 backend。
        if hasattr(self.vector_store, "reset"):
            # 逻辑注释：通过工厂 reset 向量库，保持不同 provider 的重置逻辑集中管理。
            self.vector_store = VectorStoreFactory.reset(self.vector_store)
        else:
            logger.warning("Vector store does not support reset. Skipping.")
            # 逻辑注释：先从向量库删除主记忆，后面再写 DELETE 历史记录。
            self.vector_store.delete_col()
            # 逻辑注释：创建向量存储后，记忆文本的向量和 payload 都会通过它进行插入、查询、更新和删除。
            self.vector_store = VectorStoreFactory.create(
                self.config.vector_store.provider, self.config.vector_store.config
            )
        # Reset entity store if initialized
        # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
        if self._entity_store is not None:
            try:
                # 逻辑注释：实体库已经初始化时也要重置，避免主记忆清空后实体索引残留。
                self._entity_store.reset()
            # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
            except Exception as e:
                logger.warning(f"Failed to reset entity store: {e}")
            # 逻辑注释：实体库先置空，后面通过 property 懒加载，避免不使用实体能力时创建多余向量库。
            self._entity_store = None

        # 逻辑注释：记录当前操作的遥测事件，用于观察 API 使用情况和排查性能/行为问题。
        capture_event("mem0.reset", self, {"sync_type": "sync"})

    # 逻辑注释：释放 SQLite 等持有的资源，避免长生命周期进程里连接泄漏。
    def close(self):
        """Release resources held by this Memory instance (SQLite connections, etc.)."""
        # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
        if hasattr(self, "db") and self.db is not None:
            # 逻辑注释：关闭 SQLite 连接，释放文件句柄/锁。
            self.db.close()
            # 逻辑注释：关闭后置空引用，防止后续误用已关闭连接。
            self.db = None

    # 逻辑注释：占位接口，明确当前 Memory 类还没有实现聊天能力。
    def chat(self, query):
        # 逻辑注释：显式抛出未实现错误，比静默返回更容易让调用方发现该接口不可用。
        raise NotImplementedError("Chat function not implemented yet.")


# 逻辑注释：异步版 Memory，实现与同步版几乎相同的业务流程，但通过 asyncio.to_thread 包装阻塞型 provider 调用。
class AsyncMemory(MemoryBase):
    # 逻辑注释：初始化 Memory 实例需要把配置里的各类 provider 变成真实客户端，并准备向量库、LLM、SQLite 历史库和可选 reranker。
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        # 逻辑注释：把配置保存到实例上，后续所有 provider 初始化、路径和版本信息都从这里读取。
        self.config = config

        # 逻辑注释：根据配置创建 embedding 模型；Memory 不关心具体 provider，只依赖统一 embed 接口。
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        # 逻辑注释：创建向量存储后，记忆文本的向量和 payload 都会通过它进行插入、查询、更新和删除。
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        # 逻辑注释：创建 LLM 客户端，infer/procedural 模式会用它从对话中抽取或总结记忆。
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        # 逻辑注释：SQLite 用来保存消息上下文和变更历史，和向量库形成“语义索引 + 审计记录”的双存储结构。
        self.db = SQLiteManager(self.config.history_db_path)
        # 逻辑注释：保存主记忆 collection 名，实体库会基于这个名字派生出独立 collection。
        self.collection_name = self.config.vector_store.config.collection_name
        # 逻辑注释：保存 API 版本，遥测事件会带上它，便于区分不同版本的行为。
        self.api_version = self.config.version
        # 逻辑注释：全局自定义指令会在 LLM 抽取记忆时作为默认额外要求。
        self.custom_instructions = self.config.custom_instructions
        # 逻辑注释：实体库先置空，后面通过 property 懒加载，避免不使用实体能力时创建多余向量库。
        self._entity_store = None

        # Initialize reranker if configured
        # 逻辑注释：reranker 默认不启用；只有配置显式提供时才创建，避免额外依赖和成本。
        self.reranker = None
        # 逻辑注释：检测到 reranker 配置才初始化二次排序器，搜索时也会按开关选择是否使用。
        if config.reranker:
            # 逻辑注释：通过工厂创建 reranker，使不同重排模型可以用同一套 Memory 搜索逻辑接入。
            self.reranker = RerankerFactory.create(
                config.reranker.provider,
                config.reranker.config
            )

        # 逻辑注释：只有遥测开关打开时才准备遥测专用向量库，普通运行不会产生额外存储开销。
        if MEM0_TELEMETRY:
            telemetry_config = _safe_deepcopy_config(self.config.vector_store.config)
            # 逻辑注释：遥测数据写入独立 collection，避免和用户真实记忆混在一起。
            telemetry_config.collection_name = "mem0migrations"
            # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
            if self.config.vector_store.provider in ["faiss", "qdrant"]:
                # 逻辑注释：文件型向量库需要独立目录，按 provider 名构造迁移/遥测存储路径。
                provider_path = f"migrations_{self.config.vector_store.provider}"
                telemetry_config.path = os.path.join(mem0_dir, provider_path)
                # 逻辑注释：目录不存在时提前创建，避免初始化本地向量库时因路径缺失失败。
                os.makedirs(telemetry_config.path, exist_ok=True)
            # 逻辑注释：创建遥测专用向量库客户端，后续 capture_event 可复用这个存储。
            self._telemetry_vector_store = VectorStoreFactory.create(self.config.vector_store.provider, telemetry_config)

        # 逻辑注释：初始化结束后记录一次 init 事件，并标明 sync/async，便于观测两种实现的使用情况。
        capture_event("mem0.init", self, {"sync_type": "async"})

    # 逻辑注释：把这个方法暴露成只读属性，调用方访问时像字段一样自然，同时内部仍可做懒加载。
    @property
    # 逻辑注释：实体向量库采用懒加载：只有真正需要实体链接/增强检索时才创建，减少初始化成本和嵌入式向量库锁冲突。
    def entity_store(self):
        """Lazily initialize entity store on first use."""
        # 逻辑注释：第一次访问实体库才进入初始化，后续直接复用已经创建的实例。
        if self._entity_store is None:
            # 逻辑注释：实体库复用主向量库配置的副本，避免直接修改主记忆 collection 配置。
            entity_config = _safe_deepcopy_config(self.config.vector_store.config)
            # 逻辑注释：保存主记忆 collection 名，实体库会基于这个名字派生出独立 collection。
            entity_collection = f"{self.collection_name}_entities"
            # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
            if hasattr(entity_config, 'collection_name'):
                # 逻辑注释：把副本的 collection 改成实体 collection，后续实体向量不会写入主记忆库。
                entity_config.collection_name = entity_collection
            # 逻辑注释：这是对前面判断的补充分支，用来兼容另一种输入/配置形态。
            elif isinstance(entity_config, dict):
                # 逻辑注释：把副本的 collection 改成实体 collection，后续实体向量不会写入主记忆库。
                entity_config['collection_name'] = entity_collection
            # For Qdrant, share the existing client to avoid RocksDB lock contention
            # when using embedded mode (path=...). QdrantConfig.client takes precedence
            # over host/port/path.
            # 逻辑注释：Qdrant 嵌入式模式下共享已有 client，避免同一路径被多个 RocksDB 实例同时打开导致锁冲突。
            if self.config.vector_store.provider == "qdrant" and hasattr(self.vector_store, "client"):
                # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                if hasattr(entity_config, "client"):
                    # 逻辑注释：把主向量库的 client 注入实体配置，实体库和主库共享同一个底层连接。
                    entity_config.client = self.vector_store.client
                # 逻辑注释：这是对前面判断的补充分支，用来兼容另一种输入/配置形态。
                elif isinstance(entity_config, dict):
                    # 逻辑注释：把主向量库的 client 注入实体配置，实体库和主库共享同一个底层连接。
                    entity_config["client"] = self.vector_store.client
            self._entity_store = VectorStoreFactory.create(
                self.config.vector_store.provider, entity_config
            )
        return self._entity_store

    # 逻辑注释：异步版实体写入逻辑保持和同步版一致，只是把阻塞的 embedding/vector 调用放到线程里执行。
    async def _upsert_entity_async(self, entity_text, entity_type, memory_id, filters):
        """Async variant of `_upsert_entity` — per-entity search-then-update-or-insert."""
        try:
            # 逻辑注释：实体也需要单独向量化，才能在实体库里用相似度判断是否已有同一实体。
            entity_embedding = await asyncio.to_thread(self.embedding_model.embed, entity_text, "add")
            # 逻辑注释：实体检索只使用 session 级作用域字段，保证实体链接不会跨用户/agent/run 串数据。
            search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}

            # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
            existing = await asyncio.to_thread(
                self.entity_store.search,
                query=entity_text,
                vectors=entity_embedding,
                top_k=1,
                filters=search_filters,
            )

            # 逻辑注释：0.95 作为近似同实体阈值，只有非常相近时才合并，降低误把不同实体合并的风险。
            if existing and existing[0].score >= 0.95:
                match = existing[0]
                payload = match.payload or {}
                # 逻辑注释：实体 payload 里维护反向链接列表，用来知道这个实体关联了哪些记忆。
                linked_ids = payload.get("linked_memory_ids", [])
                # 逻辑注释：只有新记忆 ID 不在列表里才追加，避免重复链接导致后续 boost 被放大。
                if memory_id not in linked_ids:
                    linked_ids.append(memory_id)
                    payload["linked_memory_ids"] = linked_ids
                    # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
                    await asyncio.to_thread(
                        self.entity_store.update,
                        vector_id=match.id,
                        vector=None,
                        payload=payload,
                    )
            else:
                # 逻辑注释：新实体需要独立 ID，和 memory_id 分开管理，便于实体库单独增删改查。
                entity_id = str(uuid.uuid4())
                # 逻辑注释：实体 payload 同时保存实体文本、类型、关联记忆和 session 过滤字段，后续搜索/清理都依赖这些信息。
                entity_payload = {
                    "data": entity_text,
                    "entity_type": entity_type,
                    "linked_memory_ids": [memory_id],
                    **{k: v for k, v in search_filters.items()},
                }
                # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
                await asyncio.to_thread(
                    self.entity_store.insert,
                    vectors=[entity_embedding],
                    ids=[entity_id],
                    payloads=[entity_payload],
                )
        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
        except Exception as e:
            # 逻辑注释：实体索引失败不应影响主记忆写入，所以这里只记录警告而不是抛出。
            logger.warning(f"Entity upsert failed for '{entity_text}' (async): {e}")

    # 逻辑注释：删除或更新记忆后清理实体索引：从实体的 linked_memory_ids 中移除该 memory_id，孤立实体直接删除。
    async def _remove_memory_from_entity_store(self, memory_id, filters):
        """Async variant of `Memory._remove_memory_from_entity_store`."""
        # 逻辑注释：第一次访问实体库才进入初始化，后续直接复用已经创建的实例。
        if self._entity_store is None:
            return
        # 逻辑注释：实体检索只使用 session 级作用域字段，保证实体链接不会跨用户/agent/run 串数据。
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        try:
            # 逻辑注释：清理时先列出当前作用域下的实体，再逐个检查是否链接了待删除/更新的记忆。
            listed = await asyncio.to_thread(self.entity_store.list, filters=search_filters, top_k=10000)
            # 逻辑注释：不同向量库 list 返回格式不一致，这里兼容嵌套列表和扁平列表两种结构。
            rows = listed[0] if isinstance(listed, (list, tuple)) and listed and isinstance(listed[0], list) else listed
            # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
            for row in rows or []:
                try:
                    # 逻辑注释：实体行可能来自不同实现，统一用 getattr 安全取 payload。
                    payload = getattr(row, "payload", None) or {}
                    linked = payload.get("linked_memory_ids", [])
                    # 逻辑注释：linked_memory_ids 不是列表或不包含目标 memory_id 时，说明这条实体不需要处理。
                    if not isinstance(linked, list) or memory_id not in linked:
                        continue
                    # 逻辑注释：构造移除目标 memory_id 后的新链接列表，用于判断实体是否还被其他记忆引用。
                    remaining = [mid for mid in linked if mid != memory_id]
                    # 逻辑注释：没有任何记忆再引用该实体时，实体节点已经孤立，可以删除。
                    if not remaining:
                        try:
                            # 逻辑注释：删除孤立实体，避免实体库里留下无法增强任何记忆的脏数据。
                            await asyncio.to_thread(self.entity_store.delete, vector_id=row.id)
                        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
                        except Exception as e:
                            logger.debug(f"Entity delete failed for id={row.id} (async): {e}")
                    else:
                        # 逻辑注释：实体仍被其他记忆引用时，需要取出实体文本重新生成向量以满足 update 接口要求。
                        entity_text = payload.get("data")
                        # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                        if not isinstance(entity_text, str) or not entity_text:
                            logger.debug(f"Entity id={row.id} missing 'data'; skipping update during cleanup (async)")
                            continue
                        try:
                            # 逻辑注释：有些向量库 update 要求同时传 vector，所以这里即使只改 payload 也重新计算实体向量。
                            vec = await asyncio.to_thread(self.embedding_model.embed, entity_text, "update")
                        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
                        except Exception as e:
                            logger.debug(f"Entity re-embed failed for '{entity_text}' (async): {e}")
                            continue
                        # 逻辑注释：保留实体原有信息，只替换 linked_memory_ids，避免丢失 entity_type/session 等字段。
                        new_payload = {**payload, "linked_memory_ids": remaining}
                        try:
                            # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
                            await asyncio.to_thread(
                                self.entity_store.update,
                                vector_id=row.id,
                                vector=vec,
                                payload=new_payload,
                            )
                        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
                        except Exception as e:
                            logger.debug(f"Entity update failed for id={row.id} (async): {e}")
                # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
                except Exception as e:
                    logger.debug(f"Entity cleanup error (async): {e}")
        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
        except Exception as e:
            logger.warning(f"Entity store cleanup failed for memory_id={memory_id} (async): {e}")

    # 逻辑注释：从单条记忆文本中抽取实体并建立链接，主要用于 update 后把新文本重新挂到实体索引上。
    async def _link_entities_for_memory(self, memory_id, text, filters):
        """Async variant of `Memory._link_entities_for_memory`."""
        try:
            # 逻辑注释：从记忆文本抽取实体，只有抽到实体才需要进入实体链接流程。
            entities = await asyncio.to_thread(extract_entities, text)
            # 逻辑注释：没有实体时直接返回，避免空循环和不必要的向量库访问。
            if not entities:
                return
            # 逻辑注释：用集合在单条文本内去重，避免同一个实体重复 upsert。
            seen = set()
            # 逻辑注释：逐个处理抽取出的实体，把每个实体都链接到当前记忆。
            for entity_type, entity_text in entities:
                # 逻辑注释：实体去重用小写+去空白后的规范 key，降低大小写和首尾空格带来的重复。
                key = entity_text.strip().lower()
                # 逻辑注释：空实体或已处理实体都跳过，保持实体链接的唯一性。
                if not key or key in seen:
                    continue
                seen.add(key)
                try:
                    # 逻辑注释：每个有效实体交给 upsert，内部决定复用旧实体还是新建实体。
                    await self._upsert_entity_async(entity_text, entity_type, memory_id, filters)
                # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
                except Exception as e:
                    logger.debug(f"Entity link failed for '{entity_text}' (async): {e}")
        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
        except Exception as e:
            logger.warning(f"Entity linking failed for memory_id={memory_id} (async): {e}")

    # 逻辑注释：类方法不依赖已有实例，适合作为另一种构造入口。
    @classmethod
    # 逻辑注释：从普通字典创建配置对象，再交给构造函数；这里把外部配置入口和类初始化解耦。
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config_dict)
        # 逻辑注释：配置校验错误需要原样抛出，调用方才能看到 Pydantic 提供的具体字段问题。
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    # 逻辑注释：静态方法不依赖实例状态，这里用于纯配置处理/转换逻辑。
    @staticmethod
    # 逻辑注释：当前只是透传配置，保留这个钩子方便以后在构造 MemoryConfig 前做兼容性转换。
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return config_dict
        # 逻辑注释：配置校验错误需要原样抛出，调用方才能看到 Pydantic 提供的具体字段问题。
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    # 逻辑注释：根据是否有 agent_id 且消息里是否出现 assistant，决定记忆应偏向 agent 视角还是 user 视角。
    def _should_use_agent_memory_extraction(self, messages, metadata):
        """Determine whether to use agent memory extraction based on the logic:
        - If agent_id is present and messages contain assistant role -> True
        - Otherwise -> False

        Args:
            messages: List of message dictionaries
            metadata: Metadata containing user_id, agent_id, etc.

        Returns:
            bool: True if should use agent memory extraction, False for user memory extraction
        """
        # Check if agent_id is present in metadata
        has_agent_id = metadata.get("agent_id") is not None

        # Check if there are assistant role messages
        has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)

        # Use agent memory extraction if agent_id is present and there are assistant messages
        return has_agent_id and has_assistant_messages

    # 逻辑注释：新增记忆的公共入口：先确定作用域和输入格式，再按 procedural/raw/infer 三条路径分流。
    async def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
        llm=None,
    ):
        """
        Create a new memory asynchronously.

        Args:
            messages (str or List[Dict[str, str]]): Messages to store in the memory.
            user_id (str, optional): ID of the user creating the memory.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            infer (bool, optional): Whether to infer the memories. Defaults to True.
            memory_type (str, optional): Type of memory to create. Defaults to None.
                                         Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.
            llm (BaseChatModel, optional): LLM class to use for generating procedural memories. Defaults to None. Useful when user is using LangChain ChatModel.
        Returns:
            dict: A dictionary containing the result of the memory addition operation.
        """
        # 逻辑注释：新增记忆前先统一构造 metadata 和 filters，保证写入、检索旧记忆和历史上下文使用同一作用域。
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_metadata=metadata
        )

        # 逻辑注释：当前只额外支持 procedural_memory；其他 memory_type 会让调用方误以为有别的处理逻辑，因此拒绝。
        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
            raise ValueError(
                f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories."
            )

        # 逻辑注释：单字符串输入被包装成 user 消息，方便后续统一按消息列表处理。
        if isinstance(messages, str):
            # 逻辑注释：把简写输入转换成标准 role/content 结构，后面的解析和 LLM prompt 不需要再分支处理。
            messages = [{"role": "user", "content": messages}]

        # 逻辑注释：这是对前面判断的补充分支，用来兼容另一种输入/配置形态。
        elif isinstance(messages, dict):
            # 逻辑注释：这里保存该阶段的中间结果，供后续抽取、去重、批量写入或返回结果复用。
            messages = [messages]

        # 逻辑注释：这是对前面判断的补充分支，用来兼容另一种输入/配置形态。
        elif not isinstance(messages, list):
            # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
            raise Mem0ValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries."
            )

        # 逻辑注释：过程性记忆要求 agent 作用域，因为它描述的是 agent 的行为流程，而不是普通用户事实。
        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            # 逻辑注释：命中过程序记忆分支后直接生成并存储 procedural memory，不再走普通事实抽取流程。
            results = await self._create_procedural_memory(
                # 逻辑注释：这里保存该阶段的中间结果，供后续抽取、去重、批量写入或返回结果复用。
                messages, metadata=processed_metadata, prompt=prompt, llm=llm
            )
            return results

        # 逻辑注释：如果配置启用视觉能力，消息解析要保留/处理图像内容，让 LLM 能理解多模态输入。
        if self.config.llm.config.get("enable_vision"):
            # 逻辑注释：视觉消息先规范化为 LLM 可接收格式；未启用视觉时仍会做基础消息清洗。
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            # 逻辑注释：视觉消息先规范化为 LLM 可接收格式；未启用视觉时仍会做基础消息清洗。
            messages = parse_vision_messages(messages)

        # 逻辑注释：普通记忆最终交给向量写入流程处理，add 只负责入口校验和分流。
        vector_store_result = await self._add_to_vector_store(messages, processed_metadata, effective_filters, infer, prompt=prompt)
        # 逻辑注释：对外统一用 results 包一层，保持 add/search/get_all 等接口返回结构一致。
        return {"results": vector_store_result}

    # 逻辑注释：真正写入向量库的核心流程：raw 模式直接存，infer 模式走“检索旧记忆→LLM 抽取→去重→批量入库→实体链接”。
    async def _add_to_vector_store(
        self,
        messages: list,
        metadata: dict,
        effective_filters: dict,
        infer: bool,
        prompt: Optional[str] = None,
    ):
        # 逻辑注释：关闭 infer 时不调用 LLM 抽取事实，而是把非 system 消息原样作为记忆写入。
        if not infer:
            # 逻辑注释：收集本次成功写入的记忆，最后按 API 约定返回给调用方。
            returned_memories = []
            # 逻辑注释：raw 模式逐条处理消息，每条有效消息都会成为一条独立记忆。
            for message_dict in messages:
                # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                if (
                    # 逻辑注释：raw 模式仍要校验每条消息至少有 role/content，否则无法形成可解释的记忆记录。
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    # 逻辑注释：单条消息格式错误只跳过并告警，不让一个坏消息导致整批写入失败。
                    logger.warning(f"Skipping invalid message format (async): {message_dict}")
                    continue

                # 逻辑注释：system 消息通常是指令/上下文，不应作为用户事实或对话记忆存储。
                if message_dict["role"] == "system":
                    continue

                # 逻辑注释：每条消息复制一份 metadata，避免给其中一条消息加 role/actor 时影响其他消息。
                per_msg_meta = deepcopy(metadata)
                # 逻辑注释：把原消息角色写入 metadata，后续读取/搜索时可以知道记忆来自 user 还是 assistant。
                per_msg_meta["role"] = message_dict["role"]

                # 逻辑注释：如果消息带 name，就把它作为 actor_id，支持多人/多角色对话里的说话人过滤。
                actor_name = message_dict.get("name")
                # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                if actor_name:
                    # 逻辑注释：actor_id 写入 metadata 后，后续可以按具体说话人查找或清理记忆。
                    per_msg_meta["actor_id"] = actor_name

                # 逻辑注释：raw 模式把原始 content 当成记忆正文，不做 LLM 改写。
                msg_content = message_dict["content"]
                # 逻辑注释：写向量库前先把文本转成 embedding，后续语义搜索才能召回这条记忆。
                msg_embeddings = await asyncio.to_thread(self.embedding_model.embed, msg_content, "add")
                # 逻辑注释：统一通过 _create_memory 写入向量库和历史表，避免 raw 模式遗漏审计记录。
                mem_id = await self._create_memory(msg_content, {msg_content: msg_embeddings}, per_msg_meta)

                # 逻辑注释：把对外需要的 id/memory/event/role 等信息记录下来，作为本次 add 的返回值。
                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )
            # 逻辑注释：返回本次实际新增的记忆列表，前面被跳过/去重的内容不会出现在结果里。
            return returned_memories

        # === V3 PHASED BATCH PIPELINE (async) ===

        # Phase 0: Context gathering
        # 逻辑注释：把 filters 转成稳定会话 key，用来读取和保存最近消息上下文。
        session_scope = _build_session_scope(effective_filters)
        # 逻辑注释：取最近对话作为 LLM 抽取记忆的上下文，帮助判断新信息是否真的值得写入。
        last_messages = await asyncio.to_thread(self.db.get_last_messages, session_scope, 10)
        # 逻辑注释：把 role/content 消息列表整理成 prompt 可读的文本，供 embedding 和 LLM 使用。
        parsed_messages = parse_messages(messages)

        # Phase 1: Existing memory retrieval
        # 逻辑注释：这里保存该阶段的中间结果，供后续抽取、去重、批量写入或返回结果复用。
        search_filters = {k: v for k, v in effective_filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        # 逻辑注释：用当前消息整体作为查询生成 embedding，先召回可能相关的旧记忆。
        query_embedding = await asyncio.to_thread(self.embedding_model.embed, parsed_messages, "search")
        # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
        existing_results = await asyncio.to_thread(
            self.vector_store.search,
            query=parsed_messages,
            vectors=query_embedding,
            top_k=10,
            filters=search_filters,
        )

        # Map UUIDs to integers (anti-hallucination)
        # 逻辑注释：只把必要的旧记忆文本传给 LLM，减少 prompt 体积和泄漏无关 payload 的风险。
        existing_memories = []
        # 逻辑注释：真实 UUID 不直接暴露给 LLM，而是映射成短编号，降低模型编造或误改 ID 的概率。
        uuid_mapping = {}
        # 逻辑注释：按召回顺序给旧记忆编号，后面 prompt 中使用这些短 ID 引用旧记忆。
        for idx, mem in enumerate(existing_results):
            # 逻辑注释：保存短编号到真实 UUID 的映射，必要时可以把 LLM 的引用还原成真实记忆 ID。
            uuid_mapping[str(idx)] = mem.id
            # 逻辑注释：构造给 LLM 的旧记忆摘要，只保留短 ID 和正文，避免 prompt 复杂化。
            existing_memories.append({"id": str(idx), "text": mem.payload.get("data", "")})

        # Phase 2: LLM extraction (single call)
        # 逻辑注释：只有 agent_id 且没有 user_id 时视为纯 agent 作用域，prompt 会额外强调 agent 语境。
        is_agent_scoped = bool(effective_filters.get("agent_id")) and not effective_filters.get("user_id")
        # 逻辑注释：使用增量抽取系统提示，目标是只抽取值得长期保存的新事实。
        system_prompt = ADDITIVE_EXTRACTION_PROMPT
        # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
        if is_agent_scoped:
            # 逻辑注释：agent 作用域下追加语境后缀，让 LLM 按 agent 记忆而不是用户画像来理解对话。
            system_prompt += AGENT_CONTEXT_SUFFIX

        # 逻辑注释：全局自定义指令会在 LLM 抽取记忆时作为默认额外要求。
        custom_instr = prompt or self.custom_instructions

        # 逻辑注释：把旧记忆、新消息、最近上下文和自定义规则合成一个用户 prompt，供 LLM 一次性判断。
        user_prompt = generate_additive_extraction_prompt(
            existing_memories=existing_memories,
            new_messages=parsed_messages,
            last_k_messages=last_messages,
            custom_instructions=custom_instr,
        )

        try:
            # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
            response = await asyncio.to_thread(
                self.llm.generate_response,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
        except Exception as e:
            # 逻辑注释：抽取失败时记录错误并返回空结果，避免把未理解的文本错误写成记忆。
            logger.error(f"LLM extraction failed (async): {e}")
            # 逻辑注释：该分支没有产生可写入/可返回的记忆，返回空列表而不是报错。
            return []

        # Parse response
        try:
            # 逻辑注释：先去掉 ```json 这类代码块包裹，提升 json.loads 成功率。
            response = remove_code_blocks(response)
            # 逻辑注释：空响应说明 LLM 没有给出可解析内容，直接视为没有抽到记忆。
            if not response or not response.strip():
                # 逻辑注释：这里保存该阶段的中间结果，供后续抽取、去重、批量写入或返回结果复用。
                extracted_memories = []
            else:
                try:
                    # 逻辑注释：按约定读取 JSON 里的 memory 数组，后续每个元素应包含待写入文本。
                    extracted_memories = json.loads(response, strict=False).get("memory", [])
                # 逻辑注释：LLM 有时不会返回严格 JSON；第一次解析失败后，再尝试从文本中抽取 JSON 片段。
                except json.JSONDecodeError:
                    # 逻辑注释：如果整段不是合法 JSON，就从文本里尽量截取 JSON 片段再解析。
                    extracted_json = extract_json(response)
                    # 逻辑注释：这里保存该阶段的中间结果，供后续抽取、去重、批量写入或返回结果复用。
                    extracted_memories = json.loads(extracted_json, strict=False).get("memory", [])
        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
        except Exception as e:
            # 逻辑注释：解析错误只影响本次抽取结果，不让异常继续破坏调用方流程。
            logger.error(f"Error parsing extraction response (async): {e}")
            # 逻辑注释：这里保存该阶段的中间结果，供后续抽取、去重、批量写入或返回结果复用。
            extracted_memories = []

        # 逻辑注释：LLM 没抽到长期记忆时仍保存原消息上下文，方便下一次抽取时参考最近对话。
        if not extracted_memories:
            # 逻辑注释：保存原始消息到历史上下文库，后续 add 可以利用 last_messages 判断记忆变化。
            await asyncio.to_thread(self.db.save_messages, messages, session_scope)
            # 逻辑注释：该分支没有产生可写入/可返回的记忆，返回空列表而不是报错。
            return []

        # Phase 3: Batch embed all extracted memory texts
        # 逻辑注释：只对非空文本生成 embedding；空文本不会成为记忆，也不浪费 embedding 调用。
        mem_texts = [m.get("text", "") for m in extracted_memories if m.get("text")]
        try:
            # 逻辑注释：批量 embedding 能减少 provider 调用次数，是批处理新增记忆的主要性能优化。
            mem_embeddings_list = await asyncio.to_thread(self.embedding_model.embed_batch, mem_texts, "add")
            # 逻辑注释：把文本和 embedding 建成映射，后面构造记录时可以 O(1) 取向量。
            embed_map = dict(zip(mem_texts, mem_embeddings_list))
        # 逻辑注释：这里进入兜底路径，通常会从批量操作退化为逐条处理，提升整体成功率。
        except Exception:
            # 逻辑注释：批量 embedding 失败后准备逐条兜底，尽量让部分可处理记忆仍能写入。
            embed_map = {}
            # 逻辑注释：逐条 embedding 作为降级路径，牺牲性能换取更高的成功率。
            for text in mem_texts:
                try:
                    # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
                    embed_map[text] = await asyncio.to_thread(self.embedding_model.embed, text, "add")
                # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
                except Exception as e:
                    # 逻辑注释：单条文本 embedding 失败只跳过该条，避免整批新增失败。
                    logger.warning(f"Failed to embed memory text (async): {e}")

        # Phase 4: Per-memory CPU processing + Phase 5: Hash dedup
        # 逻辑注释：收集旧记忆的内容哈希，后面用它快速判断是否已经存过完全相同的文本。
        existing_hashes = set()
        # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
        for mem in existing_results:
            # 逻辑注释：旧记忆 payload 里的 hash 是去重依据，比直接比较所有文本更稳定高效。
            h = mem.payload.get("hash") if hasattr(mem, "payload") and mem.payload else None
            # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
            if h:
                existing_hashes.add(h)

        # 逻辑注释：records 是批量写入的中间结构，集中保存 id、文本、向量和 payload。
        records = []
        # 逻辑注释：本批次内部也要去重，避免 LLM 在一次响应里重复抽取同一事实。
        seen_hashes = set()
        # 逻辑注释：逐条处理 LLM 抽取出的候选记忆，只有通过非空、可 embedding、非重复校验的才会入库。
        for mem in extracted_memories:
            # 逻辑注释：候选记忆以 text 字段为正文；缺失 text 的条目不具备可存储内容。
            text = mem.get("text")
            # 逻辑注释：没有正文或没有成功生成 embedding 的候选都会被跳过，保证后面 records 完整可写。
            if not text or text not in embed_map:
                continue

            # 逻辑注释：使用文本 MD5 作为内容指纹，用于跨批次和批次内的精确重复检测。
            mem_hash = hashlib.md5(text.encode()).hexdigest()
            # 逻辑注释：如果内容哈希已经出现过，就说明是精确重复记忆，跳过以保持记忆库简洁。
            if mem_hash in existing_hashes or mem_hash in seen_hashes:
                logger.debug(f"Skipping duplicate memory (hash match, async): {text[:50]}")
                continue
            seen_hashes.add(mem_hash)

            # 逻辑注释：提前保存词形归一化文本，后续关键词检索无需每次重新处理存量记忆。
            text_lemmatized = lemmatize_for_bm25(text)

            # 逻辑注释：每条记忆用 UUID 作为向量库 ID，保证跨批次新增也不会冲突。
            memory_id = str(uuid.uuid4())
            # 逻辑注释：每条候选记忆都复制一份基础 metadata，再补充该记忆自己的 data/hash/time 等字段。
            mem_metadata = deepcopy(metadata)
            # 逻辑注释：记忆正文放入 payload 的 data 字段，读取和搜索结果格式化都从这里取文本。
            mem_metadata["data"] = text
            # 逻辑注释：把 BM25 用的归一化文本一起存进 payload，服务混合检索。
            mem_metadata["text_lemmatized"] = text_lemmatized
            # 逻辑注释：hash 存入 payload，后续新增时能用旧 hash 快速去重。
            mem_metadata["hash"] = mem_hash
            # 逻辑注释：调用方没有提供创建时间时，使用当前 UTC 时间作为记忆创建时间。
            if "created_at" not in mem_metadata:
                # 逻辑注释：这里保存该阶段的中间结果，供后续抽取、去重、批量写入或返回结果复用。
                mem_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
            # 逻辑注释：新增时更新时间等于创建时间，后续 update 才会改变 updated_at。
            mem_metadata["updated_at"] = mem_metadata["created_at"]
            # 逻辑注释：LLM 如果标出事实归属，就把 attributed_to 写入 payload，便于区分事实属于谁。
            if mem.get("attributed_to"):
                # 逻辑注释：这里保存该阶段的中间结果，供后续抽取、去重、批量写入或返回结果复用。
                mem_metadata["attributed_to"] = mem["attributed_to"]

            # 逻辑注释：通过 records 聚合写入所需的四元组，后续向量插入、历史记录、实体链接都复用它。
            records.append((memory_id, text, embed_map[text], mem_metadata))

        # 逻辑注释：所有候选都被过滤/去重后，只保存上下文消息，不向向量库写任何新记忆。
        if not records:
            # 逻辑注释：保存原始消息到历史上下文库，后续 add 可以利用 last_messages 判断记忆变化。
            await asyncio.to_thread(self.db.save_messages, messages, session_scope)
            # 逻辑注释：该分支没有产生可写入/可返回的记忆，返回空列表而不是报错。
            return []

        # Phase 6: Batch persist
        # 逻辑注释：从 records 拆出向量列表，供向量库批量 insert。
        all_vectors = [r[2] for r in records]
        # 逻辑注释：从 records 拆出 ID 列表，和向量列表一一对应。
        all_ids = [r[0] for r in records]
        # 逻辑注释：从 records 拆出 payload 列表，写入后读取/过滤/关键词检索都依赖这些字段。
        all_payloads = [r[3] for r in records]

        try:
            # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
            await asyncio.to_thread(
                self.vector_store.insert,
                vectors=all_vectors,
                ids=all_ids,
                payloads=all_payloads,
            )
        # 逻辑注释：这里进入兜底路径，通常会从批量操作退化为逐条处理，提升整体成功率。
        except Exception:
            # 逻辑注释：批量插入失败后逐条重试，让部分记忆仍有机会写入成功。
            for mid, vec, pay in zip(all_ids, all_vectors, all_payloads):
                try:
                    # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
                    await asyncio.to_thread(self.vector_store.insert, vectors=[vec], ids=[mid], payloads=[pay])
                # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
                except Exception as e:
                    # 逻辑注释：逐条插入失败才记录错误；这个错误只影响对应 memory_id。
                    logger.error(f"Failed to insert memory {mid} (async): {e}")

        # Batch history
        # 逻辑注释：为每条新增记忆准备历史记录，保证向量库写入后也有可审计的 ADD 事件。
        history_records = [
            {
                "memory_id": r[0],
                "old_memory": None,
                "new_memory": r[1],
                "event": "ADD",
                "created_at": r[3].get("created_at"),
                "is_deleted": 0,
            }
            # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
            for r in records
        ]
        try:
            # 逻辑注释：优先批量写历史，和批量插入一样减少数据库调用。
            await asyncio.to_thread(self.db.batch_add_history, history_records)
        # 逻辑注释：这里进入兜底路径，通常会从批量操作退化为逐条处理，提升整体成功率。
        except Exception:
            # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
            for hr in history_records:
                try:
                    # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
                    await asyncio.to_thread(
                        # 逻辑注释：批量写历史失败后逐条补写，避免完全丢失审计记录。
                        self.db.add_history, hr["memory_id"], None, hr["new_memory"], "ADD",
                        created_at=hr.get("created_at")
                    )
                # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
                except Exception as e:
                    logger.error(f"Failed to add history for {hr['memory_id']} (async): {e}")

        # Phase 7: Batch entity linking
        try:
            # 逻辑注释：实体抽取只需要记忆文本，因此从 records 中取出所有文本做批处理。
            all_texts = [r[1] for r in records]
            # 逻辑注释：从记忆文本抽取实体，只有抽到实体才需要进入实体链接流程。
            all_entities = await asyncio.to_thread(extract_entities_batch, all_texts)

            # 7a: Global dedup
            # 逻辑注释：全局实体表把同批次重复实体合并，并记录它关联的所有 memory_id。
            global_entities = {}
            # 逻辑注释：按 records 顺序把每条记忆和对应实体列表对齐，建立实体到记忆的关系。
            for idx, (memory_id, text, embedding, payload) in enumerate(records):
                # 逻辑注释：如果批量抽取结果长度不完全匹配，就给缺失项空实体列表，避免越界。
                entities = all_entities[idx] if idx < len(all_entities) else []
                # 逻辑注释：逐个处理抽取出的实体，把每个实体都链接到当前记忆。
                for entity_type, entity_text in entities:
                    # 逻辑注释：实体去重用小写+去空白后的规范 key，降低大小写和首尾空格带来的重复。
                    key = entity_text.strip().lower()
                    # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                    if key in global_entities:
                        # 逻辑注释：同一实体已出现时只追加新的 memory_id，不重复保存实体文本。
                        global_entities[key][2].add(memory_id)
                    else:
                        # 逻辑注释：首次遇到实体时保存类型、原文和关联 memory_id 集合，后续用于批量查重/插入。
                        global_entities[key] = [entity_type, entity_text, {memory_id}]

            # 逻辑注释：只有抽到至少一个实体时才进入实体库流程，避免无意义的 embedding/search。
            if global_entities:
                # 逻辑注释：固定实体处理顺序，方便 entity_texts、embeddings 和后续结果按索引对齐。
                ordered_keys = list(global_entities.keys())
                # 逻辑注释：只把实体原文送去 embedding，类型和关联记忆保留在 global_entities 里。
                entity_texts = [global_entities[k][1] for k in ordered_keys]

                # 7b: Batch embed entities
                try:
                    # 逻辑注释：唯一实体批量 embedding，避免同一个实体在一批记忆中重复计算。
                    entity_embeddings = await asyncio.to_thread(self.embedding_model.embed_batch, entity_texts, "add")
                # 逻辑注释：这里进入兜底路径，通常会从批量操作退化为逐条处理，提升整体成功率。
                except Exception:
                    # 逻辑注释：这里保存该阶段的中间结果，供后续抽取、去重、批量写入或返回结果复用。
                    entity_embeddings = []
                    # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
                    for t in entity_texts:
                        try:
                            # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
                            entity_embeddings.append(await asyncio.to_thread(self.embedding_model.embed, t, "add"))
                        # 逻辑注释：这里进入兜底路径，通常会从批量操作退化为逐条处理，提升整体成功率。
                        except Exception:
                            # 逻辑注释：单个实体 embedding 失败时用 None 占位，保持索引对齐并在后面过滤掉。
                            entity_embeddings.append(None)

                # 逻辑注释：过滤掉 embedding 失败的实体，只对有向量的实体做实体库检索。
                valid = [(i, k) for i, k in enumerate(ordered_keys) if entity_embeddings[i] is not None]
                # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                if valid:
                    # 逻辑注释：拆出有效实体的原始索引和 key，便于同时访问 embedding 和实体元数据。
                    valid_indices, valid_keys = zip(*valid)
                    # 逻辑注释：有效实体向量按 valid_keys 顺序排列，后续 search_batch 的返回也按这个顺序对齐。
                    valid_vectors = [entity_embeddings[i] for i in valid_indices]

                    # 7c: Batch search for existing entities
                    # 逻辑注释：有效实体文本和向量一起传给批量搜索，用于判断实体是否已存在。
                    valid_texts = [global_entities[k][1] for k in valid_keys]
                    # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
                    existing_matches = await asyncio.to_thread(
                        self.entity_store.search_batch,
                        queries=valid_texts,
                        vectors_list=valid_vectors,
                        top_k=1,
                        filters=search_filters,
                    )

                    # 7d: Separate into inserts vs updates
                    # 逻辑注释：把需要新建的实体先暂存在列表里，最后统一批量 insert。
                    to_insert_vectors, to_insert_ids, to_insert_payloads = [], [], []
                    # 逻辑注释：逐个有效实体根据搜索结果决定更新已有实体还是加入新建列表。
                    for j, key in enumerate(valid_keys):
                        # 逻辑注释：这里保存该阶段的中间结果，供后续抽取、去重、批量写入或返回结果复用。
                        entity_type, entity_text, memory_ids = global_entities[key]
                        # 逻辑注释：搜索结果按实体顺序对齐；缺失时按空列表处理，表示没有匹配实体。
                        matches = existing_matches[j] if j < len(existing_matches) else []

                        # 逻辑注释：高度相似才认为是同一实体，避免实体索引过度合并。
                        if matches and matches[0].score >= 0.95:
                            # 逻辑注释：这里保存该阶段的中间结果，供后续抽取、去重、批量写入或返回结果复用。
                            match = matches[0]
                            payload = match.payload or {}
                            # 逻辑注释：用 set 合并已有链接和本批新链接，天然去重。
                            linked = set(payload.get("linked_memory_ids", []))
                            # 逻辑注释：把本批中关联该实体的所有 memory_id 合并进已有实体链接。
                            linked |= memory_ids
                            # 逻辑注释：排序后写回 payload，让结果稳定，也便于调试比较。
                            payload["linked_memory_ids"] = sorted(linked)
                            try:
                                # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
                                await asyncio.to_thread(
                                    self.entity_store.update,
                                    vector_id=match.id,
                                    vector=None,
                                    payload=payload,
                                )
                            # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
                            except Exception as e:
                                logger.debug(f"Entity update failed for '{entity_text}' (async): {e}")
                        else:
                            # 逻辑注释：没有匹配实体时，把该实体加入待插入集合，稍后统一写入。
                            to_insert_vectors.append(valid_vectors[j])
                            to_insert_ids.append(str(uuid.uuid4()))
                            # 逻辑注释：新实体 payload 带上实体信息、关联记忆和 session filters，支持后续增强检索和清理。
                            to_insert_payloads.append({
                                "data": entity_text,
                                "entity_type": entity_type,
                                "linked_memory_ids": sorted(memory_ids),
                                **search_filters,
                            })

                    # 7e: Batch insert new entities
                    # 逻辑注释：只有存在新实体时才调用 insert，避免空批次触发某些向量库异常。
                    if to_insert_vectors:
                        try:
                            # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
                            await asyncio.to_thread(
                                self.entity_store.insert,
                                vectors=to_insert_vectors,
                                ids=to_insert_ids,
                                payloads=to_insert_payloads,
                            )
                        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
                        except Exception as e:
                            # 逻辑注释：批量实体插入失败不影响已写入的记忆，只记录警告供排查。
                            logger.warning(f"Batch entity insert failed (async): {e}")
        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
        except Exception as e:
            # 逻辑注释：实体链接属于增强能力，失败时不回滚主记忆写入。
            logger.warning(f"Batch entity linking failed (async): {e}")

        # Phase 8: Save messages + return
        # 逻辑注释：保存原始消息到历史上下文库，后续 add 可以利用 last_messages 判断记忆变化。
        await asyncio.to_thread(self.db.save_messages, messages, session_scope)

        returned_memories = [
            {"id": r[0], "memory": r[1], "event": "ADD"}
            # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
            for r in records
        ]

        # 逻辑注释：遥测前对 filters 做脱敏/编码，只上报维度信息而不是原始实体 ID。
        keys, encoded_ids = process_telemetry_filters(effective_filters)
        # 逻辑注释：记录当前操作的遥测事件，用于观察 API 使用情况和排查性能/行为问题。
        capture_event(
            "mem0.add",
            self,
            # 逻辑注释：保存 API 版本，遥测事件会带上它，便于区分不同版本的行为。
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"},
        )
        # 逻辑注释：返回本次实际新增的记忆列表，前面被跳过/去重的内容不会出现在结果里。
        return returned_memories

    # 逻辑注释：按 memory_id 读取单条记忆，并把系统字段和自定义 metadata 整理成对外稳定的返回结构。
    async def get(self, memory_id):
        """
        Retrieve a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        # 逻辑注释：记录当前操作的遥测事件，用于观察 API 使用情况和排查性能/行为问题。
        capture_event("mem0.get", self, {"memory_id": memory_id, "sync_type": "async"})
        # 逻辑注释：通过向量库 ID 直接取 payload，这是 get/update/delete 的基础读取路径。
        memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        # 逻辑注释：向量库没有返回记录时表示 memory_id 不存在，get 用 None 表达未找到。
        if not memory:
            # 逻辑注释：没有可用结果时显式返回 None，让调用方能区分“没找到”和异常。
            return None

        # 逻辑注释：这些 payload 字段是常用作用域/来源信息，返回时提升到顶层，调用方读取更方便。
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        # 逻辑注释：核心字段和已提升字段不再放进 metadata，避免结果里重复出现同一信息。
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        # 逻辑注释：用 MemoryItem 统一字段名和序列化形态，屏蔽不同向量库返回对象的差异。
        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload.get("data", ""),
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump()

        # 逻辑注释：遍历可提升字段，只有 payload 里真的存在时才加入返回结果。
        for key in promoted_payload_keys:
            # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
            if key in memory.payload:
                # 逻辑注释：把作用域/角色字段放到结果顶层，方便用户直接过滤或展示。
                result_item[key] = memory.payload[key]

        # 逻辑注释：除系统字段外的 payload 都视为用户自定义 metadata，保留在 metadata 子对象里。
        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        # 逻辑注释：只有存在额外 metadata 时才添加 metadata 字段，保持返回结构简洁。
        if additional_metadata:
            # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
            result_item["metadata"] = additional_metadata

        # 逻辑注释：返回已经格式化过的结果，调用方无需理解向量库原始 payload 结构。
        return result_item

    # 逻辑注释：列出某个作用域下的记忆；先校验 filters/top_k，再委托向量库 list 并格式化结果。
    async def get_all(
        self,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        **kwargs,
    ):
        """
        List all memories.

        Args:
            filters (dict): Filter dict containing entity IDs and optional metadata filters.
                Must contain at least one of: user_id, agent_id, run_id.
                Example: filters={"user_id": "u1", "agent_id": "a1"}
            top_k (int, optional): The maximum number of memories to return. Defaults to 20.

        Returns:
            dict: A dictionary containing a list of memories under the "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", ...}]}`

        Raises:
            ValueError: If filters doesn't contain at least one of user_id, agent_id, run_id,
                or if top_k is invalid.
        """
        # Reject top-level entity params - must use filters instead
        # 逻辑注释：兼容性层面拒绝 user_id 等顶层参数，统一要求调用方通过 filters 指定作用域。
        _reject_top_level_entity_params(kwargs, "get_all")

        # Validate top_k
        # 逻辑注释：在触达向量库前校验 top_k/threshold，错误更早、更清晰。
        _validate_search_params(top_k=top_k)

        # Validate and trim entity IDs in filters
        # 逻辑注释：复制 filters 后再修改，避免 trim 或高级过滤转换影响调用方原对象。
        effective_filters = dict(filters) if filters else {}
        # 逻辑注释：如果 filters 里包含实体 ID，就先做同样的清洗校验，保证查询作用域格式正确。
        if "user_id" in effective_filters:
            # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        # 逻辑注释：如果 filters 里包含实体 ID，就先做同样的清洗校验，保证查询作用域格式正确。
        if "agent_id" in effective_filters:
            # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        # 逻辑注释：如果 filters 里包含实体 ID，就先做同样的清洗校验，保证查询作用域格式正确。
        if "run_id" in effective_filters:
            # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )

        # Validate filters contains at least one entity ID
        # 逻辑注释：读取/搜索必须至少限定一个实体作用域，防止默认扫描整个记忆库。
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
                "Example: filters={'user_id': 'u1'}"
            )

        # 逻辑注释：内部统一用 limit 表示最终返回条数，和向量库参数命名保持一致。
        limit = top_k

        # 逻辑注释：遥测前对 filters 做脱敏/编码，只上报维度信息而不是原始实体 ID。
        keys, encoded_ids = process_telemetry_filters(effective_filters)
        # 逻辑注释：记录当前操作的遥测事件，用于观察 API 使用情况和排查性能/行为问题。
        capture_event(
            "mem0.get_all", self, {"limit": limit, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"}
        )

        # 逻辑注释：实际 list 和格式化下沉到 helper，get_all 本身只处理校验和返回包装。
        all_memories_result = await self._get_all_from_vector_store(effective_filters, limit)

        # 逻辑注释：对外统一用 results 包一层，保持 add/search/get_all 等接口返回结构一致。
        return {"results": all_memories_result}

    # 逻辑注释：兼容不同向量库 list 返回结构，统一展开为 MemoryItem 列表，同时保留额外 metadata。
    async def _get_all_from_vector_store(self, filters, limit):
        # 逻辑注释：按 filters 从向量库列出记忆，top_k/limit 控制最多返回多少条。
        memories_result = await asyncio.to_thread(self.vector_store.list, filters=filters, top_k=limit)

        # Handle different vector store return formats by inspecting first element
        # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
        if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0:
            # 逻辑注释：检查第一个元素的类型，用来判断向量库返回的是嵌套列表还是扁平列表。
            first_element = memories_result[0]

            # If first element is a container, unwrap one level
            # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
            if isinstance(first_element, (list, tuple)):
                # 逻辑注释：如果第一层包了一层列表，就展开一层得到真正的记忆对象列表。
                actual_memories = first_element
            else:
                # First element is a memory object, structure is already flat
                # 逻辑注释：如果返回已经是扁平结构，就直接使用，不做额外变换。
                actual_memories = memories_result
        else:
            # 逻辑注释：如果返回已经是扁平结构，就直接使用，不做额外变换。
            actual_memories = memories_result

        # 逻辑注释：这些 payload 字段是常用作用域/来源信息，返回时提升到顶层，调用方读取更方便。
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        # 逻辑注释：核心字段和已提升字段不再放进 metadata，避免结果里重复出现同一信息。
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        # 逻辑注释：统一把向量库对象转换成 SDK 对外返回的字典列表。
        formatted_memories = []
        # 逻辑注释：逐条格式化记忆对象，处理字段提升和额外 metadata。
        for mem in actual_memories:
            # 逻辑注释：用 MemoryItem 统一字段名和序列化形态，屏蔽不同向量库返回对象的差异。
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
            ).model_dump(exclude={"score"})

            # 逻辑注释：遍历可提升字段，只有 payload 里真的存在时才加入返回结果。
            for key in promoted_payload_keys:
                # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                if key in mem.payload:
                    # 逻辑注释：把作用域/角色字段放到结果顶层，方便用户直接过滤或展示。
                    memory_item_dict[key] = mem.payload[key]

            # 逻辑注释：除系统字段外的 payload 都视为用户自定义 metadata，保留在 metadata 子对象里。
            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            # 逻辑注释：只有存在额外 metadata 时才添加 metadata 字段，保持返回结构简洁。
            if additional_metadata:
                # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        # 逻辑注释：返回已经格式化过的结果，调用方无需理解向量库原始 payload 结构。
        return formatted_memories

    # 逻辑注释：搜索入口负责校验和预处理 filters，再调用混合检索；可选 rerank 会在初排结果上二次排序。
    async def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        threshold: float = 0.1,
        rerank: bool = False,
        **kwargs,
    ):
        """
        Searches for memories based on a query.

        Args:
            query (str): Query to search for.
            top_k (int, optional): Maximum number of results to return. Defaults to 20.
            filters (dict): Filter dict containing entity IDs and optional metadata filters.
                Must contain at least one of: user_id, agent_id, run_id.
                Example: filters={"user_id": "u1", "agent_id": "a1"}

                Enhanced metadata filtering with operators:
                - {"key": "value"} - exact match
                - {"key": {"eq": "value"}} - equals
                - {"key": {"ne": "value"}} - not equals
                - {"key": {"in": ["val1", "val2"]}} - in list
                - {"key": {"nin": ["val1", "val2"]}} - not in list
                - {"key": {"gt": 10}} - greater than
                - {"key": {"gte": 10}} - greater than or equal
                - {"key": {"lt": 10}} - less than
                - {"key": {"lte": 10}} - less than or equal
                - {"key": {"contains": "text"}} - contains text
                - {"key": {"icontains": "text"}} - case-insensitive contains
                - {"key": "*"} - wildcard match (any value)
                - {"AND": [filter1, filter2]} - logical AND
                - {"OR": [filter1, filter2]} - logical OR
                - {"NOT": [filter1]} - logical NOT
            threshold (float, optional): Minimum score for a memory to be included. Defaults to 0.1.
            rerank (bool, optional): Whether to rerank results. Defaults to False.

        Returns:
            dict: A dictionary containing the search results under a "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`

        Raises:
            ValueError: If filters doesn't contain at least one of user_id, agent_id, run_id,
                or if threshold/top_k values are invalid.
        """
        # Reject top-level entity params - must use filters instead
        # 逻辑注释：兼容性层面拒绝 user_id 等顶层参数，统一要求调用方通过 filters 指定作用域。
        _reject_top_level_entity_params(kwargs, "search")

        # Validate search parameters (before applying defaults)
        # 逻辑注释：在触达向量库前校验 top_k/threshold，错误更早、更清晰。
        _validate_search_params(threshold=threshold, top_k=top_k)

        # Validate and trim entity IDs in filters
        # 逻辑注释：复制 filters 后再修改，避免 trim 或高级过滤转换影响调用方原对象。
        effective_filters = filters.copy() if filters else {}
        # 逻辑注释：如果 filters 里包含实体 ID，就先做同样的清洗校验，保证查询作用域格式正确。
        if "user_id" in effective_filters:
            # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        # 逻辑注释：如果 filters 里包含实体 ID，就先做同样的清洗校验，保证查询作用域格式正确。
        if "agent_id" in effective_filters:
            # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        # 逻辑注释：如果 filters 里包含实体 ID，就先做同样的清洗校验，保证查询作用域格式正确。
        if "run_id" in effective_filters:
            # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )

        # Validate filters contains at least one entity ID
        # 逻辑注释：读取/搜索必须至少限定一个实体作用域，防止默认扫描整个记忆库。
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
                "Example: filters={'user_id': 'u1'}"
            )

        # 逻辑注释：内部统一用 limit 表示最终返回条数，和向量库参数命名保持一致。
        limit = top_k

        # Apply enhanced metadata filtering if advanced operators are detected
        # 逻辑注释：检测到高级过滤语法时先转换成向量库兼容格式，否则简单 filters 直接透传。
        if self._has_advanced_operators(effective_filters):
            # 逻辑注释：把 AND/OR/NOT、比较操作符等高级语义转换成内部统一表达。
            processed_filters = self._process_metadata_filters(effective_filters)
            # Remove logical/operator keys that have been reprocessed
            # 逻辑注释：转换后移除原始逻辑操作符，避免同一个条件被同时以新旧两种格式传给向量库。
            for logical_key in ("AND", "OR", "NOT"):
                # 逻辑注释：已被转换的复杂字段从原 filters 删除，保持最终 filters 只有向量库能理解的结构。
                effective_filters.pop(logical_key, None)
            # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
            for fk in list(effective_filters.keys()):
                # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                if fk not in ("AND", "OR", "NOT", "user_id", "agent_id", "run_id") and isinstance(effective_filters.get(fk), dict):
                    # 逻辑注释：已被转换的复杂字段从原 filters 删除，保持最终 filters 只有向量库能理解的结构。
                    effective_filters.pop(fk, None)
            # 逻辑注释：把转换后的高级过滤条件合并回有效 filters，后续检索统一使用这份结果。
            effective_filters.update(processed_filters)

        # 逻辑注释：遥测前对 filters 做脱敏/编码，只上报维度信息而不是原始实体 ID。
        keys, encoded_ids = process_telemetry_filters(effective_filters)
        # 逻辑注释：记录当前操作的遥测事件，用于观察 API 使用情况和排查性能/行为问题。
        capture_event(
            "mem0.search",
            self,
            {
                "limit": limit,
                # 逻辑注释：保存 API 版本，遥测事件会带上它，便于区分不同版本的行为。
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "async",
                "threshold": threshold,
                "advanced_filters": bool(filters and self._has_advanced_operators(filters)),
            },
        )

        # 逻辑注释：底层搜索会完成语义、关键词、实体增强的融合排序，search 入口只负责调用。
        original_memories = await self._search_vector_store(query, effective_filters, limit, threshold)

        # Apply reranking if enabled and reranker is available
        # 逻辑注释：只有用户开启 rerank、实例也配置了 reranker 且已有初排结果时才做二次排序。
        if rerank and self.reranker and original_memories:
            try:
                # Run reranking in thread pool to avoid blocking async loop
                # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
                reranked_memories = await asyncio.to_thread(
                    # 逻辑注释：reranker 根据原始 query 对候选记忆重新排序，通常能提升相关性但会增加成本。
                    self.reranker.rerank, query, original_memories, limit
                )
                # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
                original_memories = reranked_memories
            # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
            except Exception as e:
                # 逻辑注释：重排失败不影响搜索可用性，直接退回初排结果。
                logger.warning(f"Reranking failed, using original results: {e}")

        # 逻辑注释：对外统一用 results 包一层，保持 add/search/get_all 等接口返回结构一致。
        return {"results": original_memories}

    # 逻辑注释：把平台层的增强过滤语法转换成向量库更容易消费的格式，并支持 AND/OR/NOT 组合。
    def _process_metadata_filters(self, metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process enhanced metadata filters and convert them to vector store compatible format.

        Args:
            metadata_filters: Enhanced metadata filters with operators

        Returns:
            Dict of processed filters compatible with vector store
        """
        # 逻辑注释：转换结果单独累积，最后再替换/合并到有效 filters 里。
        processed_filters = {}

        # 逻辑注释：定义 process_condition，封装这段业务逻辑，减少外部调用方理解内部细节的成本。
        def process_condition(key: str, condition: Any) -> Dict[str, Any]:
            # 逻辑注释：非 dict 条件代表简单等值匹配，是最常见、最直接的过滤形式。
            if not isinstance(condition, dict):
                # Simple equality: {"key": "value"}
                # 逻辑注释：星号表示通配字段，具体如何匹配由底层向量库适配层处理。
                if condition == "*":
                    # Wildcard: match everything for this field (implementation depends on vector store)
                    return {key: "*"}
                return {key: condition}

            result = {}
            # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
            for operator, value in condition.items():
                # Map platform operators to universal format that can be translated by each vector store
                # 逻辑注释：建立平台操作符到内部操作符的映射，当前两边同名，但保留了适配空间。
                operator_map = {
                    "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte",
                    "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                    "contains": "contains", "icontains": "icontains"
                }

                # 逻辑注释：只允许白名单里的操作符，避免未知过滤语法被静默传到向量库。
                if operator in operator_map:
                    # 逻辑注释：同一个字段可能有多个比较条件，用嵌套 dict 合并到同一字段下。
                    result.setdefault(key, {})[operator_map[operator]] = value
                else:
                    # 逻辑注释：遇到不支持的操作符立即报错，避免用户以为过滤生效但实际被忽略。
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
            return result

        # 逻辑注释：定义 merge_filters，封装这段业务逻辑，减少外部调用方理解内部细节的成本。
        def merge_filters(target: Dict[str, Any], source: Dict[str, Any]) -> None:
            """Merge source into target, deep-merging nested operator dicts for the same key."""
            # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
            for key, value in source.items():
                # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    # 逻辑注释：同一个字段的多个操作符合并到一起，例如 gte 和 lte 可以同时存在。
                    target[key].update(value)
                else:
                    # 逻辑注释：字段不存在或不是同类嵌套结构时，直接写入目标 filters。
                    target[key] = value

        # 逻辑注释：逐个处理原始 filters 条目，普通字段和逻辑操作符分开转换。
        for key, value in metadata_filters.items():
            # 逻辑注释：AND 语义是所有子条件同时成立，所以可以直接合并到同一个 filters 对象里。
            if key == "AND":
                # Logical AND: combine multiple conditions
                # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                if not isinstance(value, list):
                    # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
                    raise ValueError("AND operator requires a list of conditions")
                # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
                for condition in value:
                    for sub_key, sub_value in condition.items():
                        merge_filters(processed_filters, process_condition(sub_key, sub_value))
            # 逻辑注释：这是对前面判断的补充分支，用来兼容另一种输入/配置形态。
            elif key == "OR":
                # Logical OR: Pass through to vector store for implementation-specific handling
                # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                if not isinstance(value, list) or not value:
                    # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
                    raise ValueError("OR operator requires a non-empty list of conditions")
                # Store OR conditions in a way that vector stores can interpret
                # 逻辑注释：$or 保存多个备选条件，每个条件内部仍按普通字段规则转换。
                processed_filters["$or"] = []
                # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
                for condition in value:
                    or_condition = {}
                    # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
                    for sub_key, sub_value in condition.items():
                        merge_filters(or_condition, process_condition(sub_key, sub_value))
                    # 逻辑注释：$or 保存多个备选条件，每个条件内部仍按普通字段规则转换。
                    processed_filters["$or"].append(or_condition)
            # 逻辑注释：这是对前面判断的补充分支，用来兼容另一种输入/配置形态。
            elif key == "NOT":
                # Logical NOT: Pass through to vector store for implementation-specific handling
                # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                if not isinstance(value, list) or not value:
                    # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
                    raise ValueError("NOT operator requires a non-empty list of conditions")
                # 逻辑注释：$not 保存需要排除的条件集合，交给底层适配层处理。
                processed_filters["$not"] = []
                # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
                for condition in value:
                    not_condition = {}
                    # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
                    for sub_key, sub_value in condition.items():
                        merge_filters(not_condition, process_condition(sub_key, sub_value))
                    # 逻辑注释：$not 保存需要排除的条件集合，交给底层适配层处理。
                    processed_filters["$not"].append(not_condition)
            else:
                merge_filters(processed_filters, process_condition(key, value))

        return processed_filters

    # 逻辑注释：轻量判断 filters 是否包含高级操作符，用来决定是否需要进入转换流程。
    def _has_advanced_operators(self, filters: Dict[str, Any]) -> bool:
        """
        Check if filters contain advanced operators that need special processing.

        Args:
            filters: Dictionary of filters to check

        Returns:
            bool: True if advanced operators are detected
        """
        # 逻辑注释：非 dict filters 不可能包含高级过滤语法，直接返回 False。
        if not isinstance(filters, dict):
            return False

        # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
        for key, value in filters.items():
            # Check for platform-style logical operators
            # 逻辑注释：出现逻辑操作符就说明需要高级过滤转换。
            if key in ["AND", "OR", "NOT"]:
                return True
            # Check for comparison operators (without $ prefix for universal compatibility)
            # 逻辑注释：字段值是 dict 时可能包含 eq/gt/in 等比较操作符，需要继续检查。
            if isinstance(value, dict):
                # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
                for op in value.keys():
                    # 逻辑注释：命中任意比较/包含操作符，就判定 filters 使用了高级语法。
                    if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                        return True
            # Check for wildcard values
            # 逻辑注释：通配符也属于增强过滤语义，需要走转换逻辑。
            if value == "*":
                return True
        return False

    # 逻辑注释：底层混合检索：语义向量召回、关键词 BM25、实体增强一起打分，再统一排序和格式化。
    async def _search_vector_store(self, query, filters, limit, threshold=0.1):
        # 逻辑注释：兼容旧调用可能传 None 的情况，统一回落到默认阈值 0.1。
        if threshold is None:
            # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
            threshold = 0.1

        # Step 1: Preprocess query (CPU-bound)
        # 逻辑注释：查询文本也做词形归一化，保证和写入时保存的 text_lemmatized 在同一空间比较。
        query_lemmatized = await asyncio.to_thread(lemmatize_for_bm25, query)
        # 逻辑注释：从记忆文本抽取实体，只有抽到实体才需要进入实体链接流程。
        query_entities = await asyncio.to_thread(extract_entities, query)

        # Step 2: Embed query
        # 逻辑注释：查询向量用于语义召回，能找到表述不同但含义相近的记忆。
        embeddings = await asyncio.to_thread(self.embedding_model.embed, query, "search")

        # Step 3: Semantic search (over-fetch)
        # 逻辑注释：先多召回一些候选，再融合 BM25/实体分数排序，避免早期截断错过好结果。
        internal_limit = max(limit * 4, 60)
        # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
        semantic_results = await asyncio.to_thread(
            # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
            self.vector_store.search, query=query, vectors=embeddings, top_k=internal_limit, filters=filters
        )

        # Step 4: Keyword search (if store supports it)
        # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
        keyword_results = await asyncio.to_thread(
            # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
            self.vector_store.keyword_search, query=query_lemmatized, top_k=internal_limit, filters=filters
        )

        # Step 5: Compute BM25 scores
        # 逻辑注释：BM25 分数单独按 memory_id 保存，后面和语义分数融合。
        bm25_scores = {}
        # 逻辑注释：有些向量库可能不支持 keyword_search；None 表示跳过关键词分支。
        if keyword_results is not None:
            # 逻辑注释：根据查询长度/形态选择归一化参数，把 BM25 原始分数压到可融合区间。
            midpoint, steepness = get_bm25_params(query, lemmatized=query_lemmatized)
            # 逻辑注释：逐条读取关键词检索结果，将不同返回对象格式统一成 memory_id 和 raw_score。
            for mem in keyword_results:
                # 逻辑注释：兼容对象式和 dict 式结果，统一转成字符串 ID 作为打分 key。
                mem_id = str(mem.id) if hasattr(mem, 'id') else str(mem.get('id', ''))
                # 逻辑注释：同样兼容对象式/dict 式 score 字段，避免绑定某一种向量库返回类型。
                raw_score = mem.score if hasattr(mem, 'score') else mem.get('score', 0)
                # 逻辑注释：只有正向关键词匹配分才参与融合，零分或空值不会影响排序。
                if raw_score and raw_score > 0:
                    # 逻辑注释：把 BM25 原始分归一化，和语义/实体分数处在可比较尺度上。
                    bm25_scores[mem_id] = normalize_bm25(raw_score, midpoint, steepness)

        # Step 6: Compute entity boosts
        # 逻辑注释：实体增强默认为空；没有抽到查询实体时，最终排序不会受到实体分支影响。
        entity_boosts = {}
        # 逻辑注释：只有查询里有实体时才访问实体库，减少普通搜索的额外开销。
        if query_entities:
            # 逻辑注释：实体增强会把命中实体关联的记忆额外加分，让精确实体相关结果更靠前。
            entity_boosts = await self._compute_entity_boosts_async(query_entities, filters)

        # Step 7: Build candidate set from semantic results
        # 逻辑注释：把语义召回结果转换成统一候选结构，供 score_and_rank 融合排序。
        candidates = []
        # 逻辑注释：遍历语义候选，保留 id、语义分和 payload，后续格式化也依赖 payload。
        for mem in semantic_results:
            # 逻辑注释：兼容对象式和 dict 式结果，统一转成字符串 ID 作为打分 key。
            mem_id = str(mem.id)
            candidates.append({
                "id": mem_id,
                "score": mem.score,
                # 逻辑注释：payload 里包含记忆正文、metadata、hash 等返回所需信息。
                "payload": mem.payload if hasattr(mem, 'payload') else {},
            })

        # Step 8: Score and rank
        # 逻辑注释：统一融合语义分、BM25 分和实体 boost，并按阈值/top_k 截断。
        scored_results = score_and_rank(
            semantic_results=candidates,
            bm25_scores=bm25_scores,
            entity_boosts=entity_boosts,
            threshold=threshold,
            top_k=limit,
        )

        # Step 9: Format results
        # 逻辑注释：这些 payload 字段是常用作用域/来源信息，返回时提升到顶层，调用方读取更方便。
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        # 逻辑注释：核心字段和已提升字段不再放进 metadata，避免结果里重复出现同一信息。
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
        original_memories = []
        # 逻辑注释：只格式化融合排序后的最终结果，而不是所有召回候选。
        for scored in scored_results:
            # 逻辑注释：从候选中安全取 payload；缺失时用空 dict 防止字段访问异常。
            payload = scored.get("payload") or {}
            # 逻辑注释：没有 data 的候选不是有效记忆文本，跳过避免返回空 memory。
            if not payload.get("data"):
                continue

            # 逻辑注释：用 MemoryItem 统一字段名和序列化形态，屏蔽不同向量库返回对象的差异。
            memory_item_dict = MemoryItem(
                id=scored["id"],
                memory=payload.get("data", ""),
                hash=payload.get("hash"),
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at"),
                score=scored["score"],
            ).model_dump()

            # 逻辑注释：遍历可提升字段，只有 payload 里真的存在时才加入返回结果。
            for key in promoted_payload_keys:
                # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                if key in payload:
                    # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
                    memory_item_dict[key] = payload[key]

            # 逻辑注释：除系统字段外的 payload 都视为用户自定义 metadata，保留在 metadata 子对象里。
            additional_metadata = {k: v for k, v in payload.items() if k not in core_and_promoted_keys}
            # 逻辑注释：只有存在额外 metadata 时才添加 metadata 字段，保持返回结构简洁。
            if additional_metadata:
                # 逻辑注释：向量库没有返回记录时表示 memory_id 不存在，get 用 None 表达未找到。
                if not memory_item_dict.get("metadata"):
                    # 逻辑注释：这里构造中间变量来统一不同输入/后端返回形态，后续逻辑只依赖规范化后的结构。
                    memory_item_dict["metadata"] = {}
                # 逻辑注释：把额外 metadata 合并到结果对象，既保留系统字段，又不丢调用方自定义字段。
                memory_item_dict["metadata"].update(additional_metadata)

            original_memories.append(memory_item_dict)

        # 逻辑注释：返回已经格式化过的结果，调用方无需理解向量库原始 payload 结构。
        return original_memories

    # 逻辑注释：异步版实体增强计算，把 embedding 和向量库查询放到线程池，避免阻塞事件循环。
    async def _compute_entity_boosts_async(self, query_entities, filters):
        """Async version of entity boost computation."""
        # 逻辑注释：用集合在单条文本内去重，避免同一个实体重复 upsert。
        seen = set()
        # 逻辑注释：实体增强前先准备去重后的实体列表，避免重复查询同一实体。
        deduped = []
        # 逻辑注释：最多处理前 8 个实体，防止复杂查询触发过多实体库查询。
        for entity_type, entity_text in query_entities[:8]:
            # 逻辑注释：实体去重用小写+去空白后的规范 key，降低大小写和首尾空格带来的重复。
            key = entity_text.strip().lower()
            # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
            if key and key not in seen:
                seen.add(key)
                # 逻辑注释：只把非空且未见过的实体加入待查询列表。
                deduped.append((entity_type, entity_text))

        # 逻辑注释：去重后没有实体时无需访问实体库，直接返回空 boost。
        if not deduped:
            # 逻辑注释：没有实体增强可用时返回空映射，后续融合打分自然退化为普通检索。
            return {}

        # 逻辑注释：实体检索只使用 session 级作用域字段，保证实体链接不会跨用户/agent/run 串数据。
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        # 逻辑注释：最终按 memory_id 保存 boost，多个实体命中同一记忆时取最大值。
        memory_boosts = {}

        try:
            # 逻辑注释：逐个查询实体库，每个实体都可能为一批关联记忆提供加分。
            for _, entity_text in deduped:
                # 逻辑注释：实体也需要单独向量化，才能在实体库里用相似度判断是否已有同一实体。
                entity_embedding = await asyncio.to_thread(self.embedding_model.embed, entity_text, "search")
                # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
                matches = await asyncio.to_thread(
                    self.entity_store.search,
                    query=entity_text,
                    vectors=entity_embedding,
                    top_k=500,
                    filters=search_filters,
                )

                # 逻辑注释：这里按集合顺序逐项处理，通常是为了把批量输入拆成可验证、可写入或可格式化的单元。
                for match in matches:
                    similarity = match.score if hasattr(match, 'score') else 0.0
                    # 逻辑注释：实体匹配太弱时不加分，避免噪声实体影响搜索排序。
                    if similarity < 0.5:
                        continue

                    payload = match.payload if hasattr(match, 'payload') else {}
                    # 逻辑注释：实体节点的反向链接列表告诉我们哪些记忆与该实体有关。
                    linked_memory_ids = payload.get("linked_memory_ids", [])
                    # 逻辑注释：链接字段异常时跳过该实体，避免坏 payload 影响搜索。
                    if not isinstance(linked_memory_ids, list):
                        continue

                    # 逻辑注释：实体关联的记忆越多，越可能是泛化实体，需要降低单条记忆的 boost。
                    num_linked = max(len(linked_memory_ids), 1)
                    # 逻辑注释：用扩散衰减权重抑制“高频实体”造成的过度加分。
                    memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
                    # 逻辑注释：最终实体 boost 同时考虑实体相似度、全局权重和扩散衰减。
                    boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight

                    # 逻辑注释：把同一实体带来的 boost 分发到它关联的每条记忆上。
                    for memory_id in linked_memory_ids:
                        # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
                        if memory_id:
                            memory_key = str(memory_id)
                            # 逻辑注释：同一记忆被多个实体命中时取最大 boost，避免简单累加导致多实体查询过度放大。
                            memory_boosts[memory_key] = max(memory_boosts.get(memory_key, 0.0), boost)

        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
        except Exception as e:
            # 逻辑注释：实体增强失败时保留普通混合检索结果，搜索功能不中断。
            logger.warning(f"Entity boost computation failed: {e}")

        return memory_boosts

    # 逻辑注释：更新入口先生成新文本 embedding，再交给内部方法处理向量、metadata、历史和实体索引同步。
    async def update(self, memory_id, data, metadata: Optional[Dict[str, Any]] = None):
        """
        Update a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to update.
            data (str): New content to update the memory with.
            metadata (dict, optional): Metadata to update with the memory. Defaults to None.

        Returns:
            dict: Success message indicating the memory was updated.

        Example:
            >>> await m.update(memory_id="mem_123", data="Likes to play tennis on weekends")
            {'message': 'Memory updated successfully!'}
        """
        # 逻辑注释：记录当前操作的遥测事件，用于观察 API 使用情况和排查性能/行为问题。
        capture_event("mem0.update", self, {"memory_id": memory_id, "sync_type": "async"})

        # 逻辑注释：查询向量用于语义召回，能找到表述不同但含义相近的记忆。
        embeddings = await asyncio.to_thread(self.embedding_model.embed, data, "update")
        # 逻辑注释：提前计算新文本 embedding，并用 dict 传给内部更新方法，避免重复计算。
        existing_embeddings = {data: embeddings}

        # 逻辑注释：内部更新方法负责真正修改向量库、写历史并同步实体索引。
        await self._update_memory(memory_id, data, existing_embeddings, metadata)
        return {"message": "Memory updated successfully!"}

    # 逻辑注释：删除入口先确认 memory_id 存在，再删除向量记录并写入删除历史。
    async def delete(self, memory_id):
        """
        Delete a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        # 逻辑注释：记录当前操作的遥测事件，用于观察 API 使用情况和排查性能/行为问题。
        capture_event("mem0.delete", self, {"memory_id": memory_id, "sync_type": "async"})

        # 逻辑注释：通过向量库 ID 直接取 payload，这是 get/update/delete 的基础读取路径。
        existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        # 逻辑注释：找不到旧记忆时不能继续更新/删除，必须向调用方报告无效 memory_id。
        if existing_memory is None:
            # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
            raise ValueError(f"Memory with id {memory_id} not found")

        # 逻辑注释：内部删除方法统一处理向量库删除、历史记录和实体索引清理。
        await self._delete_memory(memory_id, existing_memory)
        return {"message": "Memory deleted successfully!"}

    # 逻辑注释：按作用域批量删除记忆；要求至少一个实体过滤条件，避免误删整个库。
    async def delete_all(self, user_id=None, agent_id=None, run_id=None):
        """
        Delete all memories asynchronously.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters = {}
        # 逻辑注释：有 user_id 时同时写入 metadata 和 filters，新增记忆和查询旧记忆会落在同一个用户作用域。
        if user_id:
            filters["user_id"] = user_id
        # 逻辑注释：agent_id 也参与存储和过滤，支持按 agent 维度隔离记忆。
        if agent_id:
            filters["agent_id"] = agent_id
        # 逻辑注释：run_id 用于一次运行/会话级别的隔离，适合临时任务或批处理场景。
        if run_id:
            filters["run_id"] = run_id

        # 逻辑注释：没有任何过滤条件时拒绝批量删除，避免误删所有记忆；全量清空必须显式调用 reset。
        if not filters:
            # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        # 逻辑注释：遥测前对 filters 做脱敏/编码，只上报维度信息而不是原始实体 ID。
        keys, encoded_ids = process_telemetry_filters(filters)
        # 逻辑注释：记录当前操作的遥测事件，用于观察 API 使用情况和排查性能/行为问题。
        capture_event("mem0.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"})
        # 逻辑注释：先列出当前作用域下所有记忆，再逐条走统一删除逻辑，确保历史和实体清理不遗漏。
        memories = await asyncio.to_thread(self.vector_store.list, filters=filters)

        delete_tasks = []
        # 逻辑注释：逐条删除可以复用 _delete_memory 的审计和实体清理流程。
        for memory in memories[0]:
            # 逻辑注释：内部删除方法统一处理向量库删除、历史记录和实体索引清理。
            delete_tasks.append(self._delete_memory(memory.id))

        await asyncio.gather(*delete_tasks)

        logger.info(f"Deleted {len(memories[0])} memories")

        return {"message": "Memories deleted successfully!"}

    # 逻辑注释：读取某条记忆的变更历史，方便审计 ADD/UPDATE/DELETE 过程。
    async def history(self, memory_id):
        """
        Get the history of changes for a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        # 逻辑注释：记录当前操作的遥测事件，用于观察 API 使用情况和排查性能/行为问题。
        capture_event("mem0.history", self, {"memory_id": memory_id, "sync_type": "async"})
        # 逻辑注释：返回 memory_id 让上层可以继续记录、链接实体或给用户展示操作结果。
        return await asyncio.to_thread(self.db.get_history, memory_id)

    # 逻辑注释：创建单条记忆的通用 helper：生成 ID、补齐 metadata/hash/time、写向量库并记录历史。
    async def _create_memory(self, data, existing_embeddings, metadata=None):
        # 逻辑注释：创建前打 debug 日志，调试时可看到即将写入的记忆正文。
        logger.debug(f"Creating memory with {data=}")
        # 逻辑注释：如果上层已传入新文本 embedding，就直接复用，避免二次 embedding 调用。
        if data in existing_embeddings:
            # 逻辑注释：复用调用方已经计算好的 embedding，减少重复计算和 provider 成本。
            embeddings = existing_embeddings[data]
        else:
            # 逻辑注释：查询向量用于语义召回，能找到表述不同但含义相近的记忆。
            embeddings = await asyncio.to_thread(self.embedding_model.embed, data, memory_action="add")

        # 逻辑注释：每条记忆用 UUID 作为向量库 ID，保证跨批次新增也不会冲突。
        memory_id = str(uuid.uuid4())
        # 逻辑注释：更新时先从调用方新 metadata 开始，再补齐系统字段和旧作用域字段。
        new_metadata = deepcopy(metadata) if metadata is not None else {}
        # 逻辑注释：更新后的正文写回 data 字段，读取和搜索都会看到新文本。
        new_metadata["data"] = data
        # 逻辑注释：更新后重新计算内容 hash，保证后续去重依据和新文本一致。
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
        if "created_at" not in new_metadata:
            # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
            new_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        new_metadata["updated_at"] = new_metadata["created_at"]
        # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data)

        # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
        await asyncio.to_thread(
            self.vector_store.insert,
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[new_metadata],
        )

        # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            None,
            data,
            "ADD",
            created_at=new_metadata.get("created_at"),
            updated_at=new_metadata.get("updated_at"),
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )

        # 逻辑注释：返回 memory_id 让上层可以继续记录、链接实体或给用户展示操作结果。
        return memory_id

    # 逻辑注释：把一段对话压缩成“过程性记忆”再存储，适合记录 agent 的长期操作流程。
    async def _create_procedural_memory(self, messages, metadata=None, llm=None, prompt=None):
        """
        Create a procedural memory asynchronously

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            llm (llm, optional): LLM to use for the procedural memory creation. Defaults to None.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        try:
            from langchain_core.messages.utils import (
                convert_to_messages,  # type: ignore
            )
        # 逻辑注释：这里进入兜底路径，通常会从批量操作退化为逐条处理，提升整体成功率。
        except Exception:
            logger.error(
                "Import error while loading langchain-core. Please install 'langchain-core' to use procedural memory."
            )
            raise

        # 逻辑注释：过程性记忆生成前记录日志，因为它会调用 LLM 做总结，成本和普通写入不同。
        logger.info("Creating procedural memory")

        # 逻辑注释：构造用于过程性记忆的消息序列：系统提示、原对话、最后的总结指令。
        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {"role": "user", "content": "Create procedural memory of the above conversation."},
        ]

        try:
            # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
            if llm is not None:
                parsed_messages = convert_to_messages(parsed_messages)
                # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
                response = await asyncio.to_thread(llm.invoke, input=parsed_messages)
                procedural_memory = response.content
            else:
                # 逻辑注释：调用 LLM 把整段对话总结成可长期保存的流程/操作记忆。
                procedural_memory = await asyncio.to_thread(self.llm.generate_response, messages=parsed_messages)
                # 逻辑注释：去掉 LLM 可能包上的代码块标记，存储时只保留纯文本记忆。
                procedural_memory = remove_code_blocks(procedural_memory)
        
        # 逻辑注释：捕获通用异常用于记录上下文；是否继续取决于该步骤是不是主路径必需。
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        # 逻辑注释：过程性记忆必须有 metadata/作用域，否则总结出来的流程无法归属到具体 agent/run。
        if metadata is None:
            # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
            raise ValueError("Metadata cannot be done for procedural memory.")

        # 逻辑注释：在原 metadata 基础上标记 memory_type，后续可区分普通事实记忆和过程性记忆。
        metadata = {**metadata, "memory_type": MemoryType.PROCEDURAL.value}
        # 逻辑注释：查询向量用于语义召回，能找到表述不同但含义相近的记忆。
        embeddings = await asyncio.to_thread(self.embedding_model.embed, procedural_memory, memory_action="add")
        # 逻辑注释：过程性记忆最终仍按普通记忆写入向量库和历史表，只是正文来自 LLM 总结。
        memory_id = await self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        # 逻辑注释：记录当前操作的遥测事件，用于观察 API 使用情况和排查性能/行为问题。
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "async"})

        # 逻辑注释：按 add 接口的返回格式包装过程性记忆创建结果。
        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    # 逻辑注释：内部更新流程不仅改向量和 payload，还保留创建时间/作用域，记录历史，并重建相关实体链接。
    async def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
        logger.info(f"Updating memory with {data=}")

        try:
            # 逻辑注释：通过向量库 ID 直接取 payload，这是 get/update/delete 的基础读取路径。
            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        # 逻辑注释：这里进入兜底路径，通常会从批量操作退化为逐条处理，提升整体成功率。
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        # 逻辑注释：找不到旧记忆时不能继续更新/删除，必须向调用方报告无效 memory_id。
        if existing_memory is None:
            # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
            raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")

        # 逻辑注释：保存旧文本，后面写历史记录时能形成 old_memory → new_memory 的变更链。
        prev_value = existing_memory.payload.get("data")

        # 逻辑注释：更新时先从调用方新 metadata 开始，再补齐系统字段和旧作用域字段。
        new_metadata = deepcopy(metadata) if metadata is not None else {}

        # 逻辑注释：更新后的正文写回 data 字段，读取和搜索都会看到新文本。
        new_metadata["data"] = data
        # 逻辑注释：更新后重新计算内容 hash，保证后续去重依据和新文本一致。
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data)
        # 逻辑注释：更新不能改变原创建时间，因此从旧 payload 继承 created_at。
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        # 逻辑注释：更新时间使用当前 UTC 时间，表示这次 update 的发生时间。
        new_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Preserve session identifiers from existing memory only if not provided in new metadata
        # 逻辑注释：如果调用方没显式覆盖 user_id，就沿用旧记忆的 user_id，避免更新后丢失作用域。
        if "user_id" not in new_metadata and "user_id" in existing_memory.payload:
            # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        # 逻辑注释：agent_id 同样默认继承旧值，保持记忆仍在原 agent 作用域内。
        if "agent_id" not in new_metadata and "agent_id" in existing_memory.payload:
            # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        # 逻辑注释：run_id 默认继承旧值，避免单条更新把记忆移出原运行范围。
        if "run_id" not in new_metadata and "run_id" in existing_memory.payload:
            # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
            new_metadata["run_id"] = existing_memory.payload["run_id"]

        # 逻辑注释：actor_id 来自原消息说话人，更新时继续保留，除非业务另行处理。
        if "actor_id" in existing_memory.payload:
            # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        # 逻辑注释：role 默认继承旧值，让更新后的记忆仍知道原始消息角色。
        if "role" not in new_metadata and "role" in existing_memory.payload:
            # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
            new_metadata["role"] = existing_memory.payload["role"]

        # 逻辑注释：如果上层已传入新文本 embedding，就直接复用，避免二次 embedding 调用。
        if data in existing_embeddings:
            # 逻辑注释：复用调用方已经计算好的 embedding，减少重复计算和 provider 成本。
            embeddings = existing_embeddings[data]
        else:
            # 逻辑注释：查询向量用于语义召回，能找到表述不同但含义相近的记忆。
            embeddings = await asyncio.to_thread(self.embedding_model.embed, data, "update")

        # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
        await asyncio.to_thread(
            self.vector_store.update,
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )

        # Entity-store cleanup: strip this memory's id from old-text entities,
        # then re-extract entities from the new text and link them back.
        # 逻辑注释：从更新后的 metadata 提取 session filters，用于限定实体清理/重建的作用域。
        session_filters = {k: new_metadata[k] for k in ("user_id", "agent_id", "run_id") if new_metadata.get(k)}
        # 逻辑注释：先把该 memory_id 从旧实体链接里移除，避免旧文本实体继续影响搜索。
        await self._remove_memory_from_entity_store(memory_id, session_filters)
        # 逻辑注释：再按新文本重新抽实体并链接，让实体索引和更新后的记忆保持一致。
        await self._link_entities_for_memory(memory_id, data, session_filters)

        # 逻辑注释：返回 memory_id 让上层可以继续记录、链接实体或给用户展示操作结果。
        return memory_id

    # 逻辑注释：内部删除流程把向量库删除和历史审计打包在一起，并同步清理实体反向索引。
    async def _delete_memory(self, memory_id, existing_memory=None):
        # 逻辑注释：删除前记录目标 ID，方便调试删除链路。
        logger.info(f"Deleting memory with {memory_id=}")
        # 逻辑注释：找不到旧记忆时不能继续更新/删除，必须向调用方报告无效 memory_id。
        if existing_memory is None:
            # 逻辑注释：通过向量库 ID 直接取 payload，这是 get/update/delete 的基础读取路径。
            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
            # 逻辑注释：找不到旧记忆时不能继续更新/删除，必须向调用方报告无效 memory_id。
            if existing_memory is None:
                # 逻辑注释：发现调用方式或内部状态不满足要求时立即抛错，避免错误数据继续进入存储/检索流程。
                raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")
        # 逻辑注释：删除历史需要保留被删除前的文本，因此先从 payload 取出旧值。
        prev_value = existing_memory.payload.get("data", "")
        # 逻辑注释：删除历史里保留原创建时间，并统一带时区时间到 UTC，方便审计排序。
        created_at = _normalize_iso_timestamp_to_utc(existing_memory.payload.get("created_at"))
        # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
        updated_at = datetime.now(timezone.utc).isoformat()
        # 逻辑注释：旧 payload 可能为空，用空 dict 兜底以便安全提取 session filters。
        payload = existing_memory.payload or {}
        # 逻辑注释：这里补齐或保存记忆生命周期相关字段，确保写入、更新、删除都有一致的元信息。
        session_filters = {k: payload[k] for k in ("user_id", "agent_id", "run_id") if payload.get(k)}

        # 逻辑注释：先从向量库删除主记忆，后面再写 DELETE 历史记录。
        await asyncio.to_thread(self.vector_store.delete, vector_id=memory_id)
        # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            None,
            "DELETE",
            created_at=created_at,
            updated_at=updated_at,
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )

        # Entity-store cleanup: strip this memory's id from any entity records
        # that linked to it. Non-fatal — the helper swallows errors.
        # 逻辑注释：先把该 memory_id 从旧实体链接里移除，避免旧文本实体继续影响搜索。
        await self._remove_memory_from_entity_store(memory_id, session_filters)

        # 逻辑注释：返回 memory_id 让上层可以继续记录、链接实体或给用户展示操作结果。
        return memory_id

    # 逻辑注释：重置整个记忆系统：清理历史表、重建向量库，并在实体库已初始化时一并重置。
    async def reset(self):
        """
        Reset the memory store asynchronously by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        # 逻辑注释：reset 是破坏性操作，先用 warning 日志提示会清空所有记忆。
        logger.warning("Resetting all memories")
        # 逻辑注释：先从向量库删除主记忆，后面再写 DELETE 历史记录。
        await asyncio.to_thread(self.vector_store.delete_col)

        gc.collect()

        # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
        if hasattr(self.vector_store, "client") and hasattr(self.vector_store.client, "close"):
            # 逻辑注释：这是异步版对阻塞同步调用的包装：把 CPU/IO 工作放到线程池，避免卡住事件循环。
            await asyncio.to_thread(self.vector_store.client.close)

        # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
        if hasattr(self.db, "connection") and self.db.connection:
            # 逻辑注释：先删历史表，确保 reset 后历史状态和向量库状态一致地从空开始。
            await asyncio.to_thread(lambda: self.db.connection.execute("DROP TABLE IF EXISTS history"))
            # 逻辑注释：删除表后关闭旧 SQLite 连接，避免后续继续使用失效连接。
            await asyncio.to_thread(self.db.connection.close)

        # 逻辑注释：SQLite 用来保存消息上下文和变更历史，和向量库形成“语义索引 + 审计记录”的双存储结构。
        self.db = SQLiteManager(self.config.history_db_path)

        # 逻辑注释：创建向量存储后，记忆文本的向量和 payload 都会通过它进行插入、查询、更新和删除。
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )

        # 逻辑注释：记录当前操作的遥测事件，用于观察 API 使用情况和排查性能/行为问题。
        capture_event("mem0.reset", self, {"sync_type": "async"})

    # 逻辑注释：释放 SQLite 等持有的资源，避免长生命周期进程里连接泄漏。
    def close(self):
        """Release resources held by this AsyncMemory instance."""
        # 逻辑注释：这里根据当前状态选择分支，目的是只在满足业务前提时才继续执行后续操作。
        if hasattr(self, "db") and self.db is not None:
            # 逻辑注释：关闭 SQLite 连接，释放文件句柄/锁。
            self.db.close()
            # 逻辑注释：关闭后置空引用，防止后续误用已关闭连接。
            self.db = None

    # 逻辑注释：占位接口，明确当前 Memory 类还没有实现聊天能力。
    async def chat(self, query):
        # 逻辑注释：显式抛出未实现错误，比静默返回更容易让调用方发现该接口不可用。
        raise NotImplementedError("Chat function not implemented yet.")
