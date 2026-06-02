# -*- coding: utf-8 -*-
# 说明：本文件在原始代码基础上补充中文注释；原有注释和代码均已保留。
# 说明：注释尽量按代码执行顺序逐行解释，纯括号、空行、简单参数行等无必要行未额外注释。

# 注释：导入 asyncio 模块，供后续代码使用。
import asyncio
# 注释：导入 gc 模块，供后续代码使用。
import gc
# 注释：导入 hashlib 模块，供后续代码使用。
import hashlib
# 注释：导入 json 模块，供后续代码使用。
import json
# 注释：导入 logging 模块，供后续代码使用。
import logging
# 注释：导入 os 模块，供后续代码使用。
import os
# 注释：导入 uuid 模块，供后续代码使用。
import uuid
# 注释：导入 warnings 模块，供后续代码使用。
import warnings
# 注释：从 copy 模块导入 deepcopy。
from copy import deepcopy
# 注释：从 datetime 模块导入 datetime, timezone。
from datetime import datetime, timezone
# 注释：从 typing 模块导入 Any, Dict, Optional。
from typing import Any, Dict, Optional

# 注释：从 pydantic 模块导入 ValidationError。
from pydantic import ValidationError

# 注释：从 mem0.configs.base 模块导入 MemoryConfig, MemoryItem。
from mem0.configs.base import MemoryConfig, MemoryItem
# 注释：从 mem0.configs.enums 模块导入 MemoryType。
from mem0.configs.enums import MemoryType
# 注释：从 mem0.configs.prompts 模块批量导入后续列出的对象。
from mem0.configs.prompts import (
    ADDITIVE_EXTRACTION_PROMPT,
    AGENT_CONTEXT_SUFFIX,
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
    generate_additive_extraction_prompt,
)
# 注释：从 mem0.exceptions 模块导入 ValidationError as Mem0ValidationError。
from mem0.exceptions import ValidationError as Mem0ValidationError
# 注释：从 mem0.memory.base 模块导入 MemoryBase。
from mem0.memory.base import MemoryBase
# 注释：从 mem0.memory.setup 模块导入 mem0_dir, setup_config。
from mem0.memory.setup import mem0_dir, setup_config
# 注释：从 mem0.memory.storage 模块导入 SQLiteManager。
from mem0.memory.storage import SQLiteManager
# 注释：从 mem0.memory.telemetry 模块导入 MEM0_TELEMETRY, capture_event。
from mem0.memory.telemetry import MEM0_TELEMETRY, capture_event
# 注释：从 mem0.memory.utils 模块批量导入后续列出的对象。
from mem0.memory.utils import (
    extract_json,
    parse_messages,
    parse_vision_messages,
    process_telemetry_filters,
    remove_code_blocks,
)
# 注释：从 mem0.utils.entity_extraction 模块导入 extract_entities, extract_entities_batch。
from mem0.utils.entity_extraction import extract_entities, extract_entities_batch
# 注释：从 mem0.utils.factory 模块批量导入后续列出的对象。
from mem0.utils.factory import (
    EmbedderFactory,
    LlmFactory,
    RerankerFactory,
    VectorStoreFactory,
)
# 注释：从 mem0.utils.lemmatization 模块导入 lemmatize_for_bm25。
from mem0.utils.lemmatization import lemmatize_for_bm25
# 注释：从 mem0.utils.scoring 模块批量导入后续列出的对象。
from mem0.utils.scoring import (
    ENTITY_BOOST_WEIGHT,
    get_bm25_params,
    normalize_bm25,
    score_and_rank,
)

# Suppress SWIG deprecation warnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")
# 注释：配置 warnings 过滤规则。
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")

# Initialize logger early for util functions
logger = logging.getLogger(__name__)


# Fields that hold runtime auth/connection objects and must be preserved.
# These are non-serializable objects (e.g. AWSV4SignerAuth, RequestsHttpConnection)
# needed by clients like OpenSearch — not sensitive strings to redact.
_RUNTIME_FIELDS = frozenset({
    "http_auth",
    "auth",
    "connection_class",
    "ssl_context",
})

# Fields that are known to contain sensitive secrets and must be redacted.
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
_SENSITIVE_SUFFIXES = (
    "_password",
    "_secret",
    "_token",
    "_credential",
    "_credentials",
)

# Entity parameters that must be passed via filters, not top-level kwargs
ENTITY_PARAMS = frozenset({"user_id", "agent_id", "run_id"})


# 注释：禁止在顶层参数中直接传入实体 ID。
def _reject_top_level_entity_params(kwargs: Dict[str, Any], method_name: str) -> None:
    """Reject top-level entity parameters - must use filters instead."""
    # 注释：计算并保存 invalid_keys 变量，供后续逻辑使用。
    invalid_keys = ENTITY_PARAMS & set(kwargs.keys())
    # 注释：判断条件 `invalid_keys` 是否成立。
    if invalid_keys:
        # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
        raise ValueError(
            f"Top-level entity parameters {invalid_keys} are not supported in {method_name}(). "
            f"Use filters={{'user_id': '...'}} instead."
        )


# 注释：校验并规范化实体 ID。
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
    # 注释：判断条件 `value is None` 是否成立。
    if value is None:
        # 注释：返回 `None` 给调用方。
        return None
    # 注释：计算并保存 trimmed 变量，供后续逻辑使用。
    trimmed = value.strip()
    # 注释：判断条件 `trimmed == ""` 是否成立。
    if trimmed == "":
        # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
        raise ValueError(
            f"Invalid {name}: cannot be empty or whitespace-only. Provide a valid identifier."
        )
    # 注释：判断条件 `any(c.isspace() for c in trimmed)` 是否成立。
    if any(c.isspace() for c in trimmed):
        # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
        raise ValueError(
            f"Invalid {name}: cannot contain whitespace. Provide a valid identifier without spaces."
        )
    # 注释：返回 `trimmed` 给调用方。
    return trimmed


# 注释：校验搜索参数是否合法。
def _validate_search_params(threshold: Optional[float] = None, top_k: Optional[int] = None) -> None:
    """
    Validates search parameters.

    Args:
        threshold: Similarity threshold (must be between 0 and 1)
        top_k: Number of results to return (must be non-negative integer)

    Raises:
        ValueError: If threshold or top_k are invalid
    """
    # 注释：判断条件 `threshold is not None` 是否成立。
    if threshold is not None:
        # 注释：判断条件 `not isinstance(threshold, (int, float))` 是否成立。
        if not isinstance(threshold, (int, float)):
            # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
            raise ValueError("threshold must be a valid number")
        # 注释：判断条件 `threshold < 0 or threshold > 1` 是否成立。
        if threshold < 0 or threshold > 1:
            # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
            raise ValueError(
                f"Invalid threshold: {threshold}. Must be between 0 and 1 (inclusive)."
            )
    # 注释：判断条件 `top_k is not None` 是否成立。
    if top_k is not None:
        # 注释：判断条件 `not isinstance(top_k, int) or isinstance(top_k, bool)` 是否成立。
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
            raise ValueError("top_k must be a valid integer")
        # 注释：判断条件 `top_k < 0` 是否成立。
        if top_k < 0:
            # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
            raise ValueError(
                f"Invalid top_k: {top_k}. Must be a non-negative integer."
            )


# 注释：判断字段是否属于敏感字段。
def _is_sensitive_field(field_name: str) -> bool:
    """Check if a field should be redacted for telemetry safety.

    Uses a layered approach:
    1. Runtime fields (allowlist) — always preserved, highest priority.
    2. Exact deny list — known secret field names.
    3. Suffix deny list — catches patterns like db_password, auth_secret, etc.
    """
    # 注释：计算并保存 name 变量，供后续逻辑使用。
    name = field_name.lower().strip()
    # 注释：判断条件 `name in _RUNTIME_FIELDS` 是否成立。
    if name in _RUNTIME_FIELDS:
        # 注释：返回 `False` 给调用方。
        return False
    # 注释：判断条件 `name in _SENSITIVE_FIELDS_EXACT` 是否成立。
    if name in _SENSITIVE_FIELDS_EXACT:
        # 注释：返回 `True` 给调用方。
        return True
    # 注释：返回 `any(name.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES)` 给调用方。
    return any(name.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES)


# 注释：安全复制配置对象并处理不可序列化字段。
def _safe_deepcopy_config(config):
    """Safely deepcopy config, falling back to dict-based cloning for non-serializable objects."""
    # 注释：进入可能抛出异常的代码块。
    try:
        # 注释：返回 `deepcopy(config)` 给调用方。
        return deepcopy(config)
    # 注释：捕获 Exception as e 异常并执行降级或错误处理。
    except Exception as e:
        # 注释：输出调试日志。
        logger.debug(f"Deepcopy failed, using dict-based cloning: {e}")

        # 注释：计算并保存 config_class 变量，供后续逻辑使用。
        config_class = type(config)

        # 注释：判断条件 `hasattr(config, "model_dump")` 是否成立。
        if hasattr(config, "model_dump"):
            # 注释：进入可能抛出异常的代码块。
            try:
                # 注释：计算并保存 clone_dict 变量，供后续逻辑使用。
                clone_dict = config.model_dump()
            # 注释：捕获 Exception 异常并执行降级或错误处理。
            except Exception:
                # 注释：计算并保存 clone_dict 变量，供后续逻辑使用。
                clone_dict = dict(config.__dict__)
        # 注释：处理前面条件不成立时的默认分支。
        else:
            # 注释：计算并保存 clone_dict 变量，供后续逻辑使用。
            clone_dict = dict(config.__dict__)

        # Restore runtime fields, redact sensitive ones
        for field_name in list(clone_dict.keys()):
            # 注释：判断条件 `field_name in _RUNTIME_FIELDS and hasattr(config, field_name)` 是否成立。
            if field_name in _RUNTIME_FIELDS and hasattr(config, field_name):
                # 注释：计算并保存 clone_dict 变量，供后续逻辑使用。
                clone_dict[field_name] = getattr(config, field_name)
            # 注释：当前一个条件不成立时，继续判断 `_is_sensitive_field(field_name)`。
            elif _is_sensitive_field(field_name):
                # 注释：计算并保存 clone_dict 变量，供后续逻辑使用。
                clone_dict[field_name] = None

        # 注释：进入可能抛出异常的代码块。
        try:
            # 注释：返回 `config_class(**clone_dict)` 给调用方。
            return config_class(**clone_dict)
        # 注释：捕获 Exception 异常并执行降级或错误处理。
        except Exception:
            # 注释：输出调试日志。
            logger.debug("Config reconstruction failed, returning shallow dict clone")
            # 注释：返回 `type("Config", (), clone_dict)()` 给调用方。
            return type("Config", (), clone_dict)()


# 注释：将带时区的 ISO 时间转换为 UTC。
def _normalize_iso_timestamp_to_utc(timestamp: Optional[str]) -> Optional[str]:
    """Normalize timezone-aware ISO timestamps to UTC without rewriting naive values."""
    # 注释：判断条件 `not timestamp` 是否成立。
    if not timestamp:
        # 注释：返回 `timestamp` 给调用方。
        return timestamp
    # 注释：进入可能抛出异常的代码块。
    try:
        # 注释：计算并保存 parsed 变量，供后续逻辑使用。
        parsed = datetime.fromisoformat(timestamp)
    # 注释：捕获 ValueError 异常并执行降级或错误处理。
    except ValueError:
        # 注释：返回 `timestamp` 给调用方。
        return timestamp
    # 注释：判断条件 `parsed.tzinfo is None` 是否成立。
    if parsed.tzinfo is None:
        # 注释：返回 `timestamp` 给调用方。
        return timestamp
    # 注释：返回 `parsed.astimezone(timezone.utc).isoformat()` 给调用方。
    return parsed.astimezone(timezone.utc).isoformat()


# 注释：构造写入元数据和查询过滤条件。
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

    # 注释：深拷贝生成 base_metadata_template 变量，避免修改原始输入对象。
    base_metadata_template = deepcopy(input_metadata) if input_metadata else {}
    # 注释：深拷贝生成 effective_query_filters 变量，避免修改原始输入对象。
    effective_query_filters = deepcopy(input_filters) if input_filters else {}

    # ---------- validate and add all provided session ids ----------
    session_ids_provided = []

    # Validate and trim entity IDs
    user_id = _validate_and_trim_entity_id(user_id, "user_id")
    # 注释：计算并保存 agent_id 变量，供后续逻辑使用。
    agent_id = _validate_and_trim_entity_id(agent_id, "agent_id")
    # 注释：计算并保存 run_id 变量，供后续逻辑使用。
    run_id = _validate_and_trim_entity_id(run_id, "run_id")

    # 注释：判断条件 `user_id` 是否成立。
    if user_id:
        # 注释：计算并保存 base_metadata_template 变量，供后续逻辑使用。
        base_metadata_template["user_id"] = user_id
        # 注释：计算并保存 effective_query_filters 变量，供后续逻辑使用。
        effective_query_filters["user_id"] = user_id
        # 注释：调用 session_ids_provided.append 执行对应操作。
        session_ids_provided.append("user_id")

    # 注释：判断条件 `agent_id` 是否成立。
    if agent_id:
        # 注释：计算并保存 base_metadata_template 变量，供后续逻辑使用。
        base_metadata_template["agent_id"] = agent_id
        # 注释：计算并保存 effective_query_filters 变量，供后续逻辑使用。
        effective_query_filters["agent_id"] = agent_id
        # 注释：调用 session_ids_provided.append 执行对应操作。
        session_ids_provided.append("agent_id")

    # 注释：判断条件 `run_id` 是否成立。
    if run_id:
        # 注释：计算并保存 base_metadata_template 变量，供后续逻辑使用。
        base_metadata_template["run_id"] = run_id
        # 注释：计算并保存 effective_query_filters 变量，供后续逻辑使用。
        effective_query_filters["run_id"] = run_id
        # 注释：调用 session_ids_provided.append 执行对应操作。
        session_ids_provided.append("run_id")

    # 注释：判断条件 `not session_ids_provided` 是否成立。
    if not session_ids_provided:
        # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
        raise Mem0ValidationError(
            message="At least one of 'user_id', 'agent_id', or 'run_id' must be provided.",
            error_code="VALIDATION_001",
            details={"provided_ids": {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}},
            suggestion="Please provide at least one identifier to scope the memory operation."
        )

    # ---------- optional actor filter ----------
    resolved_actor_id = actor_id or effective_query_filters.get("actor_id")
    # 注释：判断条件 `resolved_actor_id` 是否成立。
    if resolved_actor_id:
        # 注释：计算并保存 effective_query_filters 变量，供后续逻辑使用。
        effective_query_filters["actor_id"] = resolved_actor_id

    # 注释：返回 `base_metadata_template, effective_query_filters` 给调用方。
    return base_metadata_template, effective_query_filters


# 注释：根据过滤条件构造确定性的会话作用域字符串。
def _build_session_scope(filters):
    """Build deterministic session scope string from entity IDs."""
    # 注释：初始化 parts 变量 为空列表，用于后续收集数据。
    parts = []
    # 注释：遍历 sorted(["user_id", "agent_id", "run_id"]) 中的元素，并将当前项赋给 key。
    for key in sorted(["user_id", "agent_id", "run_id"]):
        # 注释：计算并保存 val 变量，供后续逻辑使用。
        val = filters.get(key)
        # 注释：判断条件 `val` 是否成立。
        if val:
            # 注释：调用 parts.append 执行对应操作。
            parts.append(f"{key}={val}")
    # 注释：返回 `"&".join(parts)` 给调用方。
    return "&".join(parts)


# 注释：初始化 mem0 运行配置。
setup_config()
# 注释：计算并保存 日志记录器，供后续逻辑使用。
logger = logging.getLogger(__name__)


# 注释：定义 Memory 类，并继承/使用 (MemoryBase) 中的基础能力。
class Memory(MemoryBase):
    # 注释：初始化 Memory 实例并创建模型、向量库、数据库等依赖。
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        # 注释：设置当前实例的 config 属性，用于后续方法共享状态。
        self.config = config

        # 注释：设置当前实例的 embedding_model 属性，用于后续方法共享状态。
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        # 注释：设置当前实例的 vector_store 属性，用于后续方法共享状态。
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        # 注释：设置当前实例的 llm 属性，用于后续方法共享状态。
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        # 注释：设置当前实例的 db 属性，用于后续方法共享状态。
        self.db = SQLiteManager(self.config.history_db_path)
        # 注释：设置当前实例的 collection_name 属性，用于后续方法共享状态。
        self.collection_name = self.config.vector_store.config.collection_name
        # 注释：设置当前实例的 api_version 属性，用于后续方法共享状态。
        self.api_version = self.config.version
        # 注释：设置当前实例的 custom_instructions 属性，用于后续方法共享状态。
        self.custom_instructions = self.config.custom_instructions

        # Initialize reranker if configured
        self.reranker = None
        # 注释：判断条件 `config.reranker` 是否成立。
        if config.reranker:
            # 注释：设置当前实例的 reranker 属性，用于后续方法共享状态。
            self.reranker = RerankerFactory.create(
                config.reranker.provider,
                config.reranker.config
            )

        # Entity store is initialized lazily on first use
        self._entity_store = None

        # 注释：判断条件 `MEM0_TELEMETRY` 是否成立。
        if MEM0_TELEMETRY:
            # Create telemetry config manually to avoid deepcopy issues with thread locks
            telemetry_config_dict = {}
            # 注释：判断条件 `hasattr(self.config.vector_store.config, 'model_dump')` 是否成立。
            if hasattr(self.config.vector_store.config, 'model_dump'):
                # For pydantic models
                telemetry_config_dict = self.config.vector_store.config.model_dump()
            # 注释：处理前面条件不成立时的默认分支。
            else:
                # For other objects, manually copy common attributes
                for attr in ['host', 'port', 'path', 'api_key', 'index_name', 'dimension', 'metric']:
                    # 注释：判断条件 `hasattr(self.config.vector_store.config, attr)` 是否成立。
                    if hasattr(self.config.vector_store.config, attr):
                        # 注释：计算并保存 telemetry_config_dict 变量，供后续逻辑使用。
                        telemetry_config_dict[attr] = getattr(self.config.vector_store.config, attr)

            # Override collection name for telemetry
            telemetry_config_dict['collection_name'] = "mem0migrations"

            # Set path for file-based vector stores
            telemetry_config = _safe_deepcopy_config(self.config.vector_store.config)
            # 注释：判断条件 `self.config.vector_store.provider in ["faiss", "qdrant"]` 是否成立。
            if self.config.vector_store.provider in ["faiss", "qdrant"]:
                # 注释：计算并保存 provider_path 变量，供后续逻辑使用。
                provider_path = f"migrations_{self.config.vector_store.provider}"
                # 注释：计算并保存 telemetry_config_dict 变量，供后续逻辑使用。
                telemetry_config_dict['path'] = os.path.join(mem0_dir, provider_path)
                # 注释：确保目标目录存在。
                os.makedirs(telemetry_config_dict['path'], exist_ok=True)

            # Create the config object using the same class as the original
            telemetry_config = self.config.vector_store.config.__class__(**telemetry_config_dict)
            # 注释：设置当前实例的 _telemetry_vector_store 属性，用于后续方法共享状态。
            self._telemetry_vector_store = VectorStoreFactory.create(
                self.config.vector_store.provider, telemetry_config
            )
        # 注释：记录遥测事件，便于统计调用行为。
        capture_event("mem0.init", self, {"sync_type": "sync"})

    # 注释：应用 property 装饰器，调整下面定义的函数或属性行为。
    @property
    # 注释：定义 entity_store 函数/方法，封装一段可复用逻辑。
    def entity_store(self):
        """Lazily initialize entity store on first use."""
        # 注释：判断条件 `self._entity_store is None` 是否成立。
        if self._entity_store is None:
            # 注释：深拷贝生成 entity_config 变量，避免修改原始输入对象。
            entity_config = _safe_deepcopy_config(self.config.vector_store.config)
            # 注释：计算并保存 entity_collection 变量，供后续逻辑使用。
            entity_collection = f"{self.collection_name}_entities"
            # Set collection name on the cloned config
            if hasattr(entity_config, 'collection_name'):
                # 注释：计算并保存 collection_name 变量，供后续逻辑使用。
                entity_config.collection_name = entity_collection
            # 注释：当前一个条件不成立时，继续判断 `isinstance(entity_config, dict)`。
            elif isinstance(entity_config, dict):
                # 注释：计算并保存 entity_config 变量，供后续逻辑使用。
                entity_config['collection_name'] = entity_collection
            # For Qdrant, share the existing client to avoid RocksDB lock contention
            # when using embedded mode (path=...). QdrantConfig.client takes precedence
            # over host/port/path.
            if self.config.vector_store.provider == "qdrant" and hasattr(self.vector_store, "client"):
                # 注释：判断条件 `hasattr(entity_config, "client")` 是否成立。
                if hasattr(entity_config, "client"):
                    # 注释：计算并保存 client 变量，供后续逻辑使用。
                    entity_config.client = self.vector_store.client
                # 注释：当前一个条件不成立时，继续判断 `isinstance(entity_config, dict)`。
                elif isinstance(entity_config, dict):
                    # 注释：计算并保存 entity_config 变量，供后续逻辑使用。
                    entity_config["client"] = self.vector_store.client
            # 注释：设置当前实例的 _entity_store 属性，用于后续方法共享状态。
            self._entity_store = VectorStoreFactory.create(
                self.config.vector_store.provider, entity_config
            )
        # 注释：返回 `self._entity_store` 给调用方。
        return self._entity_store

    # 注释：新增或更新实体索引记录。
    def _upsert_entity(self, entity_text, entity_type, memory_id, filters):
        """Upsert an entity into the entity store, linking it to a memory."""
        # 注释：进入可能抛出异常的代码块。
        try:
            # 注释：计算并保存 实体向量，供后续逻辑使用。
            entity_embedding = self.embedding_model.embed(entity_text, "add")
            # 注释：计算并保存 检索过滤条件，供后续逻辑使用。
            search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}

            # 注释：计算并保存 existing 变量，供后续逻辑使用。
            existing = self.entity_store.search(
                query=entity_text,
                vectors=entity_embedding,
                top_k=1,
                filters=search_filters,
            )

            # 注释：判断条件 `existing and existing[0].score >= 0.95` 是否成立。
            if existing and existing[0].score >= 0.95:
                # Update existing entity's linked_memory_ids
                match = existing[0]
                # 注释：计算并保存 向量库载荷数据，供后续逻辑使用。
                payload = match.payload or {}
                # 注释：计算并保存 linked_ids 变量，供后续逻辑使用。
                linked_ids = payload.get("linked_memory_ids", [])
                # 注释：判断条件 `memory_id not in linked_ids` 是否成立。
                if memory_id not in linked_ids:
                    # 注释：调用 linked_ids.append 执行对应操作。
                    linked_ids.append(memory_id)
                    # 注释：计算并保存 向量库载荷数据，供后续逻辑使用。
                    payload["linked_memory_ids"] = linked_ids
                    # 注释：调用 self.entity_store.update 执行对应操作。
                    self.entity_store.update(
                        vector_id=match.id,
                        vector=None,
                        payload=payload,
                    )
            # 注释：处理前面条件不成立时的默认分支。
            else:
                # Create new entity
                entity_id = str(uuid.uuid4())
                # 注释：计算并保存 entity_payload 变量，供后续逻辑使用。
                entity_payload = {
                    "data": entity_text,
                    "entity_type": entity_type,
                    "linked_memory_ids": [memory_id],
                    **{k: v for k, v in search_filters.items()},
                }
                # 注释：调用 self.entity_store.insert 执行对应操作。
                self.entity_store.insert(
                    vectors=[entity_embedding],
                    ids=[entity_id],
                    payloads=[entity_payload],
                )
        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
        except Exception as e:
            # 注释：输出警告日志。
            logger.warning(f"Entity upsert failed for '{entity_text}': {e}")

    # 注释：从实体索引中移除某条记忆的关联。
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
        # 注释：判断条件 `self._entity_store is None` 是否成立。
        if self._entity_store is None:
            # 注释：结束函数并返回空值。
            return
        # 注释：计算并保存 检索过滤条件，供后续逻辑使用。
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        # 注释：进入可能抛出异常的代码块。
        try:
            # 注释：计算并保存 listed 变量，供后续逻辑使用。
            listed = self.entity_store.list(filters=search_filters, top_k=10000)
            # 注释：计算并保存 rows 变量，供后续逻辑使用。
            rows = listed[0] if isinstance(listed, (list, tuple)) and listed and isinstance(listed[0], list) else listed
            # 注释：遍历 rows or [] 中的元素，并将当前项赋给 row。
            for row in rows or []:
                # 注释：进入可能抛出异常的代码块。
                try:
                    # 注释：计算并保存 向量库载荷数据，供后续逻辑使用。
                    payload = getattr(row, "payload", None) or {}
                    # 注释：计算并保存 linked 变量，供后续逻辑使用。
                    linked = payload.get("linked_memory_ids", [])
                    # 注释：判断条件 `not isinstance(linked, list) or memory_id not in linked` 是否成立。
                    if not isinstance(linked, list) or memory_id not in linked:
                        # 注释：跳过本轮循环剩余逻辑，继续处理下一项。
                        continue
                    # 注释：计算并保存 remaining 变量，供后续逻辑使用。
                    remaining = [mid for mid in linked if mid != memory_id]
                    # 注释：判断条件 `not remaining` 是否成立。
                    if not remaining:
                        # 注释：进入可能抛出异常的代码块。
                        try:
                            # 注释：调用 self.entity_store.delete 执行对应操作。
                            self.entity_store.delete(vector_id=row.id)
                        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
                        except Exception as e:
                            # 注释：输出调试日志。
                            logger.debug(f"Entity delete failed for id={row.id}: {e}")
                    # 注释：处理前面条件不成立时的默认分支。
                    else:
                        # 注释：计算并保存 实体文本，供后续逻辑使用。
                        entity_text = payload.get("data")
                        # 注释：判断条件 `not isinstance(entity_text, str) or not entity_text` 是否成立。
                        if not isinstance(entity_text, str) or not entity_text:
                            # 注释：输出调试日志。
                            logger.debug(f"Entity id={row.id} missing 'data'; skipping update during cleanup")
                            # 注释：跳过本轮循环剩余逻辑，继续处理下一项。
                            continue
                        # 注释：进入可能抛出异常的代码块。
                        try:
                            # 注释：计算并保存 vec 变量，供后续逻辑使用。
                            vec = self.embedding_model.embed(entity_text, "update")
                        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
                        except Exception as e:
                            # 注释：输出调试日志。
                            logger.debug(f"Entity re-embed failed for '{entity_text}': {e}")
                            # 注释：跳过本轮循环剩余逻辑，继续处理下一项。
                            continue
                        # 注释：计算并保存 new_payload 变量，供后续逻辑使用。
                        new_payload = {**payload, "linked_memory_ids": remaining}
                        # 注释：进入可能抛出异常的代码块。
                        try:
                            # 注释：调用 self.entity_store.update 执行对应操作。
                            self.entity_store.update(
                                vector_id=row.id,
                                vector=vec,
                                payload=new_payload,
                            )
                        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
                        except Exception as e:
                            # 注释：输出调试日志。
                            logger.debug(f"Entity update failed for id={row.id}: {e}")
                # 注释：捕获 Exception as e 异常并执行降级或错误处理。
                except Exception as e:
                    # 注释：输出调试日志。
                    logger.debug(f"Entity cleanup error: {e}")
        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
        except Exception as e:
            # 注释：输出警告日志。
            logger.warning(f"Entity store cleanup failed for memory_id={memory_id}: {e}")

    # 注释：抽取记忆文本中的实体并建立关联。
    def _link_entities_for_memory(self, memory_id, text, filters):
        """Extract entities from `text` and link them to `memory_id` in the
        entity store, scoped to `filters`. Simpler single-memory variant of
        Phase 7 in add(): per-entity search-then-update-or-insert via the
        existing `_upsert_entity` helper. Non-fatal on any failure.
        """
        # 注释：进入可能抛出异常的代码块。
        try:
            # 注释：计算并保存 entities 变量，供后续逻辑使用。
            entities = extract_entities(text)
            # 注释：判断条件 `not entities` 是否成立。
            if not entities:
                # 注释：结束函数并返回空值。
                return
            # 注释：初始化 seen 变量 为空集合，用于后续去重。
            seen = set()
            # 注释：遍历 entities 中的元素，并将当前项赋给 entity_type, entity_text。
            for entity_type, entity_text in entities:
                # 注释：计算并保存 key 变量，供后续逻辑使用。
                key = entity_text.strip().lower()
                # 注释：判断条件 `not key or key in seen` 是否成立。
                if not key or key in seen:
                    # 注释：跳过本轮循环剩余逻辑，继续处理下一项。
                    continue
                # 注释：调用 seen.add 执行对应操作。
                seen.add(key)
                # 注释：进入可能抛出异常的代码块。
                try:
                    # 注释：调用 self._upsert_entity 执行对应操作。
                    self._upsert_entity(entity_text, entity_type, memory_id, filters)
                # 注释：捕获 Exception as e 异常并执行降级或错误处理。
                except Exception as e:
                    # 注释：输出调试日志。
                    logger.debug(f"Entity link failed for '{entity_text}': {e}")
        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
        except Exception as e:
            # 注释：输出警告日志。
            logger.warning(f"Entity linking failed for memory_id={memory_id}: {e}")

    # 注释：应用 classmethod 装饰器，调整下面定义的函数或属性行为。
    @classmethod
    # 注释：根据字典配置创建 Memory 实例。
    def from_config(cls, config_dict: Dict[str, Any]):
        # 注释：进入可能抛出异常的代码块。
        try:
            # 注释：计算并保存 配置对象，供后续逻辑使用。
            config = cls._process_config(config_dict)
            # 注释：计算并保存 配置对象，供后续逻辑使用。
            config = MemoryConfig(**config_dict)
        # 注释：捕获 ValidationError as e 异常并执行降级或错误处理。
        except ValidationError as e:
            # 注释：输出错误日志。
            logger.error(f"Configuration validation error: {e}")
            # 注释：执行当前语句，推进该函数的业务流程。
            raise
        # 注释：返回 `cls(config)` 给调用方。
        return cls(config)

    # 注释：应用 staticmethod 装饰器，调整下面定义的函数或属性行为。
    @staticmethod
    # 注释：预处理配置字典。
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        # 注释：进入可能抛出异常的代码块。
        try:
            # 注释：返回 `config_dict` 给调用方。
            return config_dict
        # 注释：捕获 ValidationError as e 异常并执行降级或错误处理。
        except ValidationError as e:
            # 注释：输出错误日志。
            logger.error(f"Configuration validation error: {e}")
            # 注释：执行当前语句，推进该函数的业务流程。
            raise

    # 注释：定义 _should_use_agent_memory_extraction 函数/方法，封装一段可复用逻辑。
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

    # 注释：新增记忆入口，负责校验输入并写入记忆。
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

        # Step 1: 构造有效的 metadata 和 filters，用于后续存储和检索
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
        )

        # Step 2: 校验 memory_type，如果指定了 type 且不是 procedural_memory，抛出异常
        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
            raise Mem0ValidationError(
                message=f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories.",
                error_code="VALIDATION_002",
                details={"provided_type": memory_type, "valid_type": MemoryType.PROCEDURAL.value},
                suggestion=f"Use '{MemoryType.PROCEDURAL.value}' to create procedural memories."
            )

        # Step 3: 规范化 messages 输入
        if isinstance(messages, str):
            # 如果是字符串，封装为 [{"role": "user", "content": messages}]
            messages = [{"role": "user", "content": messages}]

        # 注释：当前一个条件不成立时，继续判断 `isinstance(messages, dict)`。
        elif isinstance(messages, dict):
            # 如果是单个字典，封装为列表
            messages = [messages]

        # 注释：当前一个条件不成立时，继续判断 `not isinstance(messages, list)`。
        elif not isinstance(messages, list):
            # 非 list / dict / str 输入类型抛出异常
            raise Mem0ValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries."
            )

        # Step 4: 如果是 procedural memory 并且提供了 agent_id，则调用专门的创建函数
        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            # 注释：计算并保存 结果集合，供后续逻辑使用。
            results = self._create_procedural_memory(messages, metadata=processed_metadata, prompt=prompt)
            # 注释：返回 `results` 给调用方。
            return results  # 直接返回，不走一般 conversation memory 流程

        # Step 5: 如果 LLM 支持视觉输入，则解析 vision 消息；否则做普通解析
        if self.config.llm.config.get("enable_vision"):
            # 注释：计算并保存 消息列表，供后续逻辑使用。
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        # 注释：处理前面条件不成立时的默认分支。
        else:
            # 注释：计算并保存 消息列表，供后续逻辑使用。
            messages = parse_vision_messages(messages)

        # Step 6: 调用核心函数 _add_to_vector_store()，完成 memory 抽取 / embed / persist / entity linking
        vector_store_result = self._add_to_vector_store(messages, processed_metadata, effective_filters, infer, prompt=prompt)

        # Step 7: 返回结构化结果，通常包含 {"results": [{id, memory, event:"ADD"}]}
        return {"results": vector_store_result}

    # 注释：将消息转换为记忆并写入向量存储。
    def _add_to_vector_store(self, messages, metadata, filters, infer, prompt=None):
        # Step 0: 判断是否启用 infer。
        # infer=False 表示不让 LLM 抽取 memory，而是把原始 messages 直接作为 memory 存进去。
        if not infer:
            # Step 0.1: 用于收集最终返回的 memory 结果。
            returned_memories = []

            # Step 0.2: 逐条处理输入 messages。
            for message_dict in messages:
                # Step 0.3: 校验每条 message 的格式。
                # 要求必须是 dict，并且至少包含 role 和 content。
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    # 注释：输出警告日志。
                    logger.warning(f"Skipping invalid message format: {message_dict}")
                    # 注释：跳过本轮循环剩余逻辑，继续处理下一项。
                    continue

                # Step 0.4: system 消息不作为 memory 存储，直接跳过。
                if message_dict["role"] == "system":
                    # 注释：跳过本轮循环剩余逻辑，继续处理下一项。
                    continue

                # Step 0.5: 为当前 message 复制一份 metadata，避免修改外部传入的 metadata。
                per_msg_meta = deepcopy(metadata)

                # Step 0.6: 把当前 message 的 role 写入 metadata。
                # 例如 user / assistant。
                per_msg_meta["role"] = message_dict["role"]

                # Step 0.7: 如果 message 里有 name 字段，则将其作为 actor_id。
                actor_name = message_dict.get("name")
                # 注释：判断条件 `actor_name` 是否成立。
                if actor_name:
                    # 注释：计算并保存 per_msg_meta 变量，供后续逻辑使用。
                    per_msg_meta["actor_id"] = actor_name

                # Step 0.8: 取出原始 message 内容。
                msg_content = message_dict["content"]

                # Step 0.9: 对原始 message 内容做 embedding。
                # 这里的 mode 是 "add"，表示用于新增 memory。
                msg_embeddings = self.embedding_model.embed(msg_content, "add")

                # Step 0.10: 调用 _create_memory() 创建 memory。
                # _create_memory 内部会把文本、embedding、metadata 写入 vector store，
                # 并写入 SQL history。
                mem_id = self._create_memory(msg_content, {msg_content: msg_embeddings}, per_msg_meta)

                # Step 0.11: 组装返回结果。
                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )

            # Step 0.12: infer=False 的路径到这里结束，直接返回原始 messages 对应的 memories。
            return returned_memories

        # === V3 PHASED BATCH PIPELINE ===

        # Step 1: infer=True 时，进入 V3 分阶段批处理 pipeline。
        # 这个路径会使用 LLM 从对话里抽取 memory，而不是直接存原文。

        # Phase 0: Context gathering
        # Step 1.1: 根据 filters 构造 session_scope。
        # session_scope 用于 SQL DB 中读取/保存该 user_id / agent_id / run_id 对应的最近消息。
        session_scope = _build_session_scope(filters)

        # Step 1.2: 从 SQL DB 中取最近 10 条消息。
        # 这些历史消息会作为 LLM 抽取 memory 时的上下文。
        last_messages = self.db.get_last_messages(session_scope, limit=10)

        # Step 1.3: 把 messages 转成适合 prompt 使用的文本格式。
        parsed_messages = parse_messages(messages)

        # Phase 1: Existing memory retrieval
        # Step 2.1: 从 filters 中提取 session 级别的过滤条件。
        # 这里只保留 user_id / agent_id / run_id，保证只检索当前作用域下的旧 memory。
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}

        # Step 2.2: 对当前新消息 parsed_messages 做 embedding。
        # 注意这里 mode 是 "search"，因为这是为了检索已有 memory。
        query_embedding = self.embedding_model.embed(parsed_messages, "search")

        # Step 2.3: 在 vector store 中检索和当前新消息最相关的旧 memories。
        # 这里是 add() 内部的轻量 retrieve，只做一次语义向量搜索，top_k 固定为 10。
        existing_results = self.vector_store.search(
            query=parsed_messages,
            vectors=query_embedding,
            top_k=10,
            filters=search_filters,
        )

        # Map UUIDs to integers (anti-hallucination)
        # Step 2.4: 把真实 UUID 映射成简单整数 id。
        # 这样可以减少 LLM 在 prompt 中处理复杂 UUID 时产生幻觉的概率。
        existing_memories = []
        # 注释：初始化 uuid_mapping 变量 为空字典，用于后续按键保存数据。
        uuid_mapping = {}

        # Step 2.5: 遍历检索到的旧 memories。
        for idx, mem in enumerate(existing_results):
            # Step 2.6: 记录整数 id 到真实 memory id 的映射。
            uuid_mapping[str(idx)] = mem.id

            # Step 2.7: 只把整数 id 和 memory 文本传给后续 prompt。
            existing_memories.append({"id": str(idx), "text": mem.payload.get("data", "")})

        # Phase 2: LLM extraction (single call)
        # Step 3.1: 判断当前是否是纯 agent 作用域。
        # 如果有 agent_id 且没有 user_id，则认为是 agent-scoped。
        is_agent_scoped = bool(filters.get("agent_id")) and not filters.get("user_id")

        # Step 3.2: 设置系统提示词，默认使用 ADDITIVE_EXTRACTION_PROMPT。
        system_prompt = ADDITIVE_EXTRACTION_PROMPT

        # Step 3.3: 如果是 agent-scoped，则追加 agent 上下文提示。
        if is_agent_scoped:
            # 注释：在原有 system_prompt 变量 基础上累加/追加新的内容。
            system_prompt += AGENT_CONTEXT_SUFFIX

        # Step 3.4: 如果调用时传入了 prompt，则优先使用 prompt；
        # 否则使用实例配置里的 custom_instructions。
        custom_instr = prompt or self.custom_instructions

        # Step 3.5: 构造给 LLM 的用户提示词。
        # 里面包含：
        # - existing_memories：相关旧 memories
        # - new_messages：当前新输入
        # - last_k_messages：最近消息上下文
        # - custom_instructions：自定义抽取指令
        user_prompt = generate_additive_extraction_prompt(
            existing_memories=existing_memories,
            new_messages=parsed_messages,
            last_k_messages=last_messages,
            custom_instructions=custom_instr,
        )

        # Step 3.6: 调用 LLM 做单次 memory extraction。
        try:
            # 注释：计算并保存 模型响应，供后续逻辑使用。
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
        except Exception as e:
            # Step 3.7: 如果 LLM 调用失败，记录错误并返回空列表。
            logger.error(f"LLM extraction failed: {e}")
            # 注释：返回 `[]` 给调用方。
            return []

        # Parse response
        # Step 4.1: 解析 LLM 返回结果。
        try:
            # Step 4.2: 移除 LLM 返回中可能包裹的 markdown code block。
            response = remove_code_blocks(response)

            # Step 4.3: 如果 response 为空，则认为没有抽取到 memory。
            if not response or not response.strip():
                # 注释：初始化 extracted_memories 变量 为空列表，用于后续收集数据。
                extracted_memories = []
            # 注释：处理前面条件不成立时的默认分支。
            else:
                # 注释：进入可能抛出异常的代码块。
                try:
                    # Step 4.4: 优先直接按 JSON 解析，并取出 memory 字段。
                    extracted_memories = json.loads(response, strict=False).get("memory", [])
                # 注释：捕获 json.JSONDecodeError 异常并执行降级或错误处理。
                except json.JSONDecodeError:
                    # Step 4.5: 如果直接解析失败，则尝试从文本中提取 JSON 片段后再解析。
                    extracted_json = extract_json(response)
                    # 注释：计算并保存 extracted_memories 变量，供后续逻辑使用。
                    extracted_memories = json.loads(extracted_json, strict=False).get("memory", [])
        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
        except Exception as e:
            # Step 4.6: 如果解析过程出错，记录错误，并认为没有抽取到 memories。
            logger.error(f"Error parsing extraction response: {e}")
            # 注释：初始化 extracted_memories 变量 为空列表，用于后续收集数据。
            extracted_memories = []

        # Step 4.7: 如果 LLM 没有抽取出任何 memory，也要保存当前 messages 到 SQL DB。
        # 这样后续 add() 仍然可以把它作为 rolling message window 的上下文。
        if not extracted_memories:
            # Save messages even if nothing extracted
            self.db.save_messages(messages, session_scope)
            # 注释：返回 `[]` 给调用方。
            return []

        # Phase 3: Batch embed all extracted memory texts
        # Step 5.1: 从 LLM 抽取结果中取出所有 memory text。
        mem_texts = [m.get("text", "") for m in extracted_memories if m.get("text")]

        # Step 5.2: 批量对 memory text 做 embedding。
        try:
            # 注释：计算并保存 mem_embeddings_list 变量，供后续逻辑使用。
            mem_embeddings_list = self.embedding_model.embed_batch(mem_texts, "add")

            # Step 5.3: 建立 text -> embedding 的映射，方便后续构造 records。
            embed_map = dict(zip(mem_texts, mem_embeddings_list))
        # 注释：捕获 Exception 异常并执行降级或错误处理。
        except Exception:
            # Fallback: embed individually
            # Step 5.4: 如果批量 embedding 失败，则降级为逐条 embedding。
            embed_map = {}
            # 注释：遍历 mem_texts 中的元素，并将当前项赋给 text。
            for text in mem_texts:
                # 注释：进入可能抛出异常的代码块。
                try:
                    # 注释：计算并保存 embed_map 变量，供后续逻辑使用。
                    embed_map[text] = self.embedding_model.embed(text, "add")
                # 注释：捕获 Exception as e 异常并执行降级或错误处理。
                except Exception as e:
                    # 注释：输出警告日志。
                    logger.warning(f"Failed to embed memory text: {e}")

        # Phase 4: Per-memory CPU processing + Phase 5: Hash dedup
        # Build set of existing hashes for dedup
        # Step 6.1: 收集旧 memories 中已有的 hash，用于和新 memory 去重。
        existing_hashes = set()
        # 注释：遍历 existing_results 中的元素，并将当前项赋给 mem。
        for mem in existing_results:
            # 注释：计算并保存 h 变量，供后续逻辑使用。
            h = mem.payload.get("hash") if hasattr(mem, "payload") and mem.payload else None
            # 注释：判断条件 `h` 是否成立。
            if h:
                # 注释：调用 existing_hashes.add 执行对应操作。
                existing_hashes.add(h)

        # Step 6.2: records 用于暂存待写入 vector store 的新 memory。
        # 每条 record 格式为：(memory_id, text, embedding, payload)
        records = []  # (memory_id, text, embedding, payload)

        # Step 6.3: seen_hashes 用于当前 batch 内部去重。
        seen_hashes = set()  # dedup within the current batch

        # Step 6.4: 遍历 LLM 抽取出来的每条 memory。
        for mem in extracted_memories:
            # Step 6.5: 取出 memory text。
            text = mem.get("text")

            # Step 6.6: 如果 text 为空，或者没有成功生成 embedding，则跳过。
            if not text or text not in embed_map:
                # 注释：跳过本轮循环剩余逻辑，继续处理下一项。
                continue

            # Step 6.7: 对 memory text 计算 MD5 hash。
            mem_hash = hashlib.md5(text.encode()).hexdigest()

            # Step 6.8: 如果 hash 已存在于旧 memory 或当前 batch，则认为重复，跳过。
            if mem_hash in existing_hashes or mem_hash in seen_hashes:
                # 注释：输出调试日志。
                logger.debug(f"Skipping duplicate memory (hash match): {text[:50]}")
                # 注释：跳过本轮循环剩余逻辑，继续处理下一项。
                continue

            # Step 6.9: 记录当前 hash，避免本批次重复写入。
            seen_hashes.add(mem_hash)

            # Step 6.10: 对文本做 lemmatization，供后续 keyword/BM25 检索使用。
            text_lemmatized = lemmatize_for_bm25(text)

            # Step 6.11: 为新 memory 生成唯一 ID。
            memory_id = str(uuid.uuid4())

            # Step 6.12: 构造 memory metadata。
            mem_metadata = deepcopy(metadata)
            # 注释：计算并保存 mem_metadata 变量，供后续逻辑使用。
            mem_metadata["data"] = text
            # 注释：计算并保存 mem_metadata 变量，供后续逻辑使用。
            mem_metadata["text_lemmatized"] = text_lemmatized
            # 注释：计算并保存 mem_metadata 变量，供后续逻辑使用。
            mem_metadata["hash"] = mem_hash

            # Step 6.13: 如果外部 metadata 没有 created_at，则使用当前 UTC 时间。
            if "created_at" not in mem_metadata:
                # 注释：计算并保存 mem_metadata 变量，供后续逻辑使用。
                mem_metadata["created_at"] = datetime.now(timezone.utc).isoformat()

            # Step 6.14: 新增 memory 时，updated_at 初始等于 created_at。
            mem_metadata["updated_at"] = mem_metadata["created_at"]

            # Step 6.15: 如果 LLM 抽取结果包含 attributed_to，则写入 metadata。
            if mem.get("attributed_to"):
                # 注释：计算并保存 mem_metadata 变量，供后续逻辑使用。
                mem_metadata["attributed_to"] = mem["attributed_to"]

            # Step 6.16: 把新 memory 加入待持久化 records。
            records.append((memory_id, text, embed_map[text], mem_metadata))

        # Step 6.17: 如果去重后没有任何新 memory，也保存 messages，然后返回空列表。
        if not records:
            # 注释：把消息保存到本地历史数据库。
            self.db.save_messages(messages, session_scope)
            # 注释：返回 `[]` 给调用方。
            return []

        # Phase 6: Batch persist
        # Step 7.1: 从 records 中拆出 vectors、ids、payloads，准备批量写入 vector store。
        all_vectors = [r[2] for r in records]
        # 注释：计算并保存 all_ids 变量，供后续逻辑使用。
        all_ids = [r[0] for r in records]
        # 注释：计算并保存 all_payloads 变量，供后续逻辑使用。
        all_payloads = [r[3] for r in records]

        # Step 7.2: 批量写入 vector store。
        try:
            # 注释：将向量和载荷写入向量库。
            self.vector_store.insert(
                vectors=all_vectors,
                ids=all_ids,
                payloads=all_payloads,
            )
        # 注释：捕获 Exception 异常并执行降级或错误处理。
        except Exception:
            # Fallback: insert one by one
            # Step 7.3: 如果批量写入失败，则降级为逐条写入。
            for mid, vec, pay in zip(all_ids, all_vectors, all_payloads):
                # 注释：进入可能抛出异常的代码块。
                try:
                    # 注释：将向量和载荷写入向量库。
                    self.vector_store.insert(vectors=[vec], ids=[mid], payloads=[pay])
                # 注释：捕获 Exception as e 异常并执行降级或错误处理。
                except Exception as e:
                    # 注释：输出错误日志。
                    logger.error(f"Failed to insert memory {mid}: {e}")

        # Batch history
        # Step 7.4: 构造 SQL history 记录。
        # 这里所有事件都是 ADD，因为这是 add-only extraction pipeline。
        history_records = [
            {
                "memory_id": r[0],
                "old_memory": None,
                "new_memory": r[1],
                "event": "ADD",
                "created_at": r[3].get("created_at"),
                "is_deleted": 0,
            }
            for r in records
        ]

        # Step 7.5: 批量写入 SQL history。
        try:
            # 注释：批量写入记忆变更历史。
            self.db.batch_add_history(history_records)
        # 注释：捕获 Exception 异常并执行降级或错误处理。
        except Exception:
            # Fallback: add one by one
            # Step 7.6: 如果批量写入 history 失败，则降级为逐条写入。
            for hr in history_records:
                # 注释：进入可能抛出异常的代码块。
                try:
                    # 注释：写入单条记忆变更历史。
                    self.db.add_history(hr["memory_id"], None, hr["new_memory"], "ADD", created_at=hr.get("created_at"))
                # 注释：捕获 Exception as e 异常并执行降级或错误处理。
                except Exception as e:
                    # 注释：输出错误日志。
                    logger.error(f"Failed to add history for {hr['memory_id']}: {e}")

        # Phase 7: Batch entity linking
        # Step 8.1: 开始批量实体链接。
        # 这一步不是写 memory 本体，而是维护 entity -> memory_ids 的辅助索引。
        try:
            # Step 8.2: 取出本批次所有新 memory 文本。
            all_texts = [r[1] for r in records]

            # Step 8.3: 批量抽取实体。
            all_entities = extract_entities_batch(all_texts)

            # 7a: Global dedup — collect unique entities across all memories
            # Step 8.4: 对本批次所有实体做全局去重。
            # key 是规范化后的 entity_text，value 包含 entity_type、entity_text、关联的 memory_ids。
            global_entities = {}  # normalized_key -> (entity_type, entity_text, set of memory_ids)

            # Step 8.5: 遍历每条新 memory 及其对应的实体列表。
            for idx, (memory_id, text, embedding, payload) in enumerate(records):
                # 注释：计算并保存 entities 变量，供后续逻辑使用。
                entities = all_entities[idx] if idx < len(all_entities) else []

                # Step 8.6: 遍历当前 memory 中抽取出的实体。
                for entity_type, entity_text in entities:
                    # Step 8.7: 用小写 + 去空格后的 entity_text 作为去重 key。
                    key = entity_text.strip().lower()

                    # Step 8.8: 如果实体已经出现过，则把当前 memory_id 加入关联集合。
                    if key in global_entities:
                        # 注释：执行当前语句，推进该函数的业务流程。
                        global_entities[key][2].add(memory_id)
                    # 注释：处理前面条件不成立时的默认分支。
                    else:
                        # Step 8.9: 如果实体首次出现，则创建一条实体记录。
                        global_entities[key] = [entity_type, entity_text, {memory_id}]

            # Step 8.10: 如果本批次存在实体，则继续处理实体 embedding 和 entity store 写入。
            if global_entities:
                # Step 8.11: 固定实体顺序，方便 embedding 和 key 对齐。
                ordered_keys = list(global_entities.keys())

                # Step 8.12: 取出实体文本列表。
                entity_texts = [global_entities[k][1] for k in ordered_keys]

                # 7b: Single batch embed for all unique entities
                # Step 8.13: 对所有唯一实体批量做 embedding。
                try:
                    # 注释：计算并保存 entity_embeddings 变量，供后续逻辑使用。
                    entity_embeddings = self.embedding_model.embed_batch(entity_texts, "add")
                # 注释：捕获 Exception 异常并执行降级或错误处理。
                except Exception:
                    # Fallback: embed individually, use None for failures
                    # Step 8.14: 如果批量实体 embedding 失败，则降级为逐条 embedding。
                    # 失败的实体用 None 占位。
                    entity_embeddings = []
                    # 注释：遍历 entity_texts 中的元素，并将当前项赋给 t。
                    for t in entity_texts:
                        # 注释：进入可能抛出异常的代码块。
                        try:
                            # 注释：调用 entity_embeddings.append 执行对应操作。
                            entity_embeddings.append(self.embedding_model.embed(t, "add"))
                        # 注释：捕获 Exception 异常并执行降级或错误处理。
                        except Exception:
                            # 注释：调用 entity_embeddings.append 执行对应操作。
                            entity_embeddings.append(None)

                # Filter out entities with failed embeddings
                # Step 8.15: 过滤掉 embedding 失败的实体。
                valid = [(i, k) for i, k in enumerate(ordered_keys) if entity_embeddings[i] is not None]

                # Step 8.16: 如果存在有效实体，则继续查找 entity store 中是否已有类似实体。
                if valid:
                    # 注释：为 `valid_indices, valid_keys` 赋值，准备后续处理所需的数据。
                    valid_indices, valid_keys = zip(*valid)

                    # Step 8.17: 取出有效实体对应的向量。
                    valid_vectors = [entity_embeddings[i] for i in valid_indices]

                    # 7c: Batch search for existing entities
                    # Step 8.18: 取出有效实体文本。
                    valid_texts = [global_entities[k][1] for k in valid_keys]

                    # Step 8.19: 在 entity store 中批量搜索已有实体。
                    # top_k=1 表示每个实体只找最相似的一个候选。
                    existing_matches = self.entity_store.search_batch(
                        queries=valid_texts,
                        vectors_list=valid_vectors,
                        top_k=1,
                        filters=search_filters,
                    )

                    # 7d: Separate into inserts vs updates
                    # Step 8.20: 准备收集需要新插入的实体。
                    to_insert_vectors, to_insert_ids, to_insert_payloads = [], [], []

                    # Step 8.21: 遍历所有有效实体，判断是更新已有实体，还是插入新实体。
                    for j, key in enumerate(valid_keys):
                        # 注释：为 `entity_type, entity_text, memory_ids` 赋值，准备后续处理所需的数据。
                        entity_type, entity_text, memory_ids = global_entities[key]
                        # 注释：计算并保存 matches 变量，供后续逻辑使用。
                        matches = existing_matches[j] if j < len(existing_matches) else []

                        # Step 8.22: 如果找到高度相似的已有实体，则更新它的 linked_memory_ids。
                        if matches and matches[0].score >= 0.95:
                            # Update existing entity
                            match = matches[0]
                            # 注释：计算并保存 向量库载荷数据，供后续逻辑使用。
                            payload = match.payload or {}

                            # Step 8.23: 取出已有 linked_memory_ids，并合并当前 memory_ids。
                            linked = set(payload.get("linked_memory_ids", []))
                            # 注释：计算并保存 linked 变量，供后续逻辑使用。
                            linked |= memory_ids
                            # 注释：计算并保存 向量库载荷数据，供后续逻辑使用。
                            payload["linked_memory_ids"] = sorted(linked)

                            # Step 8.24: 更新 entity store 中已有实体的 payload。
                            try:
                                # 注释：调用 self.entity_store.update 执行对应操作。
                                self.entity_store.update(
                                    vector_id=match.id,
                                    vector=None,
                                    payload=payload,
                                )
                            # 注释：捕获 Exception as e 异常并执行降级或错误处理。
                            except Exception as e:
                                # 注释：输出调试日志。
                                logger.debug(f"Entity update failed for '{entity_text}': {e}")
                        # 注释：处理前面条件不成立时的默认分支。
                        else:
                            # New entity — collect for batch insert
                            # Step 8.25: 如果没有匹配到已有实体，则准备插入新实体。
                            to_insert_vectors.append(valid_vectors[j])
                            # 注释：调用 to_insert_ids.append 执行对应操作。
                            to_insert_ids.append(str(uuid.uuid4()))
                            # 注释：调用 to_insert_payloads.append 执行对应操作。
                            to_insert_payloads.append({
                                "data": entity_text,
                                "entity_type": entity_type,
                                "linked_memory_ids": sorted(memory_ids),
                                **search_filters,
                            })

                    # 7e: Single batch insert for all new entities
                    # Step 8.26: 如果存在新实体，则批量写入 entity store。
                    if to_insert_vectors:
                        # 注释：进入可能抛出异常的代码块。
                        try:
                            # 注释：调用 self.entity_store.insert 执行对应操作。
                            self.entity_store.insert(
                                vectors=to_insert_vectors,
                                ids=to_insert_ids,
                                payloads=to_insert_payloads,
                            )
                        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
                        except Exception as e:
                            # 注释：输出警告日志。
                            logger.warning(f"Batch entity insert failed: {e}")
        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
        except Exception as e:
            # Step 8.27: entity linking 失败不影响主 memory 写入流程。
            logger.warning(f"Batch entity linking failed: {e}")

        # Phase 8: Save messages + return
        # Step 9.1: 把当前 messages 保存到 SQL DB 的 rolling message window。
        self.db.save_messages(messages, session_scope)

        # Step 9.2: 构造最终返回结果。
        returned_memories = [
            {"id": r[0], "memory": r[1], "event": "ADD"}
            for r in records
        ]

        # Step 9.3: 处理 telemetry filters，用于埋点上报。
        keys, encoded_ids = process_telemetry_filters(filters)

        # Step 9.4: 记录 mem0.add telemetry 事件。
        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"},
        )

        # Step 9.5: 返回本次新增的 memories。
        return returned_memories

    # 注释：按 ID 读取单条记忆。
    def get(self, memory_id):
        """
        Retrieve a memory by ID.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        # 注释：记录遥测事件，便于统计调用行为。
        capture_event("mem0.get", self, {"memory_id": memory_id, "sync_type": "sync"})
        # 注释：计算并保存 记忆内容，供后续逻辑使用。
        memory = self.vector_store.get(vector_id=memory_id)
        # 注释：判断条件 `not memory` 是否成立。
        if not memory:
            # 注释：返回 `None` 给调用方。
            return None

        # 注释：计算并保存 需要提升到返回顶层的载荷字段，供后续逻辑使用。
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        # 注释：计算并保存 核心字段和已提升字段集合，供后续逻辑使用。
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        # 注释：计算并保存 result_item 变量，供后续逻辑使用。
        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload.get("data", ""),
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump()

        # 注释：遍历 promoted_payload_keys 中的元素，并将当前项赋给 key。
        for key in promoted_payload_keys:
            # 注释：判断条件 `key in memory.payload` 是否成立。
            if key in memory.payload:
                # 注释：计算并保存 result_item 变量，供后续逻辑使用。
                result_item[key] = memory.payload[key]

        # 注释：计算并保存 额外元数据，供后续逻辑使用。
        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        # 注释：判断条件 `additional_metadata` 是否成立。
        if additional_metadata:
            # 注释：计算并保存 result_item 变量，供后续逻辑使用。
            result_item["metadata"] = additional_metadata

        # 注释：返回 `result_item` 给调用方。
        return result_item

    # 注释：按照过滤条件列出记忆。
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
        _reject_top_level_entity_params(kwargs, "get_all")

        # Validate top_k
        _validate_search_params(top_k=top_k)

        # Validate and trim entity IDs in filters
        effective_filters = dict(filters) if filters else {}
        # 注释：判断条件 `"user_id" in effective_filters` 是否成立。
        if "user_id" in effective_filters:
            # 注释：计算并保存 规范化后的过滤条件，供后续逻辑使用。
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        # 注释：判断条件 `"agent_id" in effective_filters` 是否成立。
        if "agent_id" in effective_filters:
            # 注释：计算并保存 规范化后的过滤条件，供后续逻辑使用。
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        # 注释：判断条件 `"run_id" in effective_filters` 是否成立。
        if "run_id" in effective_filters:
            # 注释：计算并保存 规范化后的过滤条件，供后续逻辑使用。
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )

        # Validate filters contains at least one entity ID
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                "Example: filters={'user_id': 'u1'}"
            )

        # 注释：计算并保存 limit 变量，供后续逻辑使用。
        limit = top_k

        # 注释：为 `keys, encoded_ids` 赋值，准备后续处理所需的数据。
        keys, encoded_ids = process_telemetry_filters(effective_filters)
        # 注释：记录遥测事件，便于统计调用行为。
        capture_event(
            "mem0.get_all", self, {"limit": limit, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"}
        )

        # 注释：计算并保存 all_memories_result 变量，供后续逻辑使用。
        all_memories_result = self._get_all_from_vector_store(effective_filters, limit)

        # 注释：返回 `{"results": all_memories_result}` 给调用方。
        return {"results": all_memories_result}

    # 注释：从向量库中读取并格式化多条记忆。
    def _get_all_from_vector_store(self, filters, limit):
        # 注释：计算并保存 memories_result 变量，供后续逻辑使用。
        memories_result = self.vector_store.list(filters=filters, top_k=limit)

        # Handle different vector store return formats by inspecting first element
        if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0:
            # 注释：计算并保存 first_element 变量，供后续逻辑使用。
            first_element = memories_result[0]

            # If first element is a container, unwrap one level
            if isinstance(first_element, (list, tuple)):
                # 注释：计算并保存 actual_memories 变量，供后续逻辑使用。
                actual_memories = first_element
            # 注释：处理前面条件不成立时的默认分支。
            else:
                # First element is a memory object, structure is already flat
                actual_memories = memories_result
        # 注释：处理前面条件不成立时的默认分支。
        else:
            # 注释：计算并保存 actual_memories 变量，供后续逻辑使用。
            actual_memories = memories_result

        # 注释：计算并保存 需要提升到返回顶层的载荷字段，供后续逻辑使用。
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        # 注释：计算并保存 核心字段和已提升字段集合，供后续逻辑使用。
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        # 注释：初始化 formatted_memories 变量 为空列表，用于后续收集数据。
        formatted_memories = []
        # 注释：遍历 actual_memories 中的元素，并将当前项赋给 mem。
        for mem in actual_memories:
            # 注释：计算并保存 格式化后的记忆字典，供后续逻辑使用。
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            ).model_dump(exclude={"score"})

            # 注释：遍历 promoted_payload_keys 中的元素，并将当前项赋给 key。
            for key in promoted_payload_keys:
                # 注释：判断条件 `key in mem.payload` 是否成立。
                if key in mem.payload:
                    # 注释：计算并保存 格式化后的记忆字典，供后续逻辑使用。
                    memory_item_dict[key] = mem.payload[key]

            # 注释：计算并保存 额外元数据，供后续逻辑使用。
            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            # 注释：判断条件 `additional_metadata` 是否成立。
            if additional_metadata:
                # 注释：计算并保存 格式化后的记忆字典，供后续逻辑使用。
                memory_item_dict["metadata"] = additional_metadata

            # 注释：调用 formatted_memories.append 执行对应操作。
            formatted_memories.append(memory_item_dict)

        # 注释：返回 `formatted_memories` 给调用方。
        return formatted_memories

    # 注释：根据查询文本搜索相关记忆。
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
        _reject_top_level_entity_params(kwargs, "search")

        # Validate search parameters (before applying defaults)
        _validate_search_params(threshold=threshold, top_k=top_k)

        # Validate and trim entity IDs in filters
        effective_filters = filters.copy() if filters else {}
        # 注释：判断条件 `"user_id" in effective_filters` 是否成立。
        if "user_id" in effective_filters:
            # 注释：计算并保存 规范化后的过滤条件，供后续逻辑使用。
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        # 注释：判断条件 `"agent_id" in effective_filters` 是否成立。
        if "agent_id" in effective_filters:
            # 注释：计算并保存 规范化后的过滤条件，供后续逻辑使用。
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        # 注释：判断条件 `"run_id" in effective_filters` 是否成立。
        if "run_id" in effective_filters:
            # 注释：计算并保存 规范化后的过滤条件，供后续逻辑使用。
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )
        # 注释：判断条件 `not any(key in effective_filters for key in ("user_id", "agent_id", "run_id"))` 是否成立。
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                "Example: filters={'user_id': 'u1'}"
            )

        # 注释：计算并保存 limit 变量，供后续逻辑使用。
        limit = top_k

        # Apply enhanced metadata filtering if advanced operators are detected
        if self._has_advanced_operators(effective_filters):
            # 注释：计算并保存 处理后的过滤条件，供后续逻辑使用。
            processed_filters = self._process_metadata_filters(effective_filters)
            # Remove logical/operator keys that have been reprocessed
            for logical_key in ("AND", "OR", "NOT"):
                # 注释：调用 effective_filters.pop 执行对应操作。
                effective_filters.pop(logical_key, None)
            # 注释：遍历 list(effective_filters.keys()) 中的元素，并将当前项赋给 fk。
            for fk in list(effective_filters.keys()):
                # 注释：判断条件 `fk not in ("AND", "OR", "NOT", "user_id", "agent_id", "run_id") and isinstanc...` 是否成立。
                if fk not in ("AND", "OR", "NOT", "user_id", "agent_id", "run_id") and isinstance(effective_filters.get(fk), dict):
                    # 注释：调用 effective_filters.pop 执行对应操作。
                    effective_filters.pop(fk, None)
            # 注释：调用 effective_filters.update 执行对应操作。
            effective_filters.update(processed_filters)

        # 注释：为 `keys, encoded_ids` 赋值，准备后续处理所需的数据。
        keys, encoded_ids = process_telemetry_filters(effective_filters)
        # 注释：记录遥测事件，便于统计调用行为。
        capture_event(
            "mem0.search",
            self,
            {
                "limit": limit,
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "sync",
                "threshold": threshold,
                "advanced_filters": bool(filters and self._has_advanced_operators(filters)),
            },
        )

        # 注释：计算并保存 original_memories 变量，供后续逻辑使用。
        original_memories = self._search_vector_store(query, effective_filters, limit, threshold)

        # Apply reranking if enabled and reranker is available
        if rerank and self.reranker and original_memories:
            # 注释：进入可能抛出异常的代码块。
            try:
                # 注释：计算并保存 reranked_memories 变量，供后续逻辑使用。
                reranked_memories = self.reranker.rerank(query, original_memories, limit)
                # 注释：计算并保存 original_memories 变量，供后续逻辑使用。
                original_memories = reranked_memories
            # 注释：捕获 Exception as e 异常并执行降级或错误处理。
            except Exception as e:
                # 注释：输出警告日志。
                logger.warning(f"Reranking failed, using original results: {e}")

        # 注释：返回 `{"results": original_memories}` 给调用方。
        return {"results": original_memories}

    # 注释：处理高级元数据过滤表达式。
    def _process_metadata_filters(self, metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process enhanced metadata filters and convert them to vector store compatible format.

        Args:
            metadata_filters: Enhanced metadata filters with operators

        Returns:
            Dict of processed filters compatible with vector store
        """
        # 注释：初始化 处理后的过滤条件 为空字典，用于后续按键保存数据。
        processed_filters = {}

        # 注释：定义 process_condition 函数/方法，封装一段可复用逻辑。
        def process_condition(key: str, condition: Any) -> Dict[str, Any]:
            # 注释：判断条件 `not isinstance(condition, dict)` 是否成立。
            if not isinstance(condition, dict):
                # Simple equality: {"key": "value"}
                if condition == "*":
                    # Wildcard: match everything for this field (implementation depends on vector store)
                    return {key: "*"}
                # 注释：返回 `{key: condition}` 给调用方。
                return {key: condition}

            # 注释：初始化 result 变量 为空字典，用于后续按键保存数据。
            result = {}
            # 注释：遍历 condition.items() 中的元素，并将当前项赋给 operator, value。
            for operator, value in condition.items():
                # Map platform operators to universal format that can be translated by each vector store
                operator_map = {
                    "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte",
                    "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                    "contains": "contains", "icontains": "icontains"
                }

                # 注释：判断条件 `operator in operator_map` 是否成立。
                if operator in operator_map:
                    # 注释：调用 result.setdefault 执行对应操作。
                    result.setdefault(key, {})[operator_map[operator]] = value
                # 注释：处理前面条件不成立时的默认分支。
                else:
                    # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
            # 注释：返回 `result` 给调用方。
            return result

        # 注释：定义 merge_filters 函数/方法，封装一段可复用逻辑。
        def merge_filters(target: Dict[str, Any], source: Dict[str, Any]) -> None:
            """Merge source into target, deep-merging nested operator dicts for the same key."""
            # 注释：遍历 source.items() 中的元素，并将当前项赋给 key, value。
            for key, value in source.items():
                # 注释：判断条件 `key in target and isinstance(target[key], dict) and isinstance(value, dict)` 是否成立。
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    # 注释：执行当前语句，推进该函数的业务流程。
                    target[key].update(value)
                # 注释：处理前面条件不成立时的默认分支。
                else:
                    # 注释：计算并保存 target 变量，供后续逻辑使用。
                    target[key] = value

        # 注释：遍历 metadata_filters.items() 中的元素，并将当前项赋给 key, value。
        for key, value in metadata_filters.items():
            # 注释：判断条件 `key == "AND"` 是否成立。
            if key == "AND":
                # Logical AND: combine multiple conditions
                if not isinstance(value, list):
                    # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
                    raise ValueError("AND operator requires a list of conditions")
                # 注释：遍历 value 中的元素，并将当前项赋给 condition。
                for condition in value:
                    # 注释：遍历 condition.items() 中的元素，并将当前项赋给 sub_key, sub_value。
                    for sub_key, sub_value in condition.items():
                        # 注释：调用 merge_filters 执行对应操作。
                        merge_filters(processed_filters, process_condition(sub_key, sub_value))
            # 注释：当前一个条件不成立时，继续判断 `key == "OR"`。
            elif key == "OR":
                # Logical OR: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
                    raise ValueError("OR operator requires a non-empty list of conditions")
                # Store OR conditions in a way that vector stores can interpret
                processed_filters["$or"] = []
                # 注释：遍历 value 中的元素，并将当前项赋给 condition。
                for condition in value:
                    # 注释：初始化 or_condition 变量 为空字典，用于后续按键保存数据。
                    or_condition = {}
                    # 注释：遍历 condition.items() 中的元素，并将当前项赋给 sub_key, sub_value。
                    for sub_key, sub_value in condition.items():
                        # 注释：调用 merge_filters 执行对应操作。
                        merge_filters(or_condition, process_condition(sub_key, sub_value))
                    # 注释：执行当前语句，推进该函数的业务流程。
                    processed_filters["$or"].append(or_condition)
            # 注释：当前一个条件不成立时，继续判断 `key == "NOT"`。
            elif key == "NOT":
                # Logical NOT: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
                    raise ValueError("NOT operator requires a non-empty list of conditions")
                # 注释：初始化 处理后的过滤条件 为空列表，用于后续收集数据。
                processed_filters["$not"] = []
                # 注释：遍历 value 中的元素，并将当前项赋给 condition。
                for condition in value:
                    # 注释：初始化 not_condition 变量 为空字典，用于后续按键保存数据。
                    not_condition = {}
                    # 注释：遍历 condition.items() 中的元素，并将当前项赋给 sub_key, sub_value。
                    for sub_key, sub_value in condition.items():
                        # 注释：调用 merge_filters 执行对应操作。
                        merge_filters(not_condition, process_condition(sub_key, sub_value))
                    # 注释：执行当前语句，推进该函数的业务流程。
                    processed_filters["$not"].append(not_condition)
            # 注释：处理前面条件不成立时的默认分支。
            else:
                # 注释：调用 merge_filters 执行对应操作。
                merge_filters(processed_filters, process_condition(key, value))

        # 注释：返回 `processed_filters` 给调用方。
        return processed_filters

    # 注释：判断过滤条件中是否包含高级操作符。
    def _has_advanced_operators(self, filters: Dict[str, Any]) -> bool:
        """
        Check if filters contain advanced operators that need special processing.
        
        Args:
            filters: Dictionary of filters to check
            
        Returns:
            bool: True if advanced operators are detected
        """
        # 注释：判断条件 `not isinstance(filters, dict)` 是否成立。
        if not isinstance(filters, dict):
            # 注释：返回 `False` 给调用方。
            return False
            
        # 注释：遍历 filters.items() 中的元素，并将当前项赋给 key, value。
        for key, value in filters.items():
            # Check for platform-style logical operators
            if key in ["AND", "OR", "NOT"]:
                # 注释：返回 `True` 给调用方。
                return True
            # Check for comparison operators (without $ prefix for universal compatibility)
            if isinstance(value, dict):
                # 注释：遍历 value.keys() 中的元素，并将当前项赋给 op。
                for op in value.keys():
                    # 注释：判断条件 `op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "iconta...` 是否成立。
                    if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                        # 注释：返回 `True` 给调用方。
                        return True
            # Check for wildcard values
            if value == "*":
                # 注释：返回 `True` 给调用方。
                return True
        # 注释：返回 `False` 给调用方。
        return False

    # 注释：执行向量检索、关键词检索和综合排序。
    def _search_vector_store(self, query, filters, limit, threshold=0.1):
        # Guard against None threshold (backward compat)
        if threshold is None:
            # 注释：计算并保存 相似度阈值，供后续逻辑使用。
            threshold = 0.1

        # Step 1: Preprocess query
        query_lemmatized = lemmatize_for_bm25(query)
        # 注释：计算并保存 query_entities 变量，供后续逻辑使用。
        query_entities = extract_entities(query)

        # Step 2: Embed query
        embeddings = self.embedding_model.embed(query, "search")

        # Step 3: Semantic search (over-fetch for scoring pool)
        internal_limit = max(limit * 4, 60)
        # 注释：计算并保存 semantic_results 变量，供后续逻辑使用。
        semantic_results = self.vector_store.search(
            query=query, vectors=embeddings, top_k=internal_limit, filters=filters
        )

        # Step 4: Keyword search (if store supports it)
        keyword_results = self.vector_store.keyword_search(
            query=query_lemmatized, top_k=internal_limit, filters=filters
        )

        # Step 5: Compute BM25 scores from keyword results
        bm25_scores = {}
        # 注释：判断条件 `keyword_results is not None` 是否成立。
        if keyword_results is not None:
            # 注释：为 `midpoint, steepness` 赋值，准备后续处理所需的数据。
            midpoint, steepness = get_bm25_params(query, lemmatized=query_lemmatized)
            # 注释：遍历 keyword_results 中的元素，并将当前项赋给 mem。
            for mem in keyword_results:
                # 注释：计算并保存 mem_id 变量，供后续逻辑使用。
                mem_id = str(mem.id) if hasattr(mem, 'id') else str(mem.get('id', ''))
                # 注释：计算并保存 raw_score 变量，供后续逻辑使用。
                raw_score = mem.score if hasattr(mem, 'score') else mem.get('score', 0)
                # 注释：判断条件 `raw_score and raw_score > 0` 是否成立。
                if raw_score and raw_score > 0:
                    # 注释：计算并保存 bm25_scores 变量，供后续逻辑使用。
                    bm25_scores[mem_id] = normalize_bm25(raw_score, midpoint, steepness)

        # Step 6: Compute entity boosts
        entity_boosts = {}
        # 注释：判断条件 `query_entities` 是否成立。
        if query_entities:
            # 注释：计算并保存 entity_boosts 变量，供后续逻辑使用。
            entity_boosts = self._compute_entity_boosts(query_entities, filters)

        # Step 7: Build candidate set from semantic results
        candidates = []
        # 注释：遍历 semantic_results 中的元素，并将当前项赋给 mem。
        for mem in semantic_results:
            # 注释：计算并保存 mem_id 变量，供后续逻辑使用。
            mem_id = str(mem.id)
            # 注释：调用 candidates.append 执行对应操作。
            candidates.append({
                "id": mem_id,
                "score": mem.score,
                "payload": mem.payload if hasattr(mem, 'payload') else {},
            })

        # Step 8: Score and rank
        scored_results = score_and_rank(
            semantic_results=candidates,
            bm25_scores=bm25_scores,
            entity_boosts=entity_boosts,
            threshold=threshold,
            top_k=limit,
        )

        # Step 9: Format results
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        # 注释：计算并保存 核心字段和已提升字段集合，供后续逻辑使用。
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        # 注释：初始化 original_memories 变量 为空列表，用于后续收集数据。
        original_memories = []
        # 注释：遍历 scored_results 中的元素，并将当前项赋给 scored。
        for scored in scored_results:
            # 注释：计算并保存 向量库载荷数据，供后续逻辑使用。
            payload = scored.get("payload") or {}

            # 注释：判断条件 `not payload.get("data")` 是否成立。
            if not payload.get("data"):
                # 注释：跳过本轮循环剩余逻辑，继续处理下一项。
                continue  # Skip candidates with no payload data

            # 注释：计算并保存 格式化后的记忆字典，供后续逻辑使用。
            memory_item_dict = MemoryItem(
                id=scored["id"],
                memory=payload.get("data", ""),
                hash=payload.get("hash"),
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at"),
                score=scored["score"],
            ).model_dump()

            # 注释：遍历 promoted_payload_keys 中的元素，并将当前项赋给 key。
            for key in promoted_payload_keys:
                # 注释：判断条件 `key in payload` 是否成立。
                if key in payload:
                    # 注释：计算并保存 格式化后的记忆字典，供后续逻辑使用。
                    memory_item_dict[key] = payload[key]

            # 注释：计算并保存 额外元数据，供后续逻辑使用。
            additional_metadata = {k: v for k, v in payload.items() if k not in core_and_promoted_keys}
            # 注释：判断条件 `additional_metadata` 是否成立。
            if additional_metadata:
                # 注释：判断条件 `not memory_item_dict.get("metadata")` 是否成立。
                if not memory_item_dict.get("metadata"):
                    # 注释：初始化 格式化后的记忆字典 为空字典，用于后续按键保存数据。
                    memory_item_dict["metadata"] = {}
                # 注释：执行当前语句，推进该函数的业务流程。
                memory_item_dict["metadata"].update(additional_metadata)

            # 注释：调用 original_memories.append 执行对应操作。
            original_memories.append(memory_item_dict)

        # 注释：返回 `original_memories` 给调用方。
        return original_memories

    # 注释：根据实体匹配结果计算记忆加权分数。
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
        seen = set()
        # 注释：初始化 deduped 变量 为空列表，用于后续收集数据。
        deduped = []
        # 注释：遍历 query_entities[ 中的元素，并将当前项赋给 entity_type, entity_text。
        for entity_type, entity_text in query_entities[:8]:
            # 注释：计算并保存 key 变量，供后续逻辑使用。
            key = entity_text.strip().lower()
            # 注释：判断条件 `key and key not in seen` 是否成立。
            if key and key not in seen:
                # 注释：调用 seen.add 执行对应操作。
                seen.add(key)
                # 注释：调用 deduped.append 执行对应操作。
                deduped.append((entity_type, entity_text))

        # 注释：判断条件 `not deduped` 是否成立。
        if not deduped:
            # 注释：返回 `{}` 给调用方。
            return {}

        # 注释：计算并保存 检索过滤条件，供后续逻辑使用。
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        # 注释：初始化 memory_boosts 变量 为空字典，用于后续按键保存数据。
        memory_boosts = {}

        # 注释：进入可能抛出异常的代码块。
        try:
            # 注释：遍历 deduped 中的元素，并将当前项赋给 _, entity_text。
            for _, entity_text in deduped:
                # 注释：计算并保存 实体向量，供后续逻辑使用。
                entity_embedding = self.embedding_model.embed(entity_text, "search")
                # 注释：计算并保存 matches 变量，供后续逻辑使用。
                matches = self.entity_store.search(
                    query=entity_text,
                    vectors=entity_embedding,
                    top_k=500,
                    filters=search_filters,
                )

                # 注释：遍历 matches 中的元素，并将当前项赋给 match。
                for match in matches:
                    # 注释：计算并保存 similarity 变量，供后续逻辑使用。
                    similarity = match.score if hasattr(match, 'score') else 0.0
                    # 注释：判断条件 `similarity < 0.5` 是否成立。
                    if similarity < 0.5:
                        # 注释：跳过本轮循环剩余逻辑，继续处理下一项。
                        continue

                    # 注释：计算并保存 向量库载荷数据，供后续逻辑使用。
                    payload = match.payload if hasattr(match, 'payload') else {}
                    # 注释：计算并保存 关联记忆 ID 列表，供后续逻辑使用。
                    linked_memory_ids = payload.get("linked_memory_ids", [])
                    # 注释：判断条件 `not isinstance(linked_memory_ids, list)` 是否成立。
                    if not isinstance(linked_memory_ids, list):
                        # 注释：跳过本轮循环剩余逻辑，继续处理下一项。
                        continue

                    # Spread-attenuated boost: entities linking to many memories get attenuated
                    num_linked = max(len(linked_memory_ids), 1)
                    # 注释：计算并保存 memory_count_weight 变量，供后续逻辑使用。
                    memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
                    # 注释：计算并保存 boost 变量，供后续逻辑使用。
                    boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight

                    # 注释：遍历 linked_memory_ids 中的元素，并将当前项赋给 memory_id。
                    for memory_id in linked_memory_ids:
                        # 注释：判断条件 `memory_id` 是否成立。
                        if memory_id:
                            # 注释：计算并保存 memory_key 变量，供后续逻辑使用。
                            memory_key = str(memory_id)
                            # 注释：计算并保存 memory_boosts 变量，供后续逻辑使用。
                            memory_boosts[memory_key] = max(memory_boosts.get(memory_key, 0.0), boost)

        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
        except Exception as e:
            # 注释：输出警告日志。
            logger.warning(f"Entity boost computation failed: {e}")

        # 注释：返回 `memory_boosts` 给调用方。
        return memory_boosts

    # 注释：更新指定 ID 的记忆内容。
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
        # 注释：记录遥测事件，便于统计调用行为。
        capture_event("mem0.update", self, {"memory_id": memory_id, "sync_type": "sync"})

        # 注释：计算并保存 existing_embeddings 变量，供后续逻辑使用。
        existing_embeddings = {data: self.embedding_model.embed(data, "update")}

        # 注释：调用 self._update_memory 执行对应操作。
        self._update_memory(memory_id, data, existing_embeddings, metadata)
        # 注释：返回 `{"message": "Memory updated successfully!"}` 给调用方。
        return {"message": "Memory updated successfully!"}

    # 注释：删除指定 ID 的记忆。
    def delete(self, memory_id):
        """
        Delete a memory by ID.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        # 注释：记录遥测事件，便于统计调用行为。
        capture_event("mem0.delete", self, {"memory_id": memory_id, "sync_type": "sync"})

        # 注释：计算并保存 existing_memory 变量，供后续逻辑使用。
        existing_memory = self.vector_store.get(vector_id=memory_id)
        # 注释：判断条件 `existing_memory is None` 是否成立。
        if existing_memory is None:
            # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
            raise ValueError(f"Memory with id {memory_id} not found")

        # 注释：调用 self._delete_memory 执行对应操作。
        self._delete_memory(memory_id, existing_memory)
        # 注释：返回 `{"message": "Memory deleted successfully!"}` 给调用方。
        return {"message": "Memory deleted successfully!"}

    # 注释：按用户、代理或运行 ID 批量删除记忆。
    def delete_all(self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None):
        """
        Delete all memories.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        # 注释：为 `filters: Dict[str, Any]` 赋值，准备后续处理所需的数据。
        filters: Dict[str, Any] = {}
        # 注释：判断条件 `user_id` 是否成立。
        if user_id:
            # 注释：计算并保存 过滤条件，供后续逻辑使用。
            filters["user_id"] = user_id
        # 注释：判断条件 `agent_id` 是否成立。
        if agent_id:
            # 注释：计算并保存 过滤条件，供后续逻辑使用。
            filters["agent_id"] = agent_id
        # 注释：判断条件 `run_id` 是否成立。
        if run_id:
            # 注释：计算并保存 过滤条件，供后续逻辑使用。
            filters["run_id"] = run_id

        # 注释：判断条件 `not filters` 是否成立。
        if not filters:
            # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        # 注释：为 `keys, encoded_ids` 赋值，准备后续处理所需的数据。
        keys, encoded_ids = process_telemetry_filters(filters)
        # 注释：记录遥测事件，便于统计调用行为。
        capture_event("mem0.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"})
        # delete all vector memories and reset the collections
        memories = self.vector_store.list(filters=filters)[0]
        # 注释：遍历 memories 中的元素，并将当前项赋给 memory。
        for memory in memories:
            # 注释：调用 self._delete_memory 执行对应操作。
            self._delete_memory(memory.id)

        # 注释：输出信息日志。
        logger.info(f"Deleted {len(memories)} memories")

        # 注释：返回 `{"message": "Memories deleted successfully!"}` 给调用方。
        return {"message": "Memories deleted successfully!"}

    # 注释：读取指定记忆的变更历史。
    def history(self, memory_id):
        """
        Get the history of changes for a memory by ID.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        # 注释：记录遥测事件，便于统计调用行为。
        capture_event("mem0.history", self, {"memory_id": memory_id, "sync_type": "sync"})
        # 注释：返回 `self.db.get_history(memory_id)` 给调用方。
        return self.db.get_history(memory_id)

    # 注释：创建一条新记忆并写入向量库和历史表。
    def _create_memory(self, data, existing_embeddings, metadata=None):
        # 注释：输出调试日志。
        logger.debug(f"Creating memory with {data=}")
        # 注释：判断条件 `data in existing_embeddings` 是否成立。
        if data in existing_embeddings:
            # 注释：计算并保存 向量表示，供后续逻辑使用。
            embeddings = existing_embeddings[data]
        # 注释：处理前面条件不成立时的默认分支。
        else:
            # 注释：计算并保存 向量表示，供后续逻辑使用。
            embeddings = self.embedding_model.embed(data, memory_action="add")
        # 注释：计算并保存 记忆 ID，供后续逻辑使用。
        memory_id = str(uuid.uuid4())
        # 注释：深拷贝生成 new_metadata 变量，避免修改原始输入对象。
        new_metadata = deepcopy(metadata) if metadata is not None else {}
        # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
        new_metadata["data"] = data
        # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        # 注释：判断条件 `"created_at" not in new_metadata` 是否成立。
        if "created_at" not in new_metadata:
            # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
            new_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
        new_metadata["updated_at"] = new_metadata["created_at"]
        # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data)

        # 注释：将向量和载荷写入向量库。
        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[new_metadata],
        )
        # 注释：写入单条记忆变更历史。
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
        # 注释：返回 `memory_id` 给调用方。
        return memory_id

    # 注释：创建程序性记忆。
    def _create_procedural_memory(self, messages, metadata=None, prompt=None):
        """
        Create a procedural memory

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        # 注释：输出信息日志。
        logger.info("Creating procedural memory")

        # 注释：计算并保存 parsed_messages 变量，供后续逻辑使用。
        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {
                "role": "user",
                "content": "Create procedural memory of the above conversation.",
            },
        ]

        # 注释：进入可能抛出异常的代码块。
        try:
            # 注释：计算并保存 procedural_memory 变量，供后续逻辑使用。
            procedural_memory = self.llm.generate_response(messages=parsed_messages)
            # 注释：计算并保存 procedural_memory 变量，供后续逻辑使用。
            procedural_memory = remove_code_blocks(procedural_memory)
        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
        except Exception as e:
            # 注释：输出错误日志。
            logger.error(f"Error generating procedural memory summary: {e}")
            # 注释：执行当前语句，推进该函数的业务流程。
            raise

        # 注释：判断条件 `metadata is None` 是否成立。
        if metadata is None:
            # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
            raise ValueError("Metadata cannot be done for procedural memory.")

        # 注释：计算并保存 元数据，供后续逻辑使用。
        metadata = {**metadata, "memory_type": MemoryType.PROCEDURAL.value}
        # 注释：计算并保存 向量表示，供后续逻辑使用。
        embeddings = self.embedding_model.embed(procedural_memory, memory_action="add")
        # 注释：计算并保存 记忆 ID，供后续逻辑使用。
        memory_id = self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        # 注释：记录遥测事件，便于统计调用行为。
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "sync"})

        # 注释：计算并保存 result 变量，供后续逻辑使用。
        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        # 注释：返回 `result` 给调用方。
        return result

    # 注释：更新记忆的向量、载荷和历史记录。
    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        # 注释：输出信息日志。
        logger.info(f"Updating memory with {data=}")

        # 注释：进入可能抛出异常的代码块。
        try:
            # 注释：计算并保存 existing_memory 变量，供后续逻辑使用。
            existing_memory = self.vector_store.get(vector_id=memory_id)
        # 注释：捕获 Exception 异常并执行降级或错误处理。
        except Exception:
            # 注释：输出错误日志。
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        # 注释：判断条件 `existing_memory is None` 是否成立。
        if existing_memory is None:
            # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
            raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")

        # 注释：计算并保存 prev_value 变量，供后续逻辑使用。
        prev_value = existing_memory.payload.get("data")

        # 注释：深拷贝生成 new_metadata 变量，避免修改原始输入对象。
        new_metadata = deepcopy(metadata) if metadata is not None else {}

        # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
        new_metadata["data"] = data
        # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data)
        # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
        new_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Preserve session identifiers from existing memory only if not provided in new metadata
        if "user_id" not in new_metadata and "user_id" in existing_memory.payload:
            # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        # 注释：判断条件 `"agent_id" not in new_metadata and "agent_id" in existing_memory.payload` 是否成立。
        if "agent_id" not in new_metadata and "agent_id" in existing_memory.payload:
            # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        # 注释：判断条件 `"run_id" not in new_metadata and "run_id" in existing_memory.payload` 是否成立。
        if "run_id" not in new_metadata and "run_id" in existing_memory.payload:
            # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
            new_metadata["run_id"] = existing_memory.payload["run_id"]
        # 注释：判断条件 `"actor_id" in existing_memory.payload` 是否成立。
        if "actor_id" in existing_memory.payload:
            # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        # 注释：判断条件 `"role" not in new_metadata and "role" in existing_memory.payload` 是否成立。
        if "role" not in new_metadata and "role" in existing_memory.payload:
            # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
            new_metadata["role"] = existing_memory.payload["role"]

        # 注释：判断条件 `data in existing_embeddings` 是否成立。
        if data in existing_embeddings:
            # 注释：计算并保存 向量表示，供后续逻辑使用。
            embeddings = existing_embeddings[data]
        # 注释：处理前面条件不成立时的默认分支。
        else:
            # 注释：计算并保存 向量表示，供后续逻辑使用。
            embeddings = self.embedding_model.embed(data, "update")

        # 注释：更新向量库中的指定记录。
        self.vector_store.update(
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        # 注释：输出信息日志。
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        # 注释：写入单条记忆变更历史。
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
        session_filters = {k: new_metadata[k] for k in ("user_id", "agent_id", "run_id") if new_metadata.get(k)}
        # 注释：调用 self._remove_memory_from_entity_store 执行对应操作。
        self._remove_memory_from_entity_store(memory_id, session_filters)
        # 注释：调用 self._link_entities_for_memory 执行对应操作。
        self._link_entities_for_memory(memory_id, data, session_filters)

        # 注释：返回 `memory_id` 给调用方。
        return memory_id

    # 注释：删除记忆并清理相关历史或实体索引。
    def _delete_memory(self, memory_id, existing_memory=None):
        # 注释：输出信息日志。
        logger.info(f"Deleting memory with {memory_id=}")
        # 注释：判断条件 `existing_memory is None` 是否成立。
        if existing_memory is None:
            # 注释：计算并保存 existing_memory 变量，供后续逻辑使用。
            existing_memory = self.vector_store.get(vector_id=memory_id)
            # 注释：判断条件 `existing_memory is None` 是否成立。
            if existing_memory is None:
                # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
                raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")
        # 注释：计算并保存 prev_value 变量，供后续逻辑使用。
        prev_value = existing_memory.payload.get("data", "")
        # 注释：计算并保存 created_at 变量，供后续逻辑使用。
        created_at = _normalize_iso_timestamp_to_utc(existing_memory.payload.get("created_at"))
        # 注释：计算并保存 updated_at 变量，供后续逻辑使用。
        updated_at = datetime.now(timezone.utc).isoformat()
        # 注释：计算并保存 向量库载荷数据，供后续逻辑使用。
        payload = existing_memory.payload or {}
        # 注释：计算并保存 session_filters 变量，供后续逻辑使用。
        session_filters = {k: payload[k] for k in ("user_id", "agent_id", "run_id") if payload.get(k)}
        # 注释：删除向量库中的指定记录。
        self.vector_store.delete(vector_id=memory_id)
        # 注释：写入单条记忆变更历史。
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
        self._remove_memory_from_entity_store(memory_id, session_filters)

        # 注释：返回 `memory_id` 给调用方。
        return memory_id

    # 注释：重置底层存储中的所有记忆数据。
    def reset(self):
        """
        Reset the memory store by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        # 注释：输出警告日志。
        logger.warning("Resetting all memories")

        # 注释：判断条件 `hasattr(self.db, "connection") and self.db.connection` 是否成立。
        if hasattr(self.db, "connection") and self.db.connection:
            # 注释：调用 self.db.connection.execute 执行对应操作。
            self.db.connection.execute("DROP TABLE IF EXISTS history")
            # 注释：调用 self.db.connection.close 执行对应操作。
            self.db.connection.close()

        # 注释：设置当前实例的 db 属性，用于后续方法共享状态。
        self.db = SQLiteManager(self.config.history_db_path)

        # 注释：判断条件 `hasattr(self.vector_store, "reset")` 是否成立。
        if hasattr(self.vector_store, "reset"):
            # 注释：设置当前实例的 vector_store 属性，用于后续方法共享状态。
            self.vector_store = VectorStoreFactory.reset(self.vector_store)
        # 注释：处理前面条件不成立时的默认分支。
        else:
            # 注释：输出警告日志。
            logger.warning("Vector store does not support reset. Skipping.")
            # 注释：调用 self.vector_store.delete_col 执行对应操作。
            self.vector_store.delete_col()
            # 注释：设置当前实例的 vector_store 属性，用于后续方法共享状态。
            self.vector_store = VectorStoreFactory.create(
                self.config.vector_store.provider, self.config.vector_store.config
            )
        # Reset entity store if initialized
        if self._entity_store is not None:
            # 注释：进入可能抛出异常的代码块。
            try:
                # 注释：调用 self._entity_store.reset 执行对应操作。
                self._entity_store.reset()
            # 注释：捕获 Exception as e 异常并执行降级或错误处理。
            except Exception as e:
                # 注释：输出警告日志。
                logger.warning(f"Failed to reset entity store: {e}")
            # 注释：设置当前实例的 _entity_store 属性，用于后续方法共享状态。
            self._entity_store = None

        # 注释：记录遥测事件，便于统计调用行为。
        capture_event("mem0.reset", self, {"sync_type": "sync"})

    # 注释：定义 close 函数/方法，封装一段可复用逻辑。
    def close(self):
        """Release resources held by this Memory instance (SQLite connections, etc.)."""
        # 注释：判断条件 `hasattr(self, "db") and self.db is not None` 是否成立。
        if hasattr(self, "db") and self.db is not None:
            # 注释：调用 self.db.close 执行对应操作。
            self.db.close()
            # 注释：设置当前实例的 db 属性，用于后续方法共享状态。
            self.db = None

    # 注释：预留聊天接口。
    def chat(self, query):
        # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
        raise NotImplementedError("Chat function not implemented yet.")


# 注释：定义 AsyncMemory 类，并继承/使用 (MemoryBase) 中的基础能力。
class AsyncMemory(MemoryBase):
    # 注释：初始化 Memory 实例并创建模型、向量库、数据库等依赖。
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        # 注释：设置当前实例的 config 属性，用于后续方法共享状态。
        self.config = config

        # 注释：设置当前实例的 embedding_model 属性，用于后续方法共享状态。
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        # 注释：设置当前实例的 vector_store 属性，用于后续方法共享状态。
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        # 注释：设置当前实例的 llm 属性，用于后续方法共享状态。
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        # 注释：设置当前实例的 db 属性，用于后续方法共享状态。
        self.db = SQLiteManager(self.config.history_db_path)
        # 注释：设置当前实例的 collection_name 属性，用于后续方法共享状态。
        self.collection_name = self.config.vector_store.config.collection_name
        # 注释：设置当前实例的 api_version 属性，用于后续方法共享状态。
        self.api_version = self.config.version
        # 注释：设置当前实例的 custom_instructions 属性，用于后续方法共享状态。
        self.custom_instructions = self.config.custom_instructions
        # 注释：设置当前实例的 _entity_store 属性，用于后续方法共享状态。
        self._entity_store = None

        # Initialize reranker if configured
        self.reranker = None
        # 注释：判断条件 `config.reranker` 是否成立。
        if config.reranker:
            # 注释：设置当前实例的 reranker 属性，用于后续方法共享状态。
            self.reranker = RerankerFactory.create(
                config.reranker.provider,
                config.reranker.config
            )

        # 注释：判断条件 `MEM0_TELEMETRY` 是否成立。
        if MEM0_TELEMETRY:
            # 注释：深拷贝生成 telemetry_config 变量，避免修改原始输入对象。
            telemetry_config = _safe_deepcopy_config(self.config.vector_store.config)
            # 注释：计算并保存 collection_name 变量，供后续逻辑使用。
            telemetry_config.collection_name = "mem0migrations"
            # 注释：判断条件 `self.config.vector_store.provider in ["faiss", "qdrant"]` 是否成立。
            if self.config.vector_store.provider in ["faiss", "qdrant"]:
                # 注释：计算并保存 provider_path 变量，供后续逻辑使用。
                provider_path = f"migrations_{self.config.vector_store.provider}"
                # 注释：计算并保存 path 变量，供后续逻辑使用。
                telemetry_config.path = os.path.join(mem0_dir, provider_path)
                # 注释：确保目标目录存在。
                os.makedirs(telemetry_config.path, exist_ok=True)
            # 注释：设置当前实例的 _telemetry_vector_store 属性，用于后续方法共享状态。
            self._telemetry_vector_store = VectorStoreFactory.create(self.config.vector_store.provider, telemetry_config)

        # 注释：记录遥测事件，便于统计调用行为。
        capture_event("mem0.init", self, {"sync_type": "async"})

    # 注释：应用 property 装饰器，调整下面定义的函数或属性行为。
    @property
    # 注释：定义 entity_store 函数/方法，封装一段可复用逻辑。
    def entity_store(self):
        """Lazily initialize entity store on first use."""
        # 注释：判断条件 `self._entity_store is None` 是否成立。
        if self._entity_store is None:
            # 注释：深拷贝生成 entity_config 变量，避免修改原始输入对象。
            entity_config = _safe_deepcopy_config(self.config.vector_store.config)
            # 注释：计算并保存 entity_collection 变量，供后续逻辑使用。
            entity_collection = f"{self.collection_name}_entities"
            # 注释：判断条件 `hasattr(entity_config, 'collection_name')` 是否成立。
            if hasattr(entity_config, 'collection_name'):
                # 注释：计算并保存 collection_name 变量，供后续逻辑使用。
                entity_config.collection_name = entity_collection
            # 注释：当前一个条件不成立时，继续判断 `isinstance(entity_config, dict)`。
            elif isinstance(entity_config, dict):
                # 注释：计算并保存 entity_config 变量，供后续逻辑使用。
                entity_config['collection_name'] = entity_collection
            # For Qdrant, share the existing client to avoid RocksDB lock contention
            # when using embedded mode (path=...). QdrantConfig.client takes precedence
            # over host/port/path.
            if self.config.vector_store.provider == "qdrant" and hasattr(self.vector_store, "client"):
                # 注释：判断条件 `hasattr(entity_config, "client")` 是否成立。
                if hasattr(entity_config, "client"):
                    # 注释：计算并保存 client 变量，供后续逻辑使用。
                    entity_config.client = self.vector_store.client
                # 注释：当前一个条件不成立时，继续判断 `isinstance(entity_config, dict)`。
                elif isinstance(entity_config, dict):
                    # 注释：计算并保存 entity_config 变量，供后续逻辑使用。
                    entity_config["client"] = self.vector_store.client
            # 注释：设置当前实例的 _entity_store 属性，用于后续方法共享状态。
            self._entity_store = VectorStoreFactory.create(
                self.config.vector_store.provider, entity_config
            )
        # 注释：返回 `self._entity_store` 给调用方。
        return self._entity_store

    # 注释：定义 _upsert_entity_async 函数/方法，封装一段可复用逻辑。
    async def _upsert_entity_async(self, entity_text, entity_type, memory_id, filters):
        """Async variant of `_upsert_entity` — per-entity search-then-update-or-insert."""
        # 注释：进入可能抛出异常的代码块。
        try:
            # 注释：计算并保存 实体向量，供后续逻辑使用。
            entity_embedding = await asyncio.to_thread(self.embedding_model.embed, entity_text, "add")
            # 注释：计算并保存 检索过滤条件，供后续逻辑使用。
            search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}

            # 注释：计算并保存 existing 变量，供后续逻辑使用。
            existing = await asyncio.to_thread(
                self.entity_store.search,
                query=entity_text,
                vectors=entity_embedding,
                top_k=1,
                filters=search_filters,
            )

            # 注释：判断条件 `existing and existing[0].score >= 0.95` 是否成立。
            if existing and existing[0].score >= 0.95:
                # 注释：计算并保存 match 变量，供后续逻辑使用。
                match = existing[0]
                # 注释：计算并保存 向量库载荷数据，供后续逻辑使用。
                payload = match.payload or {}
                # 注释：计算并保存 linked_ids 变量，供后续逻辑使用。
                linked_ids = payload.get("linked_memory_ids", [])
                # 注释：判断条件 `memory_id not in linked_ids` 是否成立。
                if memory_id not in linked_ids:
                    # 注释：调用 linked_ids.append 执行对应操作。
                    linked_ids.append(memory_id)
                    # 注释：计算并保存 向量库载荷数据，供后续逻辑使用。
                    payload["linked_memory_ids"] = linked_ids
                    # 注释：等待异步操作 `asyncio.to_thread(` 完成。
                    await asyncio.to_thread(
                        self.entity_store.update,
                        vector_id=match.id,
                        vector=None,
                        payload=payload,
                    )
            # 注释：处理前面条件不成立时的默认分支。
            else:
                # 注释：计算并保存 entity_id 变量，供后续逻辑使用。
                entity_id = str(uuid.uuid4())
                # 注释：计算并保存 entity_payload 变量，供后续逻辑使用。
                entity_payload = {
                    "data": entity_text,
                    "entity_type": entity_type,
                    "linked_memory_ids": [memory_id],
                    **{k: v for k, v in search_filters.items()},
                }
                # 注释：等待异步操作 `asyncio.to_thread(` 完成。
                await asyncio.to_thread(
                    self.entity_store.insert,
                    vectors=[entity_embedding],
                    ids=[entity_id],
                    payloads=[entity_payload],
                )
        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
        except Exception as e:
            # 注释：输出警告日志。
            logger.warning(f"Entity upsert failed for '{entity_text}' (async): {e}")

    # 注释：从实体索引中移除某条记忆的关联。
    async def _remove_memory_from_entity_store(self, memory_id, filters):
        """Async variant of `Memory._remove_memory_from_entity_store`."""
        # 注释：判断条件 `self._entity_store is None` 是否成立。
        if self._entity_store is None:
            # 注释：结束函数并返回空值。
            return
        # 注释：计算并保存 检索过滤条件，供后续逻辑使用。
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        # 注释：进入可能抛出异常的代码块。
        try:
            # 注释：计算并保存 listed 变量，供后续逻辑使用。
            listed = await asyncio.to_thread(self.entity_store.list, filters=search_filters, top_k=10000)
            # 注释：计算并保存 rows 变量，供后续逻辑使用。
            rows = listed[0] if isinstance(listed, (list, tuple)) and listed and isinstance(listed[0], list) else listed
            # 注释：遍历 rows or [] 中的元素，并将当前项赋给 row。
            for row in rows or []:
                # 注释：进入可能抛出异常的代码块。
                try:
                    # 注释：计算并保存 向量库载荷数据，供后续逻辑使用。
                    payload = getattr(row, "payload", None) or {}
                    # 注释：计算并保存 linked 变量，供后续逻辑使用。
                    linked = payload.get("linked_memory_ids", [])
                    # 注释：判断条件 `not isinstance(linked, list) or memory_id not in linked` 是否成立。
                    if not isinstance(linked, list) or memory_id not in linked:
                        # 注释：跳过本轮循环剩余逻辑，继续处理下一项。
                        continue
                    # 注释：计算并保存 remaining 变量，供后续逻辑使用。
                    remaining = [mid for mid in linked if mid != memory_id]
                    # 注释：判断条件 `not remaining` 是否成立。
                    if not remaining:
                        # 注释：进入可能抛出异常的代码块。
                        try:
                            # 注释：等待异步操作 `asyncio.to_thread(self.entity_store.delete, vector_id=row.id)` 完成。
                            await asyncio.to_thread(self.entity_store.delete, vector_id=row.id)
                        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
                        except Exception as e:
                            # 注释：输出调试日志。
                            logger.debug(f"Entity delete failed for id={row.id} (async): {e}")
                    # 注释：处理前面条件不成立时的默认分支。
                    else:
                        # 注释：计算并保存 实体文本，供后续逻辑使用。
                        entity_text = payload.get("data")
                        # 注释：判断条件 `not isinstance(entity_text, str) or not entity_text` 是否成立。
                        if not isinstance(entity_text, str) or not entity_text:
                            # 注释：输出调试日志。
                            logger.debug(f"Entity id={row.id} missing 'data'; skipping update during cleanup (async)")
                            # 注释：跳过本轮循环剩余逻辑，继续处理下一项。
                            continue
                        # 注释：进入可能抛出异常的代码块。
                        try:
                            # 注释：计算并保存 vec 变量，供后续逻辑使用。
                            vec = await asyncio.to_thread(self.embedding_model.embed, entity_text, "update")
                        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
                        except Exception as e:
                            # 注释：输出调试日志。
                            logger.debug(f"Entity re-embed failed for '{entity_text}' (async): {e}")
                            # 注释：跳过本轮循环剩余逻辑，继续处理下一项。
                            continue
                        # 注释：计算并保存 new_payload 变量，供后续逻辑使用。
                        new_payload = {**payload, "linked_memory_ids": remaining}
                        # 注释：进入可能抛出异常的代码块。
                        try:
                            # 注释：等待异步操作 `asyncio.to_thread(` 完成。
                            await asyncio.to_thread(
                                self.entity_store.update,
                                vector_id=row.id,
                                vector=vec,
                                payload=new_payload,
                            )
                        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
                        except Exception as e:
                            # 注释：输出调试日志。
                            logger.debug(f"Entity update failed for id={row.id} (async): {e}")
                # 注释：捕获 Exception as e 异常并执行降级或错误处理。
                except Exception as e:
                    # 注释：输出调试日志。
                    logger.debug(f"Entity cleanup error (async): {e}")
        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
        except Exception as e:
            # 注释：输出警告日志。
            logger.warning(f"Entity store cleanup failed for memory_id={memory_id} (async): {e}")

    # 注释：抽取记忆文本中的实体并建立关联。
    async def _link_entities_for_memory(self, memory_id, text, filters):
        """Async variant of `Memory._link_entities_for_memory`."""
        # 注释：进入可能抛出异常的代码块。
        try:
            # 注释：计算并保存 entities 变量，供后续逻辑使用。
            entities = await asyncio.to_thread(extract_entities, text)
            # 注释：判断条件 `not entities` 是否成立。
            if not entities:
                # 注释：结束函数并返回空值。
                return
            # 注释：初始化 seen 变量 为空集合，用于后续去重。
            seen = set()
            # 注释：遍历 entities 中的元素，并将当前项赋给 entity_type, entity_text。
            for entity_type, entity_text in entities:
                # 注释：计算并保存 key 变量，供后续逻辑使用。
                key = entity_text.strip().lower()
                # 注释：判断条件 `not key or key in seen` 是否成立。
                if not key or key in seen:
                    # 注释：跳过本轮循环剩余逻辑，继续处理下一项。
                    continue
                # 注释：调用 seen.add 执行对应操作。
                seen.add(key)
                # 注释：进入可能抛出异常的代码块。
                try:
                    # 注释：等待异步操作 `self._upsert_entity_async(entity_text, entity_type, memory_id, filters)` 完成。
                    await self._upsert_entity_async(entity_text, entity_type, memory_id, filters)
                # 注释：捕获 Exception as e 异常并执行降级或错误处理。
                except Exception as e:
                    # 注释：输出调试日志。
                    logger.debug(f"Entity link failed for '{entity_text}' (async): {e}")
        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
        except Exception as e:
            # 注释：输出警告日志。
            logger.warning(f"Entity linking failed for memory_id={memory_id} (async): {e}")

    # 注释：应用 classmethod 装饰器，调整下面定义的函数或属性行为。
    @classmethod
    # 注释：根据字典配置创建 Memory 实例。
    def from_config(cls, config_dict: Dict[str, Any]):
        # 注释：进入可能抛出异常的代码块。
        try:
            # 注释：计算并保存 配置对象，供后续逻辑使用。
            config = cls._process_config(config_dict)
            # 注释：计算并保存 配置对象，供后续逻辑使用。
            config = MemoryConfig(**config_dict)
        # 注释：捕获 ValidationError as e 异常并执行降级或错误处理。
        except ValidationError as e:
            # 注释：输出错误日志。
            logger.error(f"Configuration validation error: {e}")
            # 注释：执行当前语句，推进该函数的业务流程。
            raise
        # 注释：返回 `cls(config)` 给调用方。
        return cls(config)

    # 注释：应用 staticmethod 装饰器，调整下面定义的函数或属性行为。
    @staticmethod
    # 注释：预处理配置字典。
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        # 注释：进入可能抛出异常的代码块。
        try:
            # 注释：返回 `config_dict` 给调用方。
            return config_dict
        # 注释：捕获 ValidationError as e 异常并执行降级或错误处理。
        except ValidationError as e:
            # 注释：输出错误日志。
            logger.error(f"Configuration validation error: {e}")
            # 注释：执行当前语句，推进该函数的业务流程。
            raise

    # 注释：定义 _should_use_agent_memory_extraction 函数/方法，封装一段可复用逻辑。
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

    # 注释：新增记忆入口，负责校验输入并写入记忆。
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
        # 注释：为 `processed_metadata, effective_filters` 赋值，准备后续处理所需的数据。
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_metadata=metadata
        )

        # 注释：判断条件 `memory_type is not None and memory_type != MemoryType.PROCEDURAL.value` 是否成立。
        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
            raise ValueError(
                f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories."
            )

        # 注释：判断条件 `isinstance(messages, str)` 是否成立。
        if isinstance(messages, str):
            # 注释：计算并保存 消息列表，供后续逻辑使用。
            messages = [{"role": "user", "content": messages}]

        # 注释：当前一个条件不成立时，继续判断 `isinstance(messages, dict)`。
        elif isinstance(messages, dict):
            # 注释：计算并保存 消息列表，供后续逻辑使用。
            messages = [messages]

        # 注释：当前一个条件不成立时，继续判断 `not isinstance(messages, list)`。
        elif not isinstance(messages, list):
            # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
            raise Mem0ValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries."
            )

        # 注释：判断条件 `agent_id is not None and memory_type == MemoryType.PROCEDURAL.value` 是否成立。
        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            # 注释：计算并保存 结果集合，供后续逻辑使用。
            results = await self._create_procedural_memory(
                messages, metadata=processed_metadata, prompt=prompt, llm=llm
            )
            # 注释：返回 `results` 给调用方。
            return results

        # 注释：判断条件 `self.config.llm.config.get("enable_vision")` 是否成立。
        if self.config.llm.config.get("enable_vision"):
            # 注释：计算并保存 消息列表，供后续逻辑使用。
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        # 注释：处理前面条件不成立时的默认分支。
        else:
            # 注释：计算并保存 消息列表，供后续逻辑使用。
            messages = parse_vision_messages(messages)

        # 注释：计算并保存 vector_store_result 变量，供后续逻辑使用。
        vector_store_result = await self._add_to_vector_store(messages, processed_metadata, effective_filters, infer, prompt=prompt)
        # 注释：返回 `{"results": vector_store_result}` 给调用方。
        return {"results": vector_store_result}

    # 注释：将消息转换为记忆并写入向量存储。
    async def _add_to_vector_store(
        self,
        messages: list,
        metadata: dict,
        effective_filters: dict,
        infer: bool,
        prompt: Optional[str] = None,
    ):
        # 注释：判断条件 `not infer` 是否成立。
        if not infer:
            # 注释：返回 `ed_memories = []` 给调用方。
            returned_memories = []
            # 注释：遍历 messages 中的元素，并将当前项赋给 message_dict。
            for message_dict in messages:
                # 注释：判断条件 `(` 是否成立。
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    # 注释：输出警告日志。
                    logger.warning(f"Skipping invalid message format (async): {message_dict}")
                    # 注释：跳过本轮循环剩余逻辑，继续处理下一项。
                    continue

                # 注释：判断条件 `message_dict["role"] == "system"` 是否成立。
                if message_dict["role"] == "system":
                    # 注释：跳过本轮循环剩余逻辑，继续处理下一项。
                    continue

                # 注释：深拷贝生成 per_msg_meta 变量，避免修改原始输入对象。
                per_msg_meta = deepcopy(metadata)
                # 注释：计算并保存 per_msg_meta 变量，供后续逻辑使用。
                per_msg_meta["role"] = message_dict["role"]

                # 注释：计算并保存 actor_name 变量，供后续逻辑使用。
                actor_name = message_dict.get("name")
                # 注释：判断条件 `actor_name` 是否成立。
                if actor_name:
                    # 注释：计算并保存 per_msg_meta 变量，供后续逻辑使用。
                    per_msg_meta["actor_id"] = actor_name

                # 注释：计算并保存 msg_content 变量，供后续逻辑使用。
                msg_content = message_dict["content"]
                # 注释：计算并保存 msg_embeddings 变量，供后续逻辑使用。
                msg_embeddings = await asyncio.to_thread(self.embedding_model.embed, msg_content, "add")
                # 注释：计算并保存 mem_id 变量，供后续逻辑使用。
                mem_id = await self._create_memory(msg_content, {msg_content: msg_embeddings}, per_msg_meta)

                # 注释：返回 `ed_memories.append(` 给调用方。
                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )
            # 注释：返回 `returned_memories` 给调用方。
            return returned_memories

        # === V3 PHASED BATCH PIPELINE (async) ===

        # Phase 0: Context gathering
        session_scope = _build_session_scope(effective_filters)
        # 注释：计算并保存 last_messages 变量，供后续逻辑使用。
        last_messages = await asyncio.to_thread(self.db.get_last_messages, session_scope, 10)
        # 注释：计算并保存 parsed_messages 变量，供后续逻辑使用。
        parsed_messages = parse_messages(messages)

        # Phase 1: Existing memory retrieval
        search_filters = {k: v for k, v in effective_filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        # 注释：计算并保存 query_embedding 变量，供后续逻辑使用。
        query_embedding = await asyncio.to_thread(self.embedding_model.embed, parsed_messages, "search")
        # 注释：计算并保存 已有记忆检索结果，供后续逻辑使用。
        existing_results = await asyncio.to_thread(
            self.vector_store.search,
            query=parsed_messages,
            vectors=query_embedding,
            top_k=10,
            filters=search_filters,
        )

        # Map UUIDs to integers (anti-hallucination)
        existing_memories = []
        # 注释：初始化 uuid_mapping 变量 为空字典，用于后续按键保存数据。
        uuid_mapping = {}
        # 注释：遍历 enumerate(existing_results) 中的元素，并将当前项赋给 idx, mem。
        for idx, mem in enumerate(existing_results):
            # 注释：计算并保存 uuid_mapping 变量，供后续逻辑使用。
            uuid_mapping[str(idx)] = mem.id
            # 注释：调用 existing_memories.append 执行对应操作。
            existing_memories.append({"id": str(idx), "text": mem.payload.get("data", "")})

        # Phase 2: LLM extraction (single call)
        is_agent_scoped = bool(effective_filters.get("agent_id")) and not effective_filters.get("user_id")
        # 注释：计算并保存 system_prompt 变量，供后续逻辑使用。
        system_prompt = ADDITIVE_EXTRACTION_PROMPT
        # 注释：判断条件 `is_agent_scoped` 是否成立。
        if is_agent_scoped:
            # 注释：在原有 system_prompt 变量 基础上累加/追加新的内容。
            system_prompt += AGENT_CONTEXT_SUFFIX

        # 注释：计算并保存 custom_instr 变量，供后续逻辑使用。
        custom_instr = prompt or self.custom_instructions

        # 注释：计算并保存 user_prompt 变量，供后续逻辑使用。
        user_prompt = generate_additive_extraction_prompt(
            existing_memories=existing_memories,
            new_messages=parsed_messages,
            last_k_messages=last_messages,
            custom_instructions=custom_instr,
        )

        # 注释：进入可能抛出异常的代码块。
        try:
            # 注释：计算并保存 模型响应，供后续逻辑使用。
            response = await asyncio.to_thread(
                self.llm.generate_response,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
        except Exception as e:
            # 注释：输出错误日志。
            logger.error(f"LLM extraction failed (async): {e}")
            # 注释：返回 `[]` 给调用方。
            return []

        # Parse response
        try:
            # 注释：计算并保存 模型响应，供后续逻辑使用。
            response = remove_code_blocks(response)
            # 注释：判断条件 `not response or not response.strip()` 是否成立。
            if not response or not response.strip():
                # 注释：初始化 extracted_memories 变量 为空列表，用于后续收集数据。
                extracted_memories = []
            # 注释：处理前面条件不成立时的默认分支。
            else:
                # 注释：进入可能抛出异常的代码块。
                try:
                    # 注释：计算并保存 extracted_memories 变量，供后续逻辑使用。
                    extracted_memories = json.loads(response, strict=False).get("memory", [])
                # 注释：捕获 json.JSONDecodeError 异常并执行降级或错误处理。
                except json.JSONDecodeError:
                    # 注释：计算并保存 extracted_json 变量，供后续逻辑使用。
                    extracted_json = extract_json(response)
                    # 注释：计算并保存 extracted_memories 变量，供后续逻辑使用。
                    extracted_memories = json.loads(extracted_json, strict=False).get("memory", [])
        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
        except Exception as e:
            # 注释：输出错误日志。
            logger.error(f"Error parsing extraction response (async): {e}")
            # 注释：初始化 extracted_memories 变量 为空列表，用于后续收集数据。
            extracted_memories = []

        # 注释：判断条件 `not extracted_memories` 是否成立。
        if not extracted_memories:
            # 注释：等待异步操作 `asyncio.to_thread(self.db.save_messages, messages, session_scope)` 完成。
            await asyncio.to_thread(self.db.save_messages, messages, session_scope)
            # 注释：返回 `[]` 给调用方。
            return []

        # Phase 3: Batch embed all extracted memory texts
        mem_texts = [m.get("text", "") for m in extracted_memories if m.get("text")]
        # 注释：进入可能抛出异常的代码块。
        try:
            # 注释：计算并保存 mem_embeddings_list 变量，供后续逻辑使用。
            mem_embeddings_list = await asyncio.to_thread(self.embedding_model.embed_batch, mem_texts, "add")
            # 注释：计算并保存 embed_map 变量，供后续逻辑使用。
            embed_map = dict(zip(mem_texts, mem_embeddings_list))
        # 注释：捕获 Exception 异常并执行降级或错误处理。
        except Exception:
            # 注释：初始化 embed_map 变量 为空字典，用于后续按键保存数据。
            embed_map = {}
            # 注释：遍历 mem_texts 中的元素，并将当前项赋给 text。
            for text in mem_texts:
                # 注释：进入可能抛出异常的代码块。
                try:
                    # 注释：计算并保存 embed_map 变量，供后续逻辑使用。
                    embed_map[text] = await asyncio.to_thread(self.embedding_model.embed, text, "add")
                # 注释：捕获 Exception as e 异常并执行降级或错误处理。
                except Exception as e:
                    # 注释：输出警告日志。
                    logger.warning(f"Failed to embed memory text (async): {e}")

        # Phase 4: Per-memory CPU processing + Phase 5: Hash dedup
        existing_hashes = set()
        # 注释：遍历 existing_results 中的元素，并将当前项赋给 mem。
        for mem in existing_results:
            # 注释：计算并保存 h 变量，供后续逻辑使用。
            h = mem.payload.get("hash") if hasattr(mem, "payload") and mem.payload else None
            # 注释：判断条件 `h` 是否成立。
            if h:
                # 注释：调用 existing_hashes.add 执行对应操作。
                existing_hashes.add(h)

        # 注释：初始化 待持久化记录列表 为空列表，用于后续收集数据。
        records = []
        # 注释：初始化 seen_hashes 变量 为空集合，用于后续去重。
        seen_hashes = set()
        # 注释：遍历 extracted_memories 中的元素，并将当前项赋给 mem。
        for mem in extracted_memories:
            # 注释：计算并保存 text 变量，供后续逻辑使用。
            text = mem.get("text")
            # 注释：判断条件 `not text or text not in embed_map` 是否成立。
            if not text or text not in embed_map:
                # 注释：跳过本轮循环剩余逻辑，继续处理下一项。
                continue

            # 注释：计算并保存 mem_hash 变量，供后续逻辑使用。
            mem_hash = hashlib.md5(text.encode()).hexdigest()
            # 注释：判断条件 `mem_hash in existing_hashes or mem_hash in seen_hashes` 是否成立。
            if mem_hash in existing_hashes or mem_hash in seen_hashes:
                # 注释：输出调试日志。
                logger.debug(f"Skipping duplicate memory (hash match, async): {text[:50]}")
                # 注释：跳过本轮循环剩余逻辑，继续处理下一项。
                continue
            # 注释：调用 seen_hashes.add 执行对应操作。
            seen_hashes.add(mem_hash)

            # 注释：计算并保存 text_lemmatized 变量，供后续逻辑使用。
            text_lemmatized = lemmatize_for_bm25(text)

            # 注释：计算并保存 记忆 ID，供后续逻辑使用。
            memory_id = str(uuid.uuid4())
            # 注释：深拷贝生成 mem_metadata 变量，避免修改原始输入对象。
            mem_metadata = deepcopy(metadata)
            # 注释：计算并保存 mem_metadata 变量，供后续逻辑使用。
            mem_metadata["data"] = text
            # 注释：计算并保存 mem_metadata 变量，供后续逻辑使用。
            mem_metadata["text_lemmatized"] = text_lemmatized
            # 注释：计算并保存 mem_metadata 变量，供后续逻辑使用。
            mem_metadata["hash"] = mem_hash
            # 注释：判断条件 `"created_at" not in mem_metadata` 是否成立。
            if "created_at" not in mem_metadata:
                # 注释：计算并保存 mem_metadata 变量，供后续逻辑使用。
                mem_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
            # 注释：计算并保存 mem_metadata 变量，供后续逻辑使用。
            mem_metadata["updated_at"] = mem_metadata["created_at"]
            # 注释：判断条件 `mem.get("attributed_to")` 是否成立。
            if mem.get("attributed_to"):
                # 注释：计算并保存 mem_metadata 变量，供后续逻辑使用。
                mem_metadata["attributed_to"] = mem["attributed_to"]

            # 注释：调用 records.append 执行对应操作。
            records.append((memory_id, text, embed_map[text], mem_metadata))

        # 注释：判断条件 `not records` 是否成立。
        if not records:
            # 注释：等待异步操作 `asyncio.to_thread(self.db.save_messages, messages, session_scope)` 完成。
            await asyncio.to_thread(self.db.save_messages, messages, session_scope)
            # 注释：返回 `[]` 给调用方。
            return []

        # Phase 6: Batch persist
        all_vectors = [r[2] for r in records]
        # 注释：计算并保存 all_ids 变量，供后续逻辑使用。
        all_ids = [r[0] for r in records]
        # 注释：计算并保存 all_payloads 变量，供后续逻辑使用。
        all_payloads = [r[3] for r in records]

        # 注释：进入可能抛出异常的代码块。
        try:
            # 注释：等待异步操作 `asyncio.to_thread(` 完成。
            await asyncio.to_thread(
                self.vector_store.insert,
                vectors=all_vectors,
                ids=all_ids,
                payloads=all_payloads,
            )
        # 注释：捕获 Exception 异常并执行降级或错误处理。
        except Exception:
            # 注释：遍历 zip(all_ids, all_vectors, all_payloads) 中的元素，并将当前项赋给 mid, vec, pay。
            for mid, vec, pay in zip(all_ids, all_vectors, all_payloads):
                # 注释：进入可能抛出异常的代码块。
                try:
                    # 注释：等待异步操作 `asyncio.to_thread(self.vector_store.insert, vectors=[vec], ids=[mid], payload...` 完成。
                    await asyncio.to_thread(self.vector_store.insert, vectors=[vec], ids=[mid], payloads=[pay])
                # 注释：捕获 Exception as e 异常并执行降级或错误处理。
                except Exception as e:
                    # 注释：输出错误日志。
                    logger.error(f"Failed to insert memory {mid} (async): {e}")

        # Batch history
        history_records = [
            {
                "memory_id": r[0],
                "old_memory": None,
                "new_memory": r[1],
                "event": "ADD",
                "created_at": r[3].get("created_at"),
                "is_deleted": 0,
            }
            for r in records
        ]
        # 注释：进入可能抛出异常的代码块。
        try:
            # 注释：等待异步操作 `asyncio.to_thread(self.db.batch_add_history, history_records)` 完成。
            await asyncio.to_thread(self.db.batch_add_history, history_records)
        # 注释：捕获 Exception 异常并执行降级或错误处理。
        except Exception:
            # 注释：遍历 history_records 中的元素，并将当前项赋给 hr。
            for hr in history_records:
                # 注释：进入可能抛出异常的代码块。
                try:
                    # 注释：等待异步操作 `asyncio.to_thread(` 完成。
                    await asyncio.to_thread(
                        self.db.add_history, hr["memory_id"], None, hr["new_memory"], "ADD",
                        created_at=hr.get("created_at")
                    )
                # 注释：捕获 Exception as e 异常并执行降级或错误处理。
                except Exception as e:
                    # 注释：输出错误日志。
                    logger.error(f"Failed to add history for {hr['memory_id']} (async): {e}")

        # Phase 7: Batch entity linking
        try:
            # 注释：计算并保存 all_texts 变量，供后续逻辑使用。
            all_texts = [r[1] for r in records]
            # 注释：计算并保存 all_entities 变量，供后续逻辑使用。
            all_entities = await asyncio.to_thread(extract_entities_batch, all_texts)

            # 7a: Global dedup
            global_entities = {}
            # 注释：遍历 enumerate(records) 中的元素，并将当前项赋给 idx, (memory_id, text, embedding, payload)。
            for idx, (memory_id, text, embedding, payload) in enumerate(records):
                # 注释：计算并保存 entities 变量，供后续逻辑使用。
                entities = all_entities[idx] if idx < len(all_entities) else []
                # 注释：遍历 entities 中的元素，并将当前项赋给 entity_type, entity_text。
                for entity_type, entity_text in entities:
                    # 注释：计算并保存 key 变量，供后续逻辑使用。
                    key = entity_text.strip().lower()
                    # 注释：判断条件 `key in global_entities` 是否成立。
                    if key in global_entities:
                        # 注释：执行当前语句，推进该函数的业务流程。
                        global_entities[key][2].add(memory_id)
                    # 注释：处理前面条件不成立时的默认分支。
                    else:
                        # 注释：计算并保存 global_entities 变量，供后续逻辑使用。
                        global_entities[key] = [entity_type, entity_text, {memory_id}]

            # 注释：判断条件 `global_entities` 是否成立。
            if global_entities:
                # 注释：计算并保存 ordered_keys 变量，供后续逻辑使用。
                ordered_keys = list(global_entities.keys())
                # 注释：计算并保存 entity_texts 变量，供后续逻辑使用。
                entity_texts = [global_entities[k][1] for k in ordered_keys]

                # 7b: Batch embed entities
                try:
                    # 注释：计算并保存 entity_embeddings 变量，供后续逻辑使用。
                    entity_embeddings = await asyncio.to_thread(self.embedding_model.embed_batch, entity_texts, "add")
                # 注释：捕获 Exception 异常并执行降级或错误处理。
                except Exception:
                    # 注释：初始化 entity_embeddings 变量 为空列表，用于后续收集数据。
                    entity_embeddings = []
                    # 注释：遍历 entity_texts 中的元素，并将当前项赋给 t。
                    for t in entity_texts:
                        # 注释：进入可能抛出异常的代码块。
                        try:
                            # 注释：调用 entity_embeddings.append 执行对应操作。
                            entity_embeddings.append(await asyncio.to_thread(self.embedding_model.embed, t, "add"))
                        # 注释：捕获 Exception 异常并执行降级或错误处理。
                        except Exception:
                            # 注释：调用 entity_embeddings.append 执行对应操作。
                            entity_embeddings.append(None)

                # 注释：计算并保存 valid 变量，供后续逻辑使用。
                valid = [(i, k) for i, k in enumerate(ordered_keys) if entity_embeddings[i] is not None]
                # 注释：判断条件 `valid` 是否成立。
                if valid:
                    # 注释：为 `valid_indices, valid_keys` 赋值，准备后续处理所需的数据。
                    valid_indices, valid_keys = zip(*valid)
                    # 注释：计算并保存 valid_vectors 变量，供后续逻辑使用。
                    valid_vectors = [entity_embeddings[i] for i in valid_indices]

                    # 7c: Batch search for existing entities
                    valid_texts = [global_entities[k][1] for k in valid_keys]
                    # 注释：计算并保存 existing_matches 变量，供后续逻辑使用。
                    existing_matches = await asyncio.to_thread(
                        self.entity_store.search_batch,
                        queries=valid_texts,
                        vectors_list=valid_vectors,
                        top_k=1,
                        filters=search_filters,
                    )

                    # 7d: Separate into inserts vs updates
                    to_insert_vectors, to_insert_ids, to_insert_payloads = [], [], []
                    # 注释：遍历 enumerate(valid_keys) 中的元素，并将当前项赋给 j, key。
                    for j, key in enumerate(valid_keys):
                        # 注释：为 `entity_type, entity_text, memory_ids` 赋值，准备后续处理所需的数据。
                        entity_type, entity_text, memory_ids = global_entities[key]
                        # 注释：计算并保存 matches 变量，供后续逻辑使用。
                        matches = existing_matches[j] if j < len(existing_matches) else []

                        # 注释：判断条件 `matches and matches[0].score >= 0.95` 是否成立。
                        if matches and matches[0].score >= 0.95:
                            # 注释：计算并保存 match 变量，供后续逻辑使用。
                            match = matches[0]
                            # 注释：计算并保存 向量库载荷数据，供后续逻辑使用。
                            payload = match.payload or {}
                            # 注释：计算并保存 linked 变量，供后续逻辑使用。
                            linked = set(payload.get("linked_memory_ids", []))
                            # 注释：计算并保存 linked 变量，供后续逻辑使用。
                            linked |= memory_ids
                            # 注释：计算并保存 向量库载荷数据，供后续逻辑使用。
                            payload["linked_memory_ids"] = sorted(linked)
                            # 注释：进入可能抛出异常的代码块。
                            try:
                                # 注释：等待异步操作 `asyncio.to_thread(` 完成。
                                await asyncio.to_thread(
                                    self.entity_store.update,
                                    vector_id=match.id,
                                    vector=None,
                                    payload=payload,
                                )
                            # 注释：捕获 Exception as e 异常并执行降级或错误处理。
                            except Exception as e:
                                # 注释：输出调试日志。
                                logger.debug(f"Entity update failed for '{entity_text}' (async): {e}")
                        # 注释：处理前面条件不成立时的默认分支。
                        else:
                            # 注释：调用 to_insert_vectors.append 执行对应操作。
                            to_insert_vectors.append(valid_vectors[j])
                            # 注释：调用 to_insert_ids.append 执行对应操作。
                            to_insert_ids.append(str(uuid.uuid4()))
                            # 注释：调用 to_insert_payloads.append 执行对应操作。
                            to_insert_payloads.append({
                                "data": entity_text,
                                "entity_type": entity_type,
                                "linked_memory_ids": sorted(memory_ids),
                                **search_filters,
                            })

                    # 7e: Batch insert new entities
                    if to_insert_vectors:
                        # 注释：进入可能抛出异常的代码块。
                        try:
                            # 注释：等待异步操作 `asyncio.to_thread(` 完成。
                            await asyncio.to_thread(
                                self.entity_store.insert,
                                vectors=to_insert_vectors,
                                ids=to_insert_ids,
                                payloads=to_insert_payloads,
                            )
                        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
                        except Exception as e:
                            # 注释：输出警告日志。
                            logger.warning(f"Batch entity insert failed (async): {e}")
        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
        except Exception as e:
            # 注释：输出警告日志。
            logger.warning(f"Batch entity linking failed (async): {e}")

        # Phase 8: Save messages + return
        await asyncio.to_thread(self.db.save_messages, messages, session_scope)

        # 注释：返回 `ed_memories = [` 给调用方。
        returned_memories = [
            {"id": r[0], "memory": r[1], "event": "ADD"}
            for r in records
        ]

        # 注释：为 `keys, encoded_ids` 赋值，准备后续处理所需的数据。
        keys, encoded_ids = process_telemetry_filters(effective_filters)
        # 注释：记录遥测事件，便于统计调用行为。
        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"},
        )
        # 注释：返回 `returned_memories` 给调用方。
        return returned_memories

    # 注释：按 ID 读取单条记忆。
    async def get(self, memory_id):
        """
        Retrieve a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        # 注释：记录遥测事件，便于统计调用行为。
        capture_event("mem0.get", self, {"memory_id": memory_id, "sync_type": "async"})
        # 注释：计算并保存 记忆内容，供后续逻辑使用。
        memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        # 注释：判断条件 `not memory` 是否成立。
        if not memory:
            # 注释：返回 `None` 给调用方。
            return None

        # 注释：计算并保存 需要提升到返回顶层的载荷字段，供后续逻辑使用。
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        # 注释：计算并保存 核心字段和已提升字段集合，供后续逻辑使用。
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        # 注释：计算并保存 result_item 变量，供后续逻辑使用。
        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload.get("data", ""),
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump()

        # 注释：遍历 promoted_payload_keys 中的元素，并将当前项赋给 key。
        for key in promoted_payload_keys:
            # 注释：判断条件 `key in memory.payload` 是否成立。
            if key in memory.payload:
                # 注释：计算并保存 result_item 变量，供后续逻辑使用。
                result_item[key] = memory.payload[key]

        # 注释：计算并保存 额外元数据，供后续逻辑使用。
        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        # 注释：判断条件 `additional_metadata` 是否成立。
        if additional_metadata:
            # 注释：计算并保存 result_item 变量，供后续逻辑使用。
            result_item["metadata"] = additional_metadata

        # 注释：返回 `result_item` 给调用方。
        return result_item

    # 注释：按照过滤条件列出记忆。
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
        _reject_top_level_entity_params(kwargs, "get_all")

        # Validate top_k
        _validate_search_params(top_k=top_k)

        # Validate and trim entity IDs in filters
        effective_filters = dict(filters) if filters else {}
        # 注释：判断条件 `"user_id" in effective_filters` 是否成立。
        if "user_id" in effective_filters:
            # 注释：计算并保存 规范化后的过滤条件，供后续逻辑使用。
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        # 注释：判断条件 `"agent_id" in effective_filters` 是否成立。
        if "agent_id" in effective_filters:
            # 注释：计算并保存 规范化后的过滤条件，供后续逻辑使用。
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        # 注释：判断条件 `"run_id" in effective_filters` 是否成立。
        if "run_id" in effective_filters:
            # 注释：计算并保存 规范化后的过滤条件，供后续逻辑使用。
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )

        # Validate filters contains at least one entity ID
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                "Example: filters={'user_id': 'u1'}"
            )

        # 注释：计算并保存 limit 变量，供后续逻辑使用。
        limit = top_k

        # 注释：为 `keys, encoded_ids` 赋值，准备后续处理所需的数据。
        keys, encoded_ids = process_telemetry_filters(effective_filters)
        # 注释：记录遥测事件，便于统计调用行为。
        capture_event(
            "mem0.get_all", self, {"limit": limit, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"}
        )

        # 注释：计算并保存 all_memories_result 变量，供后续逻辑使用。
        all_memories_result = await self._get_all_from_vector_store(effective_filters, limit)

        # 注释：返回 `{"results": all_memories_result}` 给调用方。
        return {"results": all_memories_result}

    # 注释：从向量库中读取并格式化多条记忆。
    async def _get_all_from_vector_store(self, filters, limit):
        # 注释：计算并保存 memories_result 变量，供后续逻辑使用。
        memories_result = await asyncio.to_thread(self.vector_store.list, filters=filters, top_k=limit)

        # Handle different vector store return formats by inspecting first element
        if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0:
            # 注释：计算并保存 first_element 变量，供后续逻辑使用。
            first_element = memories_result[0]

            # If first element is a container, unwrap one level
            if isinstance(first_element, (list, tuple)):
                # 注释：计算并保存 actual_memories 变量，供后续逻辑使用。
                actual_memories = first_element
            # 注释：处理前面条件不成立时的默认分支。
            else:
                # First element is a memory object, structure is already flat
                actual_memories = memories_result
        # 注释：处理前面条件不成立时的默认分支。
        else:
            # 注释：计算并保存 actual_memories 变量，供后续逻辑使用。
            actual_memories = memories_result

        # 注释：计算并保存 需要提升到返回顶层的载荷字段，供后续逻辑使用。
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        # 注释：计算并保存 核心字段和已提升字段集合，供后续逻辑使用。
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        # 注释：初始化 formatted_memories 变量 为空列表，用于后续收集数据。
        formatted_memories = []
        # 注释：遍历 actual_memories 中的元素，并将当前项赋给 mem。
        for mem in actual_memories:
            # 注释：计算并保存 格式化后的记忆字典，供后续逻辑使用。
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            ).model_dump(exclude={"score"})

            # 注释：遍历 promoted_payload_keys 中的元素，并将当前项赋给 key。
            for key in promoted_payload_keys:
                # 注释：判断条件 `key in mem.payload` 是否成立。
                if key in mem.payload:
                    # 注释：计算并保存 格式化后的记忆字典，供后续逻辑使用。
                    memory_item_dict[key] = mem.payload[key]

            # 注释：计算并保存 额外元数据，供后续逻辑使用。
            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            # 注释：判断条件 `additional_metadata` 是否成立。
            if additional_metadata:
                # 注释：计算并保存 格式化后的记忆字典，供后续逻辑使用。
                memory_item_dict["metadata"] = additional_metadata

            # 注释：调用 formatted_memories.append 执行对应操作。
            formatted_memories.append(memory_item_dict)

        # 注释：返回 `formatted_memories` 给调用方。
        return formatted_memories

    # 注释：根据查询文本搜索相关记忆。
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
        _reject_top_level_entity_params(kwargs, "search")

        # Validate search parameters (before applying defaults)
        _validate_search_params(threshold=threshold, top_k=top_k)

        # Validate and trim entity IDs in filters
        effective_filters = filters.copy() if filters else {}
        # 注释：判断条件 `"user_id" in effective_filters` 是否成立。
        if "user_id" in effective_filters:
            # 注释：计算并保存 规范化后的过滤条件，供后续逻辑使用。
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        # 注释：判断条件 `"agent_id" in effective_filters` 是否成立。
        if "agent_id" in effective_filters:
            # 注释：计算并保存 规范化后的过滤条件，供后续逻辑使用。
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        # 注释：判断条件 `"run_id" in effective_filters` 是否成立。
        if "run_id" in effective_filters:
            # 注释：计算并保存 规范化后的过滤条件，供后续逻辑使用。
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )

        # Validate filters contains at least one entity ID
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                "Example: filters={'user_id': 'u1'}"
            )

        # 注释：计算并保存 limit 变量，供后续逻辑使用。
        limit = top_k

        # Apply enhanced metadata filtering if advanced operators are detected
        if self._has_advanced_operators(effective_filters):
            # 注释：计算并保存 处理后的过滤条件，供后续逻辑使用。
            processed_filters = self._process_metadata_filters(effective_filters)
            # Remove logical/operator keys that have been reprocessed
            for logical_key in ("AND", "OR", "NOT"):
                # 注释：调用 effective_filters.pop 执行对应操作。
                effective_filters.pop(logical_key, None)
            # 注释：遍历 list(effective_filters.keys()) 中的元素，并将当前项赋给 fk。
            for fk in list(effective_filters.keys()):
                # 注释：判断条件 `fk not in ("AND", "OR", "NOT", "user_id", "agent_id", "run_id") and isinstanc...` 是否成立。
                if fk not in ("AND", "OR", "NOT", "user_id", "agent_id", "run_id") and isinstance(effective_filters.get(fk), dict):
                    # 注释：调用 effective_filters.pop 执行对应操作。
                    effective_filters.pop(fk, None)
            # 注释：调用 effective_filters.update 执行对应操作。
            effective_filters.update(processed_filters)

        # 注释：为 `keys, encoded_ids` 赋值，准备后续处理所需的数据。
        keys, encoded_ids = process_telemetry_filters(effective_filters)
        # 注释：记录遥测事件，便于统计调用行为。
        capture_event(
            "mem0.search",
            self,
            {
                "limit": limit,
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "async",
                "threshold": threshold,
                "advanced_filters": bool(filters and self._has_advanced_operators(filters)),
            },
        )

        # 注释：计算并保存 original_memories 变量，供后续逻辑使用。
        original_memories = await self._search_vector_store(query, effective_filters, limit, threshold)

        # Apply reranking if enabled and reranker is available
        if rerank and self.reranker and original_memories:
            # 注释：进入可能抛出异常的代码块。
            try:
                # Run reranking in thread pool to avoid blocking async loop
                reranked_memories = await asyncio.to_thread(
                    self.reranker.rerank, query, original_memories, limit
                )
                # 注释：计算并保存 original_memories 变量，供后续逻辑使用。
                original_memories = reranked_memories
            # 注释：捕获 Exception as e 异常并执行降级或错误处理。
            except Exception as e:
                # 注释：输出警告日志。
                logger.warning(f"Reranking failed, using original results: {e}")

        # 注释：返回 `{"results": original_memories}` 给调用方。
        return {"results": original_memories}

    # 注释：处理高级元数据过滤表达式。
    def _process_metadata_filters(self, metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process enhanced metadata filters and convert them to vector store compatible format.

        Args:
            metadata_filters: Enhanced metadata filters with operators

        Returns:
            Dict of processed filters compatible with vector store
        """
        # 注释：初始化 处理后的过滤条件 为空字典，用于后续按键保存数据。
        processed_filters = {}

        # 注释：定义 process_condition 函数/方法，封装一段可复用逻辑。
        def process_condition(key: str, condition: Any) -> Dict[str, Any]:
            # 注释：判断条件 `not isinstance(condition, dict)` 是否成立。
            if not isinstance(condition, dict):
                # Simple equality: {"key": "value"}
                if condition == "*":
                    # Wildcard: match everything for this field (implementation depends on vector store)
                    return {key: "*"}
                # 注释：返回 `{key: condition}` 给调用方。
                return {key: condition}

            # 注释：初始化 result 变量 为空字典，用于后续按键保存数据。
            result = {}
            # 注释：遍历 condition.items() 中的元素，并将当前项赋给 operator, value。
            for operator, value in condition.items():
                # Map platform operators to universal format that can be translated by each vector store
                operator_map = {
                    "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte",
                    "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                    "contains": "contains", "icontains": "icontains"
                }

                # 注释：判断条件 `operator in operator_map` 是否成立。
                if operator in operator_map:
                    # 注释：调用 result.setdefault 执行对应操作。
                    result.setdefault(key, {})[operator_map[operator]] = value
                # 注释：处理前面条件不成立时的默认分支。
                else:
                    # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
            # 注释：返回 `result` 给调用方。
            return result

        # 注释：定义 merge_filters 函数/方法，封装一段可复用逻辑。
        def merge_filters(target: Dict[str, Any], source: Dict[str, Any]) -> None:
            """Merge source into target, deep-merging nested operator dicts for the same key."""
            # 注释：遍历 source.items() 中的元素，并将当前项赋给 key, value。
            for key, value in source.items():
                # 注释：判断条件 `key in target and isinstance(target[key], dict) and isinstance(value, dict)` 是否成立。
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    # 注释：执行当前语句，推进该函数的业务流程。
                    target[key].update(value)
                # 注释：处理前面条件不成立时的默认分支。
                else:
                    # 注释：计算并保存 target 变量，供后续逻辑使用。
                    target[key] = value

        # 注释：遍历 metadata_filters.items() 中的元素，并将当前项赋给 key, value。
        for key, value in metadata_filters.items():
            # 注释：判断条件 `key == "AND"` 是否成立。
            if key == "AND":
                # Logical AND: combine multiple conditions
                if not isinstance(value, list):
                    # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
                    raise ValueError("AND operator requires a list of conditions")
                # 注释：遍历 value 中的元素，并将当前项赋给 condition。
                for condition in value:
                    # 注释：遍历 condition.items() 中的元素，并将当前项赋给 sub_key, sub_value。
                    for sub_key, sub_value in condition.items():
                        # 注释：调用 merge_filters 执行对应操作。
                        merge_filters(processed_filters, process_condition(sub_key, sub_value))
            # 注释：当前一个条件不成立时，继续判断 `key == "OR"`。
            elif key == "OR":
                # Logical OR: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
                    raise ValueError("OR operator requires a non-empty list of conditions")
                # Store OR conditions in a way that vector stores can interpret
                processed_filters["$or"] = []
                # 注释：遍历 value 中的元素，并将当前项赋给 condition。
                for condition in value:
                    # 注释：初始化 or_condition 变量 为空字典，用于后续按键保存数据。
                    or_condition = {}
                    # 注释：遍历 condition.items() 中的元素，并将当前项赋给 sub_key, sub_value。
                    for sub_key, sub_value in condition.items():
                        # 注释：调用 merge_filters 执行对应操作。
                        merge_filters(or_condition, process_condition(sub_key, sub_value))
                    # 注释：执行当前语句，推进该函数的业务流程。
                    processed_filters["$or"].append(or_condition)
            # 注释：当前一个条件不成立时，继续判断 `key == "NOT"`。
            elif key == "NOT":
                # Logical NOT: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
                    raise ValueError("NOT operator requires a non-empty list of conditions")
                # 注释：初始化 处理后的过滤条件 为空列表，用于后续收集数据。
                processed_filters["$not"] = []
                # 注释：遍历 value 中的元素，并将当前项赋给 condition。
                for condition in value:
                    # 注释：初始化 not_condition 变量 为空字典，用于后续按键保存数据。
                    not_condition = {}
                    # 注释：遍历 condition.items() 中的元素，并将当前项赋给 sub_key, sub_value。
                    for sub_key, sub_value in condition.items():
                        # 注释：调用 merge_filters 执行对应操作。
                        merge_filters(not_condition, process_condition(sub_key, sub_value))
                    # 注释：执行当前语句，推进该函数的业务流程。
                    processed_filters["$not"].append(not_condition)
            # 注释：处理前面条件不成立时的默认分支。
            else:
                # 注释：调用 merge_filters 执行对应操作。
                merge_filters(processed_filters, process_condition(key, value))

        # 注释：返回 `processed_filters` 给调用方。
        return processed_filters

    # 注释：判断过滤条件中是否包含高级操作符。
    def _has_advanced_operators(self, filters: Dict[str, Any]) -> bool:
        """
        Check if filters contain advanced operators that need special processing.

        Args:
            filters: Dictionary of filters to check

        Returns:
            bool: True if advanced operators are detected
        """
        # 注释：判断条件 `not isinstance(filters, dict)` 是否成立。
        if not isinstance(filters, dict):
            # 注释：返回 `False` 给调用方。
            return False

        # 注释：遍历 filters.items() 中的元素，并将当前项赋给 key, value。
        for key, value in filters.items():
            # Check for platform-style logical operators
            if key in ["AND", "OR", "NOT"]:
                # 注释：返回 `True` 给调用方。
                return True
            # Check for comparison operators (without $ prefix for universal compatibility)
            if isinstance(value, dict):
                # 注释：遍历 value.keys() 中的元素，并将当前项赋给 op。
                for op in value.keys():
                    # 注释：判断条件 `op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "iconta...` 是否成立。
                    if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                        # 注释：返回 `True` 给调用方。
                        return True
            # Check for wildcard values
            if value == "*":
                # 注释：返回 `True` 给调用方。
                return True
        # 注释：返回 `False` 给调用方。
        return False

    # 注释：执行向量检索、关键词检索和综合排序。
    async def _search_vector_store(self, query, filters, limit, threshold=0.1):
        # 注释：判断条件 `threshold is None` 是否成立。
        if threshold is None:
            # 注释：计算并保存 相似度阈值，供后续逻辑使用。
            threshold = 0.1

        # Step 1: Preprocess query (CPU-bound)
        query_lemmatized = await asyncio.to_thread(lemmatize_for_bm25, query)
        # 注释：计算并保存 query_entities 变量，供后续逻辑使用。
        query_entities = await asyncio.to_thread(extract_entities, query)

        # Step 2: Embed query
        embeddings = await asyncio.to_thread(self.embedding_model.embed, query, "search")

        # Step 3: Semantic search (over-fetch)
        internal_limit = max(limit * 4, 60)
        # 注释：计算并保存 semantic_results 变量，供后续逻辑使用。
        semantic_results = await asyncio.to_thread(
            self.vector_store.search, query=query, vectors=embeddings, top_k=internal_limit, filters=filters
        )

        # Step 4: Keyword search (if store supports it)
        keyword_results = await asyncio.to_thread(
            self.vector_store.keyword_search, query=query_lemmatized, top_k=internal_limit, filters=filters
        )

        # Step 5: Compute BM25 scores
        bm25_scores = {}
        # 注释：判断条件 `keyword_results is not None` 是否成立。
        if keyword_results is not None:
            # 注释：为 `midpoint, steepness` 赋值，准备后续处理所需的数据。
            midpoint, steepness = get_bm25_params(query, lemmatized=query_lemmatized)
            # 注释：遍历 keyword_results 中的元素，并将当前项赋给 mem。
            for mem in keyword_results:
                # 注释：计算并保存 mem_id 变量，供后续逻辑使用。
                mem_id = str(mem.id) if hasattr(mem, 'id') else str(mem.get('id', ''))
                # 注释：计算并保存 raw_score 变量，供后续逻辑使用。
                raw_score = mem.score if hasattr(mem, 'score') else mem.get('score', 0)
                # 注释：判断条件 `raw_score and raw_score > 0` 是否成立。
                if raw_score and raw_score > 0:
                    # 注释：计算并保存 bm25_scores 变量，供后续逻辑使用。
                    bm25_scores[mem_id] = normalize_bm25(raw_score, midpoint, steepness)

        # Step 6: Compute entity boosts
        entity_boosts = {}
        # 注释：判断条件 `query_entities` 是否成立。
        if query_entities:
            # 注释：计算并保存 entity_boosts 变量，供后续逻辑使用。
            entity_boosts = await self._compute_entity_boosts_async(query_entities, filters)

        # Step 7: Build candidate set from semantic results
        candidates = []
        # 注释：遍历 semantic_results 中的元素，并将当前项赋给 mem。
        for mem in semantic_results:
            # 注释：计算并保存 mem_id 变量，供后续逻辑使用。
            mem_id = str(mem.id)
            # 注释：调用 candidates.append 执行对应操作。
            candidates.append({
                "id": mem_id,
                "score": mem.score,
                "payload": mem.payload if hasattr(mem, 'payload') else {},
            })

        # Step 8: Score and rank
        scored_results = score_and_rank(
            semantic_results=candidates,
            bm25_scores=bm25_scores,
            entity_boosts=entity_boosts,
            threshold=threshold,
            top_k=limit,
        )

        # Step 9: Format results
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        # 注释：计算并保存 核心字段和已提升字段集合，供后续逻辑使用。
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        # 注释：初始化 original_memories 变量 为空列表，用于后续收集数据。
        original_memories = []
        # 注释：遍历 scored_results 中的元素，并将当前项赋给 scored。
        for scored in scored_results:
            # 注释：计算并保存 向量库载荷数据，供后续逻辑使用。
            payload = scored.get("payload") or {}
            # 注释：判断条件 `not payload.get("data")` 是否成立。
            if not payload.get("data"):
                # 注释：跳过本轮循环剩余逻辑，继续处理下一项。
                continue

            # 注释：计算并保存 格式化后的记忆字典，供后续逻辑使用。
            memory_item_dict = MemoryItem(
                id=scored["id"],
                memory=payload.get("data", ""),
                hash=payload.get("hash"),
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at"),
                score=scored["score"],
            ).model_dump()

            # 注释：遍历 promoted_payload_keys 中的元素，并将当前项赋给 key。
            for key in promoted_payload_keys:
                # 注释：判断条件 `key in payload` 是否成立。
                if key in payload:
                    # 注释：计算并保存 格式化后的记忆字典，供后续逻辑使用。
                    memory_item_dict[key] = payload[key]

            # 注释：计算并保存 额外元数据，供后续逻辑使用。
            additional_metadata = {k: v for k, v in payload.items() if k not in core_and_promoted_keys}
            # 注释：判断条件 `additional_metadata` 是否成立。
            if additional_metadata:
                # 注释：判断条件 `not memory_item_dict.get("metadata")` 是否成立。
                if not memory_item_dict.get("metadata"):
                    # 注释：初始化 格式化后的记忆字典 为空字典，用于后续按键保存数据。
                    memory_item_dict["metadata"] = {}
                # 注释：执行当前语句，推进该函数的业务流程。
                memory_item_dict["metadata"].update(additional_metadata)

            # 注释：调用 original_memories.append 执行对应操作。
            original_memories.append(memory_item_dict)

        # 注释：返回 `original_memories` 给调用方。
        return original_memories

    # 注释：定义 _compute_entity_boosts_async 函数/方法，封装一段可复用逻辑。
    async def _compute_entity_boosts_async(self, query_entities, filters):
        """Async version of entity boost computation."""
        # 注释：初始化 seen 变量 为空集合，用于后续去重。
        seen = set()
        # 注释：初始化 deduped 变量 为空列表，用于后续收集数据。
        deduped = []
        # 注释：遍历 query_entities[ 中的元素，并将当前项赋给 entity_type, entity_text。
        for entity_type, entity_text in query_entities[:8]:
            # 注释：计算并保存 key 变量，供后续逻辑使用。
            key = entity_text.strip().lower()
            # 注释：判断条件 `key and key not in seen` 是否成立。
            if key and key not in seen:
                # 注释：调用 seen.add 执行对应操作。
                seen.add(key)
                # 注释：调用 deduped.append 执行对应操作。
                deduped.append((entity_type, entity_text))

        # 注释：判断条件 `not deduped` 是否成立。
        if not deduped:
            # 注释：返回 `{}` 给调用方。
            return {}

        # 注释：计算并保存 检索过滤条件，供后续逻辑使用。
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        # 注释：初始化 memory_boosts 变量 为空字典，用于后续按键保存数据。
        memory_boosts = {}

        # 注释：进入可能抛出异常的代码块。
        try:
            # 注释：遍历 deduped 中的元素，并将当前项赋给 _, entity_text。
            for _, entity_text in deduped:
                # 注释：计算并保存 实体向量，供后续逻辑使用。
                entity_embedding = await asyncio.to_thread(self.embedding_model.embed, entity_text, "search")
                # 注释：计算并保存 matches 变量，供后续逻辑使用。
                matches = await asyncio.to_thread(
                    self.entity_store.search,
                    query=entity_text,
                    vectors=entity_embedding,
                    top_k=500,
                    filters=search_filters,
                )

                # 注释：遍历 matches 中的元素，并将当前项赋给 match。
                for match in matches:
                    # 注释：计算并保存 similarity 变量，供后续逻辑使用。
                    similarity = match.score if hasattr(match, 'score') else 0.0
                    # 注释：判断条件 `similarity < 0.5` 是否成立。
                    if similarity < 0.5:
                        # 注释：跳过本轮循环剩余逻辑，继续处理下一项。
                        continue

                    # 注释：计算并保存 向量库载荷数据，供后续逻辑使用。
                    payload = match.payload if hasattr(match, 'payload') else {}
                    # 注释：计算并保存 关联记忆 ID 列表，供后续逻辑使用。
                    linked_memory_ids = payload.get("linked_memory_ids", [])
                    # 注释：判断条件 `not isinstance(linked_memory_ids, list)` 是否成立。
                    if not isinstance(linked_memory_ids, list):
                        # 注释：跳过本轮循环剩余逻辑，继续处理下一项。
                        continue

                    # 注释：计算并保存 num_linked 变量，供后续逻辑使用。
                    num_linked = max(len(linked_memory_ids), 1)
                    # 注释：计算并保存 memory_count_weight 变量，供后续逻辑使用。
                    memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
                    # 注释：计算并保存 boost 变量，供后续逻辑使用。
                    boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight

                    # 注释：遍历 linked_memory_ids 中的元素，并将当前项赋给 memory_id。
                    for memory_id in linked_memory_ids:
                        # 注释：判断条件 `memory_id` 是否成立。
                        if memory_id:
                            # 注释：计算并保存 memory_key 变量，供后续逻辑使用。
                            memory_key = str(memory_id)
                            # 注释：计算并保存 memory_boosts 变量，供后续逻辑使用。
                            memory_boosts[memory_key] = max(memory_boosts.get(memory_key, 0.0), boost)

        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
        except Exception as e:
            # 注释：输出警告日志。
            logger.warning(f"Entity boost computation failed: {e}")

        # 注释：返回 `memory_boosts` 给调用方。
        return memory_boosts

    # 注释：更新指定 ID 的记忆内容。
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
        # 注释：记录遥测事件，便于统计调用行为。
        capture_event("mem0.update", self, {"memory_id": memory_id, "sync_type": "async"})

        # 注释：计算并保存 向量表示，供后续逻辑使用。
        embeddings = await asyncio.to_thread(self.embedding_model.embed, data, "update")
        # 注释：计算并保存 existing_embeddings 变量，供后续逻辑使用。
        existing_embeddings = {data: embeddings}

        # 注释：等待异步操作 `self._update_memory(memory_id, data, existing_embeddings, metadata)` 完成。
        await self._update_memory(memory_id, data, existing_embeddings, metadata)
        # 注释：返回 `{"message": "Memory updated successfully!"}` 给调用方。
        return {"message": "Memory updated successfully!"}

    # 注释：删除指定 ID 的记忆。
    async def delete(self, memory_id):
        """
        Delete a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        # 注释：记录遥测事件，便于统计调用行为。
        capture_event("mem0.delete", self, {"memory_id": memory_id, "sync_type": "async"})

        # 注释：计算并保存 existing_memory 变量，供后续逻辑使用。
        existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        # 注释：判断条件 `existing_memory is None` 是否成立。
        if existing_memory is None:
            # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
            raise ValueError(f"Memory with id {memory_id} not found")

        # 注释：等待异步操作 `self._delete_memory(memory_id, existing_memory)` 完成。
        await self._delete_memory(memory_id, existing_memory)
        # 注释：返回 `{"message": "Memory deleted successfully!"}` 给调用方。
        return {"message": "Memory deleted successfully!"}

    # 注释：按用户、代理或运行 ID 批量删除记忆。
    async def delete_all(self, user_id=None, agent_id=None, run_id=None):
        """
        Delete all memories asynchronously.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        # 注释：初始化 过滤条件 为空字典，用于后续按键保存数据。
        filters = {}
        # 注释：判断条件 `user_id` 是否成立。
        if user_id:
            # 注释：计算并保存 过滤条件，供后续逻辑使用。
            filters["user_id"] = user_id
        # 注释：判断条件 `agent_id` 是否成立。
        if agent_id:
            # 注释：计算并保存 过滤条件，供后续逻辑使用。
            filters["agent_id"] = agent_id
        # 注释：判断条件 `run_id` 是否成立。
        if run_id:
            # 注释：计算并保存 过滤条件，供后续逻辑使用。
            filters["run_id"] = run_id

        # 注释：判断条件 `not filters` 是否成立。
        if not filters:
            # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        # 注释：为 `keys, encoded_ids` 赋值，准备后续处理所需的数据。
        keys, encoded_ids = process_telemetry_filters(filters)
        # 注释：记录遥测事件，便于统计调用行为。
        capture_event("mem0.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"})
        # 注释：计算并保存 memories 变量，供后续逻辑使用。
        memories = await asyncio.to_thread(self.vector_store.list, filters=filters)

        # 注释：初始化 delete_tasks 变量 为空列表，用于后续收集数据。
        delete_tasks = []
        # 注释：遍历 memories[0] 中的元素，并将当前项赋给 memory。
        for memory in memories[0]:
            # 注释：调用 delete_tasks.append 执行对应操作。
            delete_tasks.append(self._delete_memory(memory.id))

        # 注释：等待异步操作 `asyncio.gather(*delete_tasks)` 完成。
        await asyncio.gather(*delete_tasks)

        # 注释：输出信息日志。
        logger.info(f"Deleted {len(memories[0])} memories")

        # 注释：返回 `{"message": "Memories deleted successfully!"}` 给调用方。
        return {"message": "Memories deleted successfully!"}

    # 注释：读取指定记忆的变更历史。
    async def history(self, memory_id):
        """
        Get the history of changes for a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        # 注释：记录遥测事件，便于统计调用行为。
        capture_event("mem0.history", self, {"memory_id": memory_id, "sync_type": "async"})
        # 注释：返回 `await asyncio.to_thread(self.db.get_history, memory_id)` 给调用方。
        return await asyncio.to_thread(self.db.get_history, memory_id)

    # 注释：创建一条新记忆并写入向量库和历史表。
    async def _create_memory(self, data, existing_embeddings, metadata=None):
        # 注释：输出调试日志。
        logger.debug(f"Creating memory with {data=}")
        # 注释：判断条件 `data in existing_embeddings` 是否成立。
        if data in existing_embeddings:
            # 注释：计算并保存 向量表示，供后续逻辑使用。
            embeddings = existing_embeddings[data]
        # 注释：处理前面条件不成立时的默认分支。
        else:
            # 注释：计算并保存 向量表示，供后续逻辑使用。
            embeddings = await asyncio.to_thread(self.embedding_model.embed, data, memory_action="add")

        # 注释：计算并保存 记忆 ID，供后续逻辑使用。
        memory_id = str(uuid.uuid4())
        # 注释：深拷贝生成 new_metadata 变量，避免修改原始输入对象。
        new_metadata = deepcopy(metadata) if metadata is not None else {}
        # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
        new_metadata["data"] = data
        # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        # 注释：判断条件 `"created_at" not in new_metadata` 是否成立。
        if "created_at" not in new_metadata:
            # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
            new_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
        new_metadata["updated_at"] = new_metadata["created_at"]
        # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data)

        # 注释：等待异步操作 `asyncio.to_thread(` 完成。
        await asyncio.to_thread(
            self.vector_store.insert,
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[new_metadata],
        )

        # 注释：等待异步操作 `asyncio.to_thread(` 完成。
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

        # 注释：返回 `memory_id` 给调用方。
        return memory_id

    # 注释：创建程序性记忆。
    async def _create_procedural_memory(self, messages, metadata=None, llm=None, prompt=None):
        """
        Create a procedural memory asynchronously

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            llm (llm, optional): LLM to use for the procedural memory creation. Defaults to None.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        # 注释：进入可能抛出异常的代码块。
        try:
            # 注释：从 langchain_core.messages.utils 模块批量导入后续列出的对象。
            from langchain_core.messages.utils import (
                convert_to_messages,  # type: ignore
            )
        # 注释：捕获 Exception 异常并执行降级或错误处理。
        except Exception:
            # 注释：输出错误日志。
            logger.error(
                "Import error while loading langchain-core. Please install 'langchain-core' to use procedural memory."
            )
            # 注释：执行当前语句，推进该函数的业务流程。
            raise

        # 注释：输出信息日志。
        logger.info("Creating procedural memory")

        # 注释：计算并保存 parsed_messages 变量，供后续逻辑使用。
        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {"role": "user", "content": "Create procedural memory of the above conversation."},
        ]

        # 注释：进入可能抛出异常的代码块。
        try:
            # 注释：判断条件 `llm is not None` 是否成立。
            if llm is not None:
                # 注释：计算并保存 parsed_messages 变量，供后续逻辑使用。
                parsed_messages = convert_to_messages(parsed_messages)
                # 注释：计算并保存 模型响应，供后续逻辑使用。
                response = await asyncio.to_thread(llm.invoke, input=parsed_messages)
                # 注释：计算并保存 procedural_memory 变量，供后续逻辑使用。
                procedural_memory = response.content
            # 注释：处理前面条件不成立时的默认分支。
            else:
                # 注释：计算并保存 procedural_memory 变量，供后续逻辑使用。
                procedural_memory = await asyncio.to_thread(self.llm.generate_response, messages=parsed_messages)
                # 注释：计算并保存 procedural_memory 变量，供后续逻辑使用。
                procedural_memory = remove_code_blocks(procedural_memory)
        
        # 注释：捕获 Exception as e 异常并执行降级或错误处理。
        except Exception as e:
            # 注释：输出错误日志。
            logger.error(f"Error generating procedural memory summary: {e}")
            # 注释：执行当前语句，推进该函数的业务流程。
            raise

        # 注释：判断条件 `metadata is None` 是否成立。
        if metadata is None:
            # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
            raise ValueError("Metadata cannot be done for procedural memory.")

        # 注释：计算并保存 元数据，供后续逻辑使用。
        metadata = {**metadata, "memory_type": MemoryType.PROCEDURAL.value}
        # 注释：计算并保存 向量表示，供后续逻辑使用。
        embeddings = await asyncio.to_thread(self.embedding_model.embed, procedural_memory, memory_action="add")
        # 注释：计算并保存 记忆 ID，供后续逻辑使用。
        memory_id = await self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        # 注释：记录遥测事件，便于统计调用行为。
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "async"})

        # 注释：计算并保存 result 变量，供后续逻辑使用。
        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        # 注释：返回 `result` 给调用方。
        return result

    # 注释：更新记忆的向量、载荷和历史记录。
    async def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        # 注释：输出信息日志。
        logger.info(f"Updating memory with {data=}")

        # 注释：进入可能抛出异常的代码块。
        try:
            # 注释：计算并保存 existing_memory 变量，供后续逻辑使用。
            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        # 注释：捕获 Exception 异常并执行降级或错误处理。
        except Exception:
            # 注释：输出错误日志。
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        # 注释：判断条件 `existing_memory is None` 是否成立。
        if existing_memory is None:
            # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
            raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")

        # 注释：计算并保存 prev_value 变量，供后续逻辑使用。
        prev_value = existing_memory.payload.get("data")

        # 注释：深拷贝生成 new_metadata 变量，避免修改原始输入对象。
        new_metadata = deepcopy(metadata) if metadata is not None else {}

        # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
        new_metadata["data"] = data
        # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data)
        # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
        new_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Preserve session identifiers from existing memory only if not provided in new metadata
        if "user_id" not in new_metadata and "user_id" in existing_memory.payload:
            # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        # 注释：判断条件 `"agent_id" not in new_metadata and "agent_id" in existing_memory.payload` 是否成立。
        if "agent_id" not in new_metadata and "agent_id" in existing_memory.payload:
            # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        # 注释：判断条件 `"run_id" not in new_metadata and "run_id" in existing_memory.payload` 是否成立。
        if "run_id" not in new_metadata and "run_id" in existing_memory.payload:
            # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
            new_metadata["run_id"] = existing_memory.payload["run_id"]

        # 注释：判断条件 `"actor_id" in existing_memory.payload` 是否成立。
        if "actor_id" in existing_memory.payload:
            # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        # 注释：判断条件 `"role" not in new_metadata and "role" in existing_memory.payload` 是否成立。
        if "role" not in new_metadata and "role" in existing_memory.payload:
            # 注释：计算并保存 new_metadata 变量，供后续逻辑使用。
            new_metadata["role"] = existing_memory.payload["role"]

        # 注释：判断条件 `data in existing_embeddings` 是否成立。
        if data in existing_embeddings:
            # 注释：计算并保存 向量表示，供后续逻辑使用。
            embeddings = existing_embeddings[data]
        # 注释：处理前面条件不成立时的默认分支。
        else:
            # 注释：计算并保存 向量表示，供后续逻辑使用。
            embeddings = await asyncio.to_thread(self.embedding_model.embed, data, "update")

        # 注释：等待异步操作 `asyncio.to_thread(` 完成。
        await asyncio.to_thread(
            self.vector_store.update,
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        # 注释：输出信息日志。
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        # 注释：等待异步操作 `asyncio.to_thread(` 完成。
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
        session_filters = {k: new_metadata[k] for k in ("user_id", "agent_id", "run_id") if new_metadata.get(k)}
        # 注释：等待异步操作 `self._remove_memory_from_entity_store(memory_id, session_filters)` 完成。
        await self._remove_memory_from_entity_store(memory_id, session_filters)
        # 注释：等待异步操作 `self._link_entities_for_memory(memory_id, data, session_filters)` 完成。
        await self._link_entities_for_memory(memory_id, data, session_filters)

        # 注释：返回 `memory_id` 给调用方。
        return memory_id

    # 注释：删除记忆并清理相关历史或实体索引。
    async def _delete_memory(self, memory_id, existing_memory=None):
        # 注释：输出信息日志。
        logger.info(f"Deleting memory with {memory_id=}")
        # 注释：判断条件 `existing_memory is None` 是否成立。
        if existing_memory is None:
            # 注释：计算并保存 existing_memory 变量，供后续逻辑使用。
            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
            # 注释：判断条件 `existing_memory is None` 是否成立。
            if existing_memory is None:
                # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
                raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")
        # 注释：计算并保存 prev_value 变量，供后续逻辑使用。
        prev_value = existing_memory.payload.get("data", "")
        # 注释：计算并保存 created_at 变量，供后续逻辑使用。
        created_at = _normalize_iso_timestamp_to_utc(existing_memory.payload.get("created_at"))
        # 注释：计算并保存 updated_at 变量，供后续逻辑使用。
        updated_at = datetime.now(timezone.utc).isoformat()
        # 注释：计算并保存 向量库载荷数据，供后续逻辑使用。
        payload = existing_memory.payload or {}
        # 注释：计算并保存 session_filters 变量，供后续逻辑使用。
        session_filters = {k: payload[k] for k in ("user_id", "agent_id", "run_id") if payload.get(k)}

        # 注释：等待异步操作 `asyncio.to_thread(self.vector_store.delete, vector_id=memory_id)` 完成。
        await asyncio.to_thread(self.vector_store.delete, vector_id=memory_id)
        # 注释：等待异步操作 `asyncio.to_thread(` 完成。
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
        await self._remove_memory_from_entity_store(memory_id, session_filters)

        # 注释：返回 `memory_id` 给调用方。
        return memory_id

    # 注释：重置底层存储中的所有记忆数据。
    async def reset(self):
        """
        Reset the memory store asynchronously by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        # 注释：输出警告日志。
        logger.warning("Resetting all memories")
        # 注释：等待异步操作 `asyncio.to_thread(self.vector_store.delete_col)` 完成。
        await asyncio.to_thread(self.vector_store.delete_col)

        # 注释：调用 gc.collect 执行对应操作。
        gc.collect()

        # 注释：判断条件 `hasattr(self.vector_store, "client") and hasattr(self.vector_store.client, "c...` 是否成立。
        if hasattr(self.vector_store, "client") and hasattr(self.vector_store.client, "close"):
            # 注释：等待异步操作 `asyncio.to_thread(self.vector_store.client.close)` 完成。
            await asyncio.to_thread(self.vector_store.client.close)

        # 注释：判断条件 `hasattr(self.db, "connection") and self.db.connection` 是否成立。
        if hasattr(self.db, "connection") and self.db.connection:
            # 注释：等待异步操作 `asyncio.to_thread(lambda: self.db.connection.execute("DROP TABLE IF EXISTS hi...` 完成。
            await asyncio.to_thread(lambda: self.db.connection.execute("DROP TABLE IF EXISTS history"))
            # 注释：等待异步操作 `asyncio.to_thread(self.db.connection.close)` 完成。
            await asyncio.to_thread(self.db.connection.close)

        # 注释：设置当前实例的 db 属性，用于后续方法共享状态。
        self.db = SQLiteManager(self.config.history_db_path)

        # 注释：设置当前实例的 vector_store 属性，用于后续方法共享状态。
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )

        # 注释：记录遥测事件，便于统计调用行为。
        capture_event("mem0.reset", self, {"sync_type": "async"})

    # 注释：定义 close 函数/方法，封装一段可复用逻辑。
    def close(self):
        """Release resources held by this AsyncMemory instance."""
        # 注释：判断条件 `hasattr(self, "db") and self.db is not None` 是否成立。
        if hasattr(self, "db") and self.db is not None:
            # 注释：调用 self.db.close 执行对应操作。
            self.db.close()
            # 注释：设置当前实例的 db 属性，用于后续方法共享状态。
            self.db = None

    # 注释：预留聊天接口。
    async def chat(self, query):
        # 注释：主动抛出异常，提示调用方当前输入或状态不合法。
        raise NotImplementedError("Chat function not implemented yet.")