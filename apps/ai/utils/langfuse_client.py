import os
from typing import Any, Optional
from utils.logger import logger

_langfuse = None

def get_langfuse() -> Any:
    global _langfuse
    if _langfuse is None:
        try:
            from langfuse import Langfuse
            if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
                _langfuse = Langfuse()
                logger.info("[langfuse] Client initialized")
        except ImportError:
            logger.warning("[langfuse] langfuse package not installed")
        except Exception as error:
            logger.error(f"[langfuse] Failed to initialize: {error}")
    return _langfuse

def create_trace(
    call_id: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[dict] = None
) -> None:
    client = get_langfuse()
    if not client:
        return
    try:
        client.trace(
            id=call_id,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            tags=["voice-agent"],
        )
    except Exception as e:
        logger.warning(f"[langfuse] Failed to create trace: {e}")

def score_evaluation(
    call_id: str,
    name: str,
    value: Any,
    comment: Optional[str] = None
) -> None:
    client = get_langfuse()
    if not client:
        return
    try:
        numeric_value = None
        if isinstance(value, bool):
            numeric_value = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            numeric_value = float(value)
            
        client.score(
            trace_id=call_id,
            name=name,
            value=numeric_value if numeric_value is not None else 0.0,
            string_value=str(value) if numeric_value is None else None,
            comment=comment
        )
    except Exception as e:
        logger.warning(f"[langfuse] Failed to record score: {e}")

def flush_langfuse() -> None:
    client = get_langfuse()
    if client:
        try:
            client.flush()
            logger.info("[langfuse] Flushed pending events")
        except Exception as e:
            logger.warning(f"[langfuse] Failed to flush: {e}")
