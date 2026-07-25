import os
from dataclasses import dataclass
from typing import Any

from livekit.plugins import aws, deepgram, elevenlabs, sarvam
from livekit.agents.llm import LLM, LLMStream, ChatContext
from utils.langfuse_client import get_langfuse
import time


class ProviderAdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class VoiceProviderAdapters:
    stt: Any
    llm: Any
    tts: Any
    summary: dict[str, str]


def build_voice_provider_adapters(config: dict[str, Any]) -> VoiceProviderAdapters:
    stt = _build_stt(config["stt"], config["language"])
    
    call_id = config.get("call_id") # Note: we must ensure call_id is passed in the config
    raw_llm = _build_llm(config["llm"])
    llm = wrap_langfuse_llm(raw_llm, call_id, config["llm"]["model"])
    
    tts = _build_tts(config["tts"], config["language"])
    return VoiceProviderAdapters(
        stt=stt,
        llm=llm,
        tts=tts,
        summary={
            "stt_provider": config["stt"]["provider"],
            "stt_model": config["stt"]["model"],
            "llm_provider": config["llm"]["provider"],
            "llm_model": config["llm"]["model"],
            "tts_provider": config["tts"]["provider"],
            "tts_model": config["tts"]["model"],
            "tts_voice": config["tts"]["voice"],
        },
    )


def _build_stt(config: dict[str, Any], language: str):
    provider = config["provider"]
    model = config["model"]
    if provider == "deepgram":
        return deepgram.STT(
            model=model,
            language=_deepgram_language(language),
            api_key=_required_env("DEEPGRAM_API_KEY"),
        )
    if provider == "sarvam":
        return sarvam.STT(
            model=model,
            language=_sarvam_language(language),
            api_key=_required_env("SARVAM_API_KEY"),
        )
    raise ProviderAdapterError(f"unsupported STT provider: {provider}")


def _build_llm(config: dict[str, Any]):
    provider = config["provider"]
    if provider == "bedrock":
        kwargs = {
            "model": config["model"],
            "region": os.getenv("AWS_REGION", "us-east-1"),
        }
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if access_key or secret_key:
            if not access_key:
                raise ProviderAdapterError("AWS_ACCESS_KEY_ID is required when AWS_SECRET_ACCESS_KEY is set")
            if not secret_key:
                raise ProviderAdapterError("AWS_SECRET_ACCESS_KEY is required when AWS_ACCESS_KEY_ID is set")
            kwargs["api_key"] = access_key
            kwargs["api_secret"] = secret_key
        return aws.LLM(**kwargs)
    raise ProviderAdapterError(f"unsupported LLM provider: {provider}")


def _build_tts(config: dict[str, Any], language: str):
    provider = config["provider"]
    model = config["model"]
    voice = config["voice"]
    if provider == "elevenlabs":
        return elevenlabs.TTS(
            model=model,
            voice_id=voice,
            language=_elevenlabs_language(language),
            api_key=_required_env("ELEVENLABS_API_KEY"),
        )
    if provider == "deepgram":
        return deepgram.TTS(
            model=voice or model,
            api_key=_required_env("DEEPGRAM_API_KEY"),
        )
    if provider == "sarvam":
        return sarvam.TTS(
            model=model,
            speaker=voice,
            target_language_code=_sarvam_language(language),
            api_key=_required_env("SARVAM_API_KEY"),
        )
    raise ProviderAdapterError(f"unsupported TTS provider: {provider}")


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ProviderAdapterError(f"{name} is required for the selected voice provider")
    return value


def _deepgram_language(language: str) -> str:
    return {
        "en": "en-US",
        "en-IN": "en-IN",
        "hi": "hi",
    }.get(language, language)


def _elevenlabs_language(language: str) -> str:
    return {
        "en": "en",
        "en-IN": "en",
        "hi": "hi",
    }.get(language, language)


def _sarvam_language(language: str) -> str:
    return {
        "en": "en-IN",
        "en-IN": "en-IN",
        "hi": "hi-IN",
    }.get(language, language)


class LangfuseLLMStreamProxy(LLMStream):
    def __init__(self, inner_stream: LLMStream, generation: Any):
        super().__init__(
            llm=inner_stream.llm,
            chat_ctx=inner_stream.chat_ctx,
            fnc_ctx=inner_stream.fnc_ctx
        )
        self._inner = inner_stream
        self._generation = generation
        self._text_content = []
        self._start_time = time.time()

    def __getattr__(self, name):
        return getattr(self._inner, name)
        
    def __aiter__(self):
        return self
        
    async def __anext__(self):
        try:
            chunk = await self._inner.__anext__()
            if hasattr(chunk, "choices") and chunk.choices:
                for choice in chunk.choices:
                    if hasattr(choice, "delta") and hasattr(choice.delta, "content") and choice.delta.content:
                        self._text_content.append(choice.delta.content)
            return chunk
        except StopAsyncIteration:
            if self._generation:
                elapsed = time.time() - self._start_time
                self._generation.update(metadata={"total_latency_seconds": elapsed})
                self._generation.end(output="".join(self._text_content))
            raise
        except Exception as e:
            if self._generation:
                elapsed = time.time() - self._start_time
                self._generation.update(metadata={"total_latency_seconds": elapsed})
                self._generation.end(level="ERROR", status_message=str(e))
            raise


def wrap_langfuse_llm(llm_instance: LLM, call_id: str, model_name: str) -> LLM:
    original_chat = llm_instance.chat
    
    def chat_wrapper(chat_ctx: ChatContext, *args, **kwargs) -> LLMStream:
        client = get_langfuse()
        generation = None
        if client and call_id:
            try:
                messages = []
                for msg in chat_ctx.messages:
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    messages.append({"role": msg.role, "content": content})
                    
                trace = client.trace(id=call_id)
                generation = trace.generation(
                    name="agent_llm_turn",
                    model=model_name,
                    input=messages,
                )
            except Exception as e:
                from utils.logger import logger
                logger.warning(f"[langfuse] Failed to start generation: {e}")
                
        stream = original_chat(chat_ctx, *args, **kwargs)
        if not generation:
            return stream
            
        return LangfuseLLMStreamProxy(stream, generation)
        
    llm_instance.chat = chat_wrapper
    return llm_instance
