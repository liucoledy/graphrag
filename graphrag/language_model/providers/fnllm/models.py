# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing fnllm model provider definitions."""

from __future__ import annotations
import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING
import torch
import torch.nn.functional as F
from fnllm.openai import (
    create_openai_chat_llm,
    create_openai_client,
    create_openai_embeddings_llm,
)
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from graphrag.language_model.providers.fnllm.events import FNLLMEvents
from graphrag.language_model.providers.fnllm.utils import (
    _create_cache,
    _create_error_handler,
    _create_openai_config,
    run_coroutine_sync,
)
from graphrag.language_model.response.base import (
    BaseModelOutput,
    BaseModelResponse,
    ModelResponse,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from fnllm.openai.types.client import OpenAIChatLLM as FNLLMChatLLM
    from fnllm.openai.types.client import OpenAIEmbeddingsLLM as FNLLMEmbeddingLLM

    from graphrag.cache.pipeline_cache import PipelineCache
    from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
    from graphrag.config.models.language_model_config import (
        LanguageModelConfig,
    )


class OpenAIChatFNLLM:
    """An OpenAI Chat Model provider using the fnllm library."""

    model: FNLLMChatLLM

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        model_config = _create_openai_config(config, azure=False)
        error_handler = _create_error_handler(callbacks) if callbacks else None
        model_cache = _create_cache(cache, name)
        client = create_openai_client(model_config)
        self.model = create_openai_chat_llm(
            model_config,
            client=client,
            cache=model_cache,
            events=FNLLMEvents(error_handler) if error_handler else None,
        )
        self.config = config

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        if history is None:
            response = await self.model(prompt, **kwargs)
        else:
            response = await self.model(prompt, history=history, **kwargs)
        return BaseModelResponse(
            output=BaseModelOutput(
                content=response.output.content,
                full_response=response.output.raw_model.to_dict(),
            ),
            parsed_response=response.parsed_json,
            history=response.history,
            cache_hit=response.cache_hit,
            tool_calls=response.tool_calls,
            metrics=response.metrics,
        )

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        if history is None:
            response = await self.model(prompt, stream=True, **kwargs)
        else:
            response = await self.model(prompt, history=history, stream=True, **kwargs)
        async for chunk in response.output.content:
            if chunk is not None:
                yield chunk

    def chat(self, prompt: str, history: list | None = None, **kwargs) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        return run_coroutine_sync(self.achat(prompt, history=history, **kwargs))

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> Generator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        msg = "chat_stream is not supported for synchronous execution"
        raise NotImplementedError(msg)


class OpenAIEmbeddingFNLLM:
    """An OpenAI Embedding Model provider using the fnllm library."""

    model: FNLLMEmbeddingLLM

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        model_config = _create_openai_config(config, azure=False)
        error_handler = _create_error_handler(callbacks) if callbacks else None
        model_cache = _create_cache(cache, name)
        client = create_openai_client(model_config)
        self.model = create_openai_embeddings_llm(
            model_config,
            client=client,
            cache=model_cache,
            events=FNLLMEvents(error_handler) if error_handler else None,
        )
        self.config = config

    async def aembed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the LLM.

        Returns
        -------
            The embeddings of the text.
        """
        response = await self.model(text_list, **kwargs)
        if response.output.embeddings is None:
            msg = "No embeddings found in response"
            raise ValueError(msg)
        embeddings: list[list[float]] = response.output.embeddings
        return embeddings

    async def aembed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        response = await self.model([text], **kwargs)
        if response.output.embeddings is None:
            msg = "No embeddings found in response"
            raise ValueError(msg)
        embeddings: list[float] = response.output.embeddings[0]
        return embeddings

    def embed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the LLM.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed_batch(text_list, **kwargs))

    def embed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed(text, **kwargs))


class AzureOpenAIChatFNLLM:
    """An Azure OpenAI Chat LLM provider using the fnllm library."""

    model: FNLLMChatLLM

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        model_config = _create_openai_config(config, azure=True)
        error_handler = _create_error_handler(callbacks) if callbacks else None
        model_cache = _create_cache(cache, name)
        client = create_openai_client(model_config)
        self.model = create_openai_chat_llm(
            model_config,
            client=client,
            cache=model_cache,
            events=FNLLMEvents(error_handler) if error_handler else None,
        )
        self.config = config

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            history: The conversation history.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        if history is None:
            response = await self.model(prompt, **kwargs)
        else:
            response = await self.model(prompt, history=history, **kwargs)
        return BaseModelResponse(
            output=BaseModelOutput(
                content=response.output.content,
                full_response=response.output.raw_model.to_dict(),
            ),
            parsed_response=response.parsed_json,
            history=response.history,
            cache_hit=response.cache_hit,
            tool_calls=response.tool_calls,
            metrics=response.metrics,
        )

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            history: The conversation history.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        if history is None:
            response = await self.model(prompt, stream=True, **kwargs)
        else:
            response = await self.model(prompt, history=history, stream=True, **kwargs)
        async for chunk in response.output.content:
            if chunk is not None:
                yield chunk

    def chat(self, prompt: str, history: list | None = None, **kwargs) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        return run_coroutine_sync(self.achat(prompt, history=history, **kwargs))

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> Generator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        msg = "chat_stream is not supported for synchronous execution"
        raise NotImplementedError(msg)


class AzureOpenAIEmbeddingFNLLM:
    """An Azure OpenAI Embedding Model provider using the fnllm library."""

    model: FNLLMEmbeddingLLM

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        model_config = _create_openai_config(config, azure=True)
        error_handler = _create_error_handler(callbacks) if callbacks else None
        model_cache = _create_cache(cache, name)
        client = create_openai_client(model_config)
        self.model = create_openai_embeddings_llm(
            model_config,
            client=client,
            cache=model_cache,
            events=FNLLMEvents(error_handler) if error_handler else None,
        )
        self.config = config

    async def aembed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        response = await self.model(text_list, **kwargs)
        if response.output.embeddings is None:
            msg = "No embeddings found in response"
            raise ValueError(msg)
        embeddings: list[list[float]] = response.output.embeddings
        return embeddings

    async def aembed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        response = await self.model([text], **kwargs)
        if response.output.embeddings is None:
            msg = "No embeddings found in response"
            raise ValueError(msg)
        embeddings: list[float] = response.output.embeddings[0]
        return embeddings

    def embed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed_batch(text_list, **kwargs))

    def embed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed(text, **kwargs))


# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing HuggingFace model provider definitions."""


class LocalHuggingFaceEmbeddingModel:
    """A local HuggingFace Embedding Model provider using sentence-transformers."""

    def __init__(
            self,
            *,
            name: str,
            config: LanguageModelConfig,
            callbacks: WorkflowCallbacks | None = None,
            cache: PipelineCache | None = None,
            **kwargs  # 添加kwargs来接收自定义参数
    ) -> None:
        self.config = config
        self.model_name = config.model or "nomic-ai/nomic-embed-text-v1.5"
        # 从kwargs中获取自定义参数，如果没有则使用默认值
        self.task_type = kwargs.get('task_type', 'search_document')
        self.matryoshka_dim = kwargs.get('matryoshka_dim', 768)
        self.use_sentence_transformers = kwargs.get('use_sentence_transformers', True)

        # 设置模型缓存目录
        self._setup_cache_directory()

        # 根据Nomic文档，需要添加任务指令前缀
        self.task_prefixes = {
            'search_document': 'search_document: ',
            'search_query': 'search_query: ',
            'clustering': 'clustering: ',
            'classification': 'classification: '
        }

        # 初始化模型
        self._init_model()

    def _setup_cache_directory(self):
        """设置模型缓存目录"""
        # 获取项目根目录
        project_root = Path(__file__).parent.parent.parent.parent.parent
        models_dir = project_root / "models"

        # 创建模型目录
        models_dir.mkdir(exist_ok=True)

        # 设置环境变量
        os.environ["HF_HOME"] = str(models_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(models_dir)
        os.environ["HF_DATASETS_CACHE"] = str(models_dir)

        print(f"模型缓存目录设置为: {models_dir}")

    def _init_model(self):
        """初始化模型"""
        print(f"正在加载模型: {self.model_name}")

        if self.use_sentence_transformers:
            # 使用sentence-transformers (推荐)
            self.model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
                cache_folder=os.environ.get("HF_HOME")
            )
            self.tokenizer = None
        else:
            # 使用transformers (更灵活但需要更多配置)
            self.tokenizer = AutoTokenizer.from_pretrained(
                'bert-base-uncased',
                cache_dir=os.environ.get("HF_HOME")
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                safe_serialization=True,
                cache_dir=os.environ.get("HF_HOME")
            )
            self.model.eval()

        print(f"模型加载完成: {self.model_name}")

    def _add_task_prefix(self, text: str) -> str:
        """为文本添加任务指令前缀"""
        prefix = self.task_prefixes.get(self.task_type, 'search_document: ')
        return f"{prefix}{text}"

    def _add_task_prefix_batch(self, texts: list[str]) -> list[str]:
        """为批量文本添加任务指令前缀"""
        prefix = self.task_prefixes.get(self.task_type, 'search_document: ')
        return [f"{prefix}{text}" for text in texts]

    def _mean_pooling(self, model_output, attention_mask):
        """平均池化函数"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _get_embeddings_sentence_transformers(self, texts: list[str]) -> list[list[float]]:
        """使用sentence-transformers获取embeddings"""
        # 添加任务前缀
        prefixed_texts = self._add_task_prefix_batch(texts)

        # 获取embeddings
        embeddings = self.model.encode(prefixed_texts, convert_to_tensor=True)

        # 应用layer normalization (Nomic模型要求)
        embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))

        # 截取到指定维度 (Matryoshka Representation Learning)
        embeddings = embeddings[:, :self.matryoshka_dim]

        # 归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().tolist()

    def _get_embeddings_transformers(self, texts: list[str]) -> list[list[float]]:
        """使用transformers获取embeddings"""
        # 添加任务前缀
        prefixed_texts = self._add_task_prefix_batch(texts)

        # 编码输入
        encoded_input = self.tokenizer(
            prefixed_texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        # 获取embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # 平均池化
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])

        # 应用layer normalization
        embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))

        # 截取到指定维度
        embeddings = embeddings[:, :self.matryoshka_dim]

        # 归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().tolist()

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """获取embeddings的统一接口"""
        if self.use_sentence_transformers:
            return self._get_embeddings_sentence_transformers(texts)
        else:
            return self._get_embeddings_transformers(texts)

    async def aembed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text list using the local HuggingFace Model.

        Args:
            text_list: The list of texts to embed.
            kwargs: Additional arguments to pass to the model.

        Returns
        -------
            The embeddings of the texts.
        """
        # 在线程池中运行以避免阻塞
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, self._get_embeddings, text_list
        )
        return embeddings

    async def aembed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the local HuggingFace Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the model.

        Returns
        -------
            The embedding of the text.
        """
        embeddings = await self.aembed_batch([text], **kwargs)
        return embeddings[0]

    def embed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text list using the local HuggingFace Model.

        Args:
            text_list: The list of texts to embed.
            kwargs: Additional arguments to pass to the model.

        Returns
        -------
            The embeddings of the texts.
        """
        return run_coroutine_sync(self.aembed_batch(text_list, **kwargs))

    def embed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the local HuggingFace Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the model.

        Returns
        -------
            The embedding of the text.
        """
        return run_coroutine_sync(self.aembed(text, **kwargs))