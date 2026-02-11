"""
Confluence MCP Server - Configuration

Pydantic Settings for all configuration via environment variables.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, Literal
from pathlib import Path


class ConfluenceSettings(BaseSettings):
    """Confluence API configuration."""
    base_url: str = Field(..., alias="CONFLUENCE_BASE_URL")
    username: str = Field(..., alias="CONFLUENCE_USERNAME")
    api_token: str = Field(..., alias="CONFLUENCE_API_TOKEN")
    space_key: str = Field("DOCS", alias="CONFLUENCE_SPACE_KEY")

    model_config = {"env_prefix": "", "extra": "ignore"}


class QdrantSettings(BaseSettings):
    """Qdrant vector database configuration."""
    host: str = Field("localhost", alias="QDRANT_HOST")
    port: int = Field(6333, alias="QDRANT_PORT")
    collection: str = Field("confluence_pages", alias="QDRANT_COLLECTION")
    api_key: Optional[str] = Field(None, alias="QDRANT_API_KEY")

    model_config = {"env_prefix": "", "extra": "ignore"}


class LLMSettings(BaseSettings):
    """LLM provider configuration."""
    provider: Literal["lm_studio", "openai", "azure"] = Field(
        "lm_studio", alias="LLM_PROVIDER"
    )
    base_url: str = Field("http://localhost:1234/v1", alias="LLM_BASE_URL")
    api_key: Optional[str] = Field(None, alias="LLM_API_KEY")
    model: str = Field("qwen2.5-7b-instruct", alias="LLM_MODEL")
    timeout_ms: int = Field(30000, alias="LLM_TIMEOUT_MS")

    model_config = {"env_prefix": "", "extra": "ignore"}


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""
    model_dense: str = Field(
        "BAAI/bge-base-en-v1.5", alias="EMBEDDING_MODEL_DENSE"
    )
    model_sparse: str = Field(
        "prithivida/Splade_PP_en_v1", alias="EMBEDDING_MODEL_SPARSE"
    )
    batch_size: int = Field(32, alias="EMBEDDING_BATCH_SIZE")
    cache_dir: Path = Field(
        Path("./models_cache"), alias="MODELS_CACHE_DIR"
    )

    model_config = {"env_prefix": "", "extra": "ignore"}


class MCPSettings(BaseSettings):
    """MCP server configuration."""
    transport: Literal["sse", "stdio"] = Field("sse", alias="MCP_TRANSPORT")
    port: int = Field(8080, alias="MCP_PORT")
    host: str = Field("0.0.0.0", alias="MCP_HOST")

    model_config = {"env_prefix": "", "extra": "ignore"}


class PipelineSettings(BaseSettings):
    """Data pipeline configuration."""
    max_pages: int = Field(3000, alias="MAX_PAGES_TO_CRAWL")
    concurrency: int = Field(10, alias="PIPELINE_CONCURRENCY")
    enable_llm_summaries: bool = Field(True, alias="ENABLE_LLM_SUMMARIES")
    background_llm_summaries: bool = Field(True, alias="BACKGROUND_LLM_SUMMARIES")
    llm_summary_batch_size: int = Field(5, alias="LLM_SUMMARY_BATCH_SIZE")

    model_config = {"env_prefix": "", "extra": "ignore"}


class CacheSettings(BaseSettings):
    """Caching configuration."""
    enabled: bool = Field(True, alias="CACHE_ENABLED")
    ttl_query: int = Field(300, alias="CACHE_TTL_QUERY_SECONDS")
    ttl_content: int = Field(900, alias="CACHE_TTL_CONTENT_SECONDS")
    max_size_mb: int = Field(1024, alias="CACHE_MAX_SIZE_MB")

    model_config = {"env_prefix": "", "extra": "ignore"}


class RAGSettings(BaseSettings):
    """RAG configuration."""
    max_context_sections: int = Field(3, alias="RAG_MAX_CONTEXT_SECTIONS")
    max_section_chars: int = Field(2000, alias="RAG_MAX_SECTION_CHARS")
    verify_grounding: bool = Field(True, alias="RAG_VERIFY_GROUNDING")

    model_config = {"env_prefix": "", "extra": "ignore"}


class LogSettings(BaseSettings):
    """Logging configuration."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO", alias="LOG_LEVEL"
    )
    format: Literal["json", "text"] = Field("json", alias="LOG_FORMAT")

    model_config = {"env_prefix": "", "extra": "ignore"}


class Settings(BaseSettings):
    """Main settings aggregating all configuration."""
    confluence: ConfluenceSettings = Field(default_factory=ConfluenceSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    log: LogSettings = Field(default_factory=LogSettings)

    model_config = {"env_prefix": "", "extra": "ignore"}


def get_settings() -> Settings:
    """Load settings from environment variables."""
    from dotenv import load_dotenv
    load_dotenv()
    return Settings()
