"""Small, focused tools that specialists call.

Most tools are pure stdlib helpers. ``StdlibWebClient`` is the bounded,
opt-in network surface used only by the web research specialist.
"""

from .calculator import safe_eval
from .jsonutil import extract_json
from .openai_web_search import OpenAIWebSearchAnswerer, WebAnswerer, openai_web_search_from_env
from .text import chunk_text, merge_evidence, search_chunks
from .web import StdlibWebClient, WebClient, dumps_sources, extract_urls, html_to_text, pdf_bytes_to_text

__all__ = [
    "OpenAIWebSearchAnswerer",
    "StdlibWebClient",
    "WebAnswerer",
    "WebClient",
    "chunk_text",
    "dumps_sources",
    "extract_json",
    "extract_urls",
    "html_to_text",
    "merge_evidence",
    "openai_web_search_from_env",
    "pdf_bytes_to_text",
    "safe_eval",
    "search_chunks",
]
