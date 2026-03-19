"""Public package exports for the AI Job Intelligence Assistant core modules."""

from job_intel.insights import CompanyInsightsStore
from job_intel.chat import ChatGraphPipeline
from job_intel.jobs import JobAggregator
from job_intel.llm import create_provider
from job_intel.matching import HybridMatcher
from job_intel.rag import (
    FaissVectorDB,
    HFTextEmbedder,
    HashTextEmbedder,
    LocalRAGRetriever,
    build_company_documents,
    make_embedder,
)
from job_intel.resume import ResumeParser
from job_intel.storage import JsonRepository

__all__ = [
    "ResumeParser",
    "JobAggregator",
    "HybridMatcher",
    "JsonRepository",
    "CompanyInsightsStore",
    "LocalRAGRetriever",
    "FaissVectorDB",
    "HFTextEmbedder",
    "HashTextEmbedder",
    "build_company_documents",
    "make_embedder",
    "create_provider",
    "ChatGraphPipeline",
]
