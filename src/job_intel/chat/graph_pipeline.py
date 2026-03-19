"""Graph-style coordinator for routing chat, jobs, resume parsing, and RAG context."""

from __future__ import annotations

import os
import difflib
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Any

from job_intel.jobs import JobAggregator
from job_intel.insights import CompanyInsightsStore
from job_intel.matching import HybridMatcher
from job_intel.rag import FaissVectorDB, LocalRAGRetriever, make_embedder
from job_intel.resume import ResumeParser
from job_intel.storage import ConversationStore, JsonRepository
from job_intel.types import CandidateProfile
from job_intel.utils.text import normalize_spaces, slugify, tokenize


TokenHandler = Callable[[str], None]


@dataclass
class GraphResult:
    assistant_text: str
    sources_used: list[str]
    route: str
    shown_jobs_count: int


class ChatGraphPipeline:
    """Small graph-style orchestrator for chat, jobs, and RAG-backed context."""

    def __init__(
        self,
        store: ConversationStore,
        repo: JsonRepository,
        rag_retriever: LocalRAGRetriever | None = None,
        time_window_hours: int = 24,
        initial_visible_jobs: int = 20,
        more_jobs_step: int = 20,
    ):
        self.store = store
        self.repo = repo
        self.rag_retriever = rag_retriever or LocalRAGRetriever()
        self.time_window_hours = time_window_hours
        self.initial_visible_jobs = initial_visible_jobs
        self.more_jobs_step = more_jobs_step

        self.resume_parser = ResumeParser()
        self.aggregator = JobAggregator()
        self.matcher = HybridMatcher()
        self.insights_store = CompanyInsightsStore()

        self.openai_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
        self.openai_resume_model = os.getenv("OPENAI_RESUME_MODEL", self.openai_model)
        self.openai_client = None
        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI

                self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception:
                self.openai_client = None

        self.vector_db = None
        self.embedder = None
        try:
            self.vector_db = FaissVectorDB("data/vector_store")
            self.embedder = make_embedder("auto")
        except Exception:
            self.vector_db = None
            self.embedder = None

        self._term_aliases = {
            "langchai": "langchain",
            "mangchai": "langchain",
            "langchan": "langchain",
            "langchian": "langchain",
        }
        self._known_terms = {
            "langchain",
            "langgraph",
            "llm",
            "openai",
            "python",
            "pytorch",
            "tensorflow",
            "streamlit",
            "fastapi",
            "docker",
            "kubernetes",
            "sql",
            "rag",
        }

    def run(
        self,
        *,
        thread_id: str,
        user_prompt: str,
        upload_name: str | None = None,
        upload_bytes: bytes | None = None,
        on_token: TokenHandler | None = None,
    ) -> GraphResult:
        prompt = (user_prompt or "").strip()
        thread = self.store.get_thread(thread_id)
        context = thread.get("context", {})

        route = self._route_intent(prompt, bool(upload_name))
        answer_parts: list[str] = []
        sources_used: list[str] = []

        # Node 1: optional resume ingestion + fresh jobs refresh.
        if upload_name and upload_bytes:
            # Pipeline: Extract -> Clean -> LLM parse -> Candidate JSON.
            resume_raw = self._extract_resume_text(upload_name, upload_bytes)
            resume_clean = self._clean_resume_text(resume_raw)
            candidate = self._parse_candidate_profile(upload_name, upload_bytes, resume_clean)
            query = self._build_query(candidate, prompt)
            ranked, diagnostics = self._fetch_ranked_jobs(candidate, query)
            if not ranked:
                fallback_query = self._build_query(candidate, "")
                if fallback_query != query:
                    ranked, diagnostics = self._fetch_ranked_jobs(candidate, fallback_query)
                    if ranked:
                        query = fallback_query
            if not ranked:
                ranked, diagnostics = self._fetch_ranked_jobs(candidate, "")
                if ranked:
                    query = "broad search"
            shown = min(self.initial_visible_jobs, len(ranked))

            run_id = f"{thread_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            self.repo.save_candidate(run_id, candidate)
            self.repo.save_jobs(run_id, [x.job for x in ranked])
            self.repo.save_matches(run_id, [x.match for x in ranked])

            job_catalog = [self._job_row(x.job) for x in ranked]
            self.store.update_context(
                thread_id,
                {
                    "resume_filename": upload_name,
                    "resume_text": resume_clean[:25000],
                    "query_used": query,
                    "time_window_hours": self.time_window_hours,
                    "job_catalog": job_catalog,
                    "shown_jobs_count": shown,
                    "last_run_id": run_id,
                    "job_fetch_diagnostics": diagnostics,
                },
            )
            context = self.store.get_thread(thread_id).get("context", {})

            if shown > 0:
                source_names = {str(j.get("source_name", "")).lower() for j in context.get("job_catalog", []) if j}
                if source_names == {"mock"}:
                    answer_parts.append(
                        "Live job sources were unavailable, so these are fallback sample jobs (not live postings)."
                    )
                elif not prompt:
                    answer_parts.append(
                        f"Fetched fresh jobs from the last {self.time_window_hours} hours. Showing top {shown} jobs."
                    )
                    answer_parts.append(self._render_jobs_for_chat(context.get("job_catalog", []), limit=20))
            else:
                answer_parts.append(
                    f"I could not find fresh jobs from the last {self.time_window_hours} hours for this query yet."
                )
                answer_parts.append(
                    "Try a more specific role prompt like: 'data scientist remote', 'ml engineer python', or upload resume and ask again."
                )

        # Node 2: route-specific response
        context = self.store.get_thread(thread_id).get("context", {})
        job_catalog = context.get("job_catalog", [])
        shown = int(context.get("shown_jobs_count", 0))

        if route == "more_jobs":
            start = shown
            end = min(shown + self.more_jobs_step, len(job_catalog))
            if start >= len(job_catalog):
                answer_parts.append("No more jobs left in this thread right now.")
            else:
                answer_parts.append(f"Showing jobs {start + 1} to {end} of {len(job_catalog)}:")
                for idx, item in enumerate(job_catalog[start:end], start=start + 1):
                    answer_parts.append(self._format_job_line(idx, item))
                self.store.update_context(thread_id, {"shown_jobs_count": end})
                shown = end

        elif route == "job_search" and not upload_name:
            query = prompt or context.get("query_used", "software engineer")
            jobs, diagnostics = self.aggregator.fetch_jobs_with_diagnostics(
                query=query,
                location=None,
                time_window_hours=self.time_window_hours,
            )
            if not jobs:
                jobs, diagnostics = self.aggregator.fetch_jobs_with_diagnostics(
                    query="",
                    location=None,
                    time_window_hours=self.time_window_hours,
                )
                if jobs:
                    query = "broad search"
            jobs_sorted = sorted(jobs, key=lambda x: x.posted_at, reverse=True)
            top = jobs_sorted[: self.initial_visible_jobs]
            job_catalog = [self._job_row(j) for j in top]
            shown = min(self.initial_visible_jobs, len(job_catalog))
            self.store.update_context(
                thread_id,
                {
                    "query_used": query,
                    "time_window_hours": self.time_window_hours,
                    "job_catalog": job_catalog,
                    "shown_jobs_count": shown,
                    "job_fetch_diagnostics": diagnostics,
                },
            )
            if shown > 0:
                source_names = {str(j.get("source_name", "")).lower() for j in job_catalog if j}
                if source_names == {"mock"}:
                    answer_parts.append(
                        "Live job sources were unavailable, so these are fallback sample jobs (not live postings)."
                    )
                elif not prompt:
                    answer_parts.append(
                        f"Fetched fresh jobs from the last {self.time_window_hours} hours. Showing top {shown} jobs."
                    )
                    answer_parts.append(self._render_jobs_for_chat(job_catalog, limit=20))
            else:
                answer_parts.append(
                    f"No fresh jobs found in the last {self.time_window_hours} hours for this query."
                )
                answer_parts.append("Try: 'data scientist remote', 'python ml engineer', or 'analytics intern'.")

        handled_structured = False
        if prompt and self._is_culture_prompt(prompt):
            culture_text, culture_sources = self._render_culture_report(context, prompt)
            if culture_text:
                answer_parts.append(culture_text)
                sources_used = culture_sources
                handled_structured = True

        # Node 3: LLM chat generation (streaming) for normal Q&A or richer job explanation.
        context = self.store.get_thread(thread_id).get("context", {})
        if prompt and route != "more_jobs" and not handled_structured:
            llm_text = self._generate_chat_reply(
                thread_id=thread_id,
                question=prompt,
                route=route,
                context=context,
                on_token=on_token,
            )
            if llm_text:
                answer_parts.append(llm_text)
            if not sources_used and context.get("job_catalog"):
                sources_used = self._build_rag_insights(context.get("job_catalog", []), max_items=3, question=prompt)

        if not answer_parts:
            answer_parts.append("Ready. Upload resume or ask a question.")

        final_text = "\n".join(answer_parts)
        if sources_used:
            self.store.update_context(
                thread_id,
                {
                    "last_sources_used": sources_used,
                    "last_sources_question": prompt or "latest jobs context",
                },
            )
        return GraphResult(
            assistant_text=final_text,
            sources_used=sources_used,
            route=route,
            shown_jobs_count=shown,
        )

    def _route_intent(self, prompt: str, has_upload: bool) -> str:
        if has_upload:
            return "job_search"
        t = tokenize(prompt.lower())
        if {"more", "next", "additional"} & t or ("show" in t and "jobs" in t):
            return "more_jobs"
        if {"job", "jobs", "hiring", "openings", "roles"} & t:
            return "job_search"
        return "chat"

    @staticmethod
    def _build_query(candidate: CandidateProfile, prompt: str) -> str:
        if prompt.strip() and not ChatGraphPipeline._is_generic_job_prompt(prompt):
            return prompt.strip()
        if candidate.parsed_roles:
            return " ".join(candidate.parsed_roles[:2])
        if candidate.parsed_skills:
            return " ".join(candidate.parsed_skills[:5])
        return "software engineer"

    def _extract_resume_text(self, filename: str, raw_bytes: bytes) -> str:
        return self.resume_parser.extract_text(filename, raw_bytes)

    @staticmethod
    def _clean_resume_text(text: str) -> str:
        return normalize_spaces(text)

    def _parse_candidate_profile(
        self,
        filename: str,
        raw_bytes: bytes,
        cleaned_text: str,
    ) -> CandidateProfile:
        llm_candidate = self._llm_resume_parse(cleaned_text)
        if llm_candidate:
            return llm_candidate
        # Fallback to deterministic parser.
        if cleaned_text:
            return self.resume_parser.parse_text(cleaned_text)
        return self.resume_parser.parse_upload(filename, raw_bytes)

    def _llm_resume_parse(self, cleaned_text: str) -> CandidateProfile | None:
        if self.openai_client is None or not cleaned_text.strip():
            return None
        schema_hint = (
            '{"raw_text":"string","parsed_skills":["string"],"parsed_roles":["string"],'
            '"years_experience": number|null, "preferred_locations":["string"]}'
        )
        prompt = (
            "Extract candidate profile from resume text and return JSON only.\n"
            f"Schema:\n{schema_hint}\n\n"
            f"Resume text:\n{cleaned_text[:12000]}"
        )
        try:
            resp = self.openai_client.responses.create(
                model=self.openai_resume_model,
                input=[
                    {"role": "system", "content": "You extract structured resume fields as strict JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_output_tokens=260,
                temperature=0.0,
            )
            text = (getattr(resp, "output_text", "") or "").strip()
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1:
                return None
            import json

            data = json.loads(text[start : end + 1])
            return CandidateProfile(
                raw_text=normalize_spaces(str(data.get("raw_text") or cleaned_text))[:25000],
                parsed_skills=sorted(set(str(x).strip().lower() for x in (data.get("parsed_skills") or []) if str(x).strip())),
                parsed_roles=sorted(set(str(x).strip().lower() for x in (data.get("parsed_roles") or []) if str(x).strip())),
                years_experience=float(data["years_experience"]) if data.get("years_experience") not in (None, "") else None,
                preferred_locations=sorted(
                    set(str(x).strip() for x in (data.get("preferred_locations") or []) if str(x).strip())
                ),
            )
        except Exception:
            return None

    @staticmethod
    def _is_generic_job_prompt(prompt: str) -> bool:
        t = tokenize(prompt.lower())
        generic = {
            "find",
            "me",
            "some",
            "job",
            "jobs",
            "show",
            "latest",
            "new",
            "role",
            "roles",
            "please",
            "for",
            "a",
            "an",
            "the",
            "hi",
            "hello",
        }
        meaningful = [x for x in t if x not in generic]
        return len(meaningful) < 2

    def _fetch_ranked_jobs(self, candidate: Any, query: str):
        jobs, diagnostics = self.aggregator.fetch_jobs_with_diagnostics(
            query=query,
            location=None,
            time_window_hours=self.time_window_hours,
        )
        ranked = self.matcher.rank_jobs(candidate, jobs, top_k=max(self.initial_visible_jobs, len(jobs)))
        return ranked, diagnostics

    @staticmethod
    def _job_row(job: Any) -> dict[str, str]:
        return {
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "posted_at": job.posted_at,
            "source_name": job.source_name,
            "source_url": job.source_url,
            "salary": getattr(job, "salary", "") or "Not listed",
            "description": normalize_spaces(job.description)[:1200],
        }

    @staticmethod
    def _format_job_line(idx: int, item: dict[str, Any]) -> str:
        return (
            f"{idx}. {item.get('title', 'N/A')} @ {item.get('company', 'N/A')}"
            f" | {item.get('location', 'N/A')} | posted={item.get('posted_at', 'N/A')}"
        )

    def _render_jobs_for_chat(self, job_catalog: list[dict[str, Any]], limit: int = 20) -> str:
        visible = min(limit, len(job_catalog))
        lines = [f"### Top {visible} Latest Jobs", ""]
        lines.append("| # | Role | Location | Salary | Source | Apply |")
        lines.append("|---|------|----------|--------|--------|-------|")
        for idx, item in enumerate(job_catalog[:visible], start=1):
            role = f"{item.get('title', 'N/A')} @ {item.get('company', 'N/A')}"
            loc = item.get("location", "N/A")
            sal = item.get("salary", "Not listed")
            src = item.get("source_name", "N/A")
            url = item.get("source_url", "")
            apply = f"[Apply]({url})" if url else "N/A"
            lines.append(f"| {idx} | {role} | {loc} | {sal} | {src} | {apply} |")
        return "\n".join(lines)

    @staticmethod
    def _is_culture_prompt(prompt: str) -> bool:
        t = tokenize((prompt or "").lower())
        return bool({"culture", "workplace", "environment", "values"} & t)

    @staticmethod
    def _trim_text(text: str, limit: int = 120) -> str:
        cleaned = normalize_spaces(text or "")
        if len(cleaned) <= limit:
            return cleaned or "N/A"
        return cleaned[: limit - 3].rstrip() + "..."

    def _render_culture_report(self, context: dict[str, Any], question: str) -> tuple[str, list[str]]:
        jobs = context.get("job_catalog", []) or []
        seen: set[str] = set()
        companies: list[str] = []
        for j in jobs:
            c = normalize_spaces(str(j.get("company", "")))
            if not c:
                continue
            slug = slugify(c)
            if slug in seen:
                continue
            seen.add(slug)
            companies.append(c)
            if len(companies) >= 12:
                break

        if not companies:
            return (
                "I do not have company names in this thread yet. Ask for jobs first, then I can summarize culture.",
                [],
            )

        lines = ["### Company Culture Snapshot", ""]
        lines.append("| Company | Official Culture | Community Signal | Watch-outs |")
        lines.append("|---|---|---|---|")
        rag_sources: list[str] = []

        for company in companies:
            profile = self.insights_store.load_by_company(company)
            if profile.get("status") == "missing":
                official = "Profile not available yet."
                community = "No structured profile found."
                watchouts = "N/A"
            else:
                official = self._trim_text(
                    profile.get("culture_summary_official")
                    or profile.get("company_overview")
                    or profile.get("final_blended_insight")
                    or "N/A"
                )
                community = self._trim_text(
                    profile.get("culture_summary_community")
                    or profile.get("interview_experience_summary")
                    or profile.get("final_blended_insight")
                    or "N/A"
                )
                flags = profile.get("reddit_red_flags") or []
                if isinstance(flags, list) and flags:
                    watchouts = self._trim_text("; ".join(str(x) for x in flags[:2]), limit=100)
                else:
                    watchouts = "No major red flags captured."

            lines.append(
                f"| {company} | {official} | {community} | {watchouts} |"
            )

        rag_lines = self._build_rag_insights(jobs, max_items=3, question=question)
        if rag_lines:
            lines.append("")
            lines.append("### Retrieved Context (RAG)")
            lines.extend(rag_lines[:3])
            rag_sources = rag_lines[:3]

        lines.append("")
        lines.append("_Use this as directional guidance; team-level culture can vary._")
        return ("\n".join(lines), rag_sources)

    def _build_rag_insights(self, job_catalog: list[dict[str, Any]], max_items: int, question: str) -> list[str]:
        company_filter = [item.get("company", "") for item in job_catalog[: max(1, max_items)] if item.get("company")]
        company_filter_slugs = {slugify(c) for c in company_filter if c}
        lines: list[str] = []

        def pretty_company(md: dict[str, Any], fallback_slug: str = "") -> str:
            company = md.get("company", "")
            if isinstance(company, str) and company.strip():
                return company.strip()
            slug = (md.get("company_slug", "") or fallback_slug).strip()
            return slug.replace("_", " ").title() if slug else "Unknown"

        def pretty_section(section: str) -> str:
            return (section or "source").replace("_", " ")

        if self.vector_db is not None and self.embedder is not None:
            try:
                rows = self.vector_db.search(question, self.embedder, top_k=max_items * 6)
                if rows:
                    filtered = []
                    for row in rows:
                        md = row.get("metadata", {})
                        row_slug = md.get("company_slug", "")
                        if company_filter_slugs and row_slug not in company_filter_slugs:
                            continue
                        filtered.append(row)
                        if len(filtered) >= max_items:
                            break
                    if not filtered:
                        filtered = rows[:max_items]
                    for row in filtered:
                        md = row.get("metadata", {})
                        lines.append(
                            f"- {pretty_company(md)} • {pretty_section(md.get('section', 'chunk'))}: {row.get('text', '')[:260]}"
                        )
                    return lines
            except Exception:
                pass

        evidence = self.rag_retriever.query(question=question, company_filter=company_filter, top_k=max_items)
        if not evidence:
            return []
        for e in evidence:
            lines.append(
                f"- {e.get('company_slug', 'unknown').replace('_', ' ').title()} • {(e.get('source', 'source') or '').replace('_', ' ')}: {e.get('text', '')[:260]}"
            )
        return lines

    def _generate_chat_reply(
        self,
        *,
        thread_id: str,
        question: str,
        route: str,
        context: dict[str, Any],
        on_token: TokenHandler | None,
    ) -> str:
        corrected_question = self._normalize_question(question)
        correction_note = ""
        if corrected_question != question:
            correction_note = f"Likely intended question: {corrected_question}"

        short_term = self.store.get_short_term(thread_id, n=6)
        conversation_snippets = [
            f"{m.get('role', 'user')}: {normalize_spaces(m.get('content', ''))[:180]}" for m in short_term
        ]
        jobs = context.get("job_catalog", [])[:20]
        source_names = {str(j.get("source_name", "")).lower() for j in jobs if j}
        using_mock_only = bool(jobs) and source_names == {"mock"}
        # Always attempt RAG retrieval, even for normal chat without jobs.
        rag_lines = self._build_rag_insights(jobs, max_items=4, question=corrected_question)
        resume_text = normalize_spaces(context.get("resume_text", ""))[:2200]

        if self.openai_client is None:
            return self._fallback_chat_reply(corrected_question, route, jobs, rag_lines, resume_text)

        system = (
            "You are a practical career assistant. "
            "Keep answers concise and actionable. "
            "For job/resume/thread-specific questions, use provided context. "
            "For general knowledge questions, answer directly from your own knowledge. "
            "If jobs exist and the route is job_search/more_jobs, present a decorated top-jobs list with: "
            "title/company, location, source, posted time, salary (or Not listed), and link. "
            "Include short company backstory only when relevant evidence exists. "
            "Do not claim jobs were fetched unless they are explicitly present in Top jobs context."
        )
        if using_mock_only:
            system += " These jobs are fallback samples because live sources were unavailable; state that clearly."
        if correction_note:
            system += " If typo correction hint exists, acknowledge it briefly in one sentence."

        job_lines = [
            f"- {j.get('title')} @ {j.get('company')} | {j.get('location')} | posted={j.get('posted_at')}"
            for j in jobs[:8]
        ]
        user_msg = (
            f"Route: {route}\n"
            f"User question: {question}\n"
            + (f"Correction hint: {correction_note}\n" if correction_note else "")
            + f"Model question: {corrected_question}\n\n"
            f"Short-term memory:\n" + ("\n".join(conversation_snippets) or "(none)") + "\n\n"
            f"Resume extract:\n{resume_text or '(none)'}\n\n"
            f"Top jobs context:\n" + ("\n".join(job_lines) or "(none)") + "\n\n"
            f"RAG evidence:\n" + ("\n".join(rag_lines) or "(none)")
        )

        if on_token:
            try:
                streamed = self._llm_complete(system, user_msg, on_token=on_token)
                return self._merge_job_snapshot(route, jobs, streamed, rag_lines)
            except Exception:
                return self._fallback_chat_reply(corrected_question, route, jobs, rag_lines, resume_text)

        # Non-streaming path keeps validator loop.
        max_attempts = 2
        feedback: list[str] = []
        final_text = ""

        try:
            for _ in range(max_attempts):
                repair_msg = ""
                if feedback:
                    repair_msg = (
                        "\n\nValidator feedback from previous draft:\n- "
                        + "\n- ".join(feedback)
                        + "\nRevise to fix all issues."
                    )
                draft = self._llm_complete(system, user_msg + repair_msg)
                if not draft:
                    continue
                ok, feedback = self._validate_answer(
                    route=route,
                    question=corrected_question,
                    answer=draft,
                    jobs=jobs,
                    rag_lines=rag_lines,
                    time_window_hours=self.time_window_hours,
                )
                final_text = draft
                if ok:
                    break
        except Exception:
            return self._fallback_chat_reply(corrected_question, route, jobs, rag_lines, resume_text)

        if not final_text:
            return self._fallback_chat_reply(corrected_question, route, jobs, rag_lines, resume_text)
        return self._merge_job_snapshot(route, jobs, final_text, rag_lines)

    @staticmethod
    def _fallback_chat_reply(
        question: str,
        route: str,
        jobs: list[dict[str, Any]],
        rag_lines: list[str],
        resume_text: str,
    ) -> str:
        lines = ["Here is a quick answer from your current thread context:"]
        lines.append(f"- Question: {question}")
        lines.append(f"- Route used: {route}")
        lines.append(f"- Resume loaded: {'yes' if resume_text else 'no'}")
        lines.append(f"- Jobs in memory: {len(jobs)}")
        if jobs:
            lines.append("- Top jobs:")
            for j in jobs[:5]:
                lines.append(f"  - {j.get('title')} @ {j.get('company')} ({j.get('location')})")
        if rag_lines:
            lines.append("- Company backstory:")
            lines.extend(rag_lines[:3])
        return "\n".join(lines)

    def _llm_complete(self, system_prompt: str, user_prompt: str, on_token: TokenHandler | None = None) -> str:
        if on_token:
            chunks: list[str] = []
            stream = self.openai_client.responses.create(
                model=self.openai_model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_output_tokens=380,
                temperature=0.2,
                stream=True,
            )
            for event in stream:
                if getattr(event, "type", "") == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        chunks.append(delta)
                        on_token(delta)
            return normalize_spaces("".join(chunks)).strip()

        resp = self.openai_client.responses.create(
            model=self.openai_model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_output_tokens=380,
            temperature=0.2,
        )
        text = normalize_spaces(getattr(resp, "output_text", "") or "")
        return text.strip()

    def _merge_job_snapshot(
        self,
        route: str,
        jobs: list[dict[str, Any]],
        answer: str,
        rag_lines: list[str] | None = None,
    ) -> str:
        rag_block = ""
        if rag_lines:
            rag_block = "\n\n### Retrieved Context (RAG)\n" + "\n".join(
                f"- {x.lstrip('- ').strip()}" for x in rag_lines[:3]
            )

        if route in {"job_search", "more_jobs"} and jobs:
            snapshot = self._render_jobs_for_chat(jobs, limit=20)
            return f"{snapshot}{rag_block}\n\n{answer}".strip()

        if rag_block:
            return f"{answer}{rag_block}".strip()
        return answer

    def _normalize_question(self, question: str) -> str:
        tokens = (question or "").split()
        if not tokens:
            return question

        fixed: list[str] = []
        for tok in tokens:
            raw = tok.strip()
            t = raw.lower()
            if t in self._term_aliases:
                fixed.append(self._term_aliases[t])
                continue
            if len(t) >= 5 and t.isalpha():
                match = difflib.get_close_matches(t, list(self._known_terms), n=1, cutoff=0.72)
                if match:
                    fixed.append(match[0])
                    continue
            fixed.append(raw)
        return " ".join(fixed)

    @staticmethod
    def _stream_text(text: str, on_token: TokenHandler):
        for token in text.split(" "):
            on_token(token + " ")

    @staticmethod
    def _validate_answer(
        *,
        route: str,
        question: str,
        answer: str,
        jobs: list[dict[str, Any]],
        rag_lines: list[str],
        time_window_hours: int,
    ) -> tuple[bool, list[str]]:
        feedback: list[str] = []
        answer_l = (answer or "").lower()

        if len(answer.strip()) < 80:
            feedback.append("Answer is too short; provide clearer actionable detail.")

        if route == "job_search":
            if jobs and not any(j.get("title", "").lower() in answer_l for j in jobs[:5]):
                feedback.append("Mention at least one concrete job title from the available jobs.")
            window_token = str(time_window_hours)
            if jobs and window_token not in answer_l and f"last {window_token}" not in answer_l:
                feedback.append(f"Remind the user jobs are from the last {time_window_hours} hours.")
            if rag_lines and "backstory" not in answer_l and "insight" not in answer_l:
                feedback.append("Include short company backstory or insight by default.")
            if not jobs and "no fresh jobs" not in answer_l and "could not find" not in answer_l:
                feedback.append("State clearly that no fresh jobs were found for this query.")

        # Always ensure the user question intent is addressed.
        q_tokens = tokenize(question)
        a_tokens = tokenize(answer)
        if q_tokens and len(q_tokens & a_tokens) == 0:
            feedback.append("Address the user's actual question more directly.")

        return (len(feedback) == 0), feedback
