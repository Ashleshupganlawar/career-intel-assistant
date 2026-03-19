"""Streamlit UI for threaded chat, resume upload, jobs, and RAG-assisted answers."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_intel.chat import ChatGraphPipeline
from job_intel.insights import CompanyInsightsStore
from job_intel.storage import ConversationStore, JsonRepository
from job_intel.utils.text import normalize_spaces, tokenize


DEFAULT_TIME_WINDOW_HOURS = 24
INITIAL_VISIBLE_JOBS = 20
MORE_JOBS_STEP = 20

st.set_page_config(page_title="AI Job Intelligence Assistant", layout="wide")

store = ConversationStore()
repo = JsonRepository()
insights_store = CompanyInsightsStore()
graph = ChatGraphPipeline(
    store=store,
    repo=repo,
    time_window_hours=DEFAULT_TIME_WINDOW_HOURS,
    initial_visible_jobs=INITIAL_VISIBLE_JOBS,
    more_jobs_step=MORE_JOBS_STEP,
)


def summarize_prompt(text: str, max_words: int = 18) -> str:
    words = normalize_spaces(text).split(" ")
    if not words or words == [""]:
        return "(empty prompt)"
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")


def is_more_jobs_prompt(question: str) -> bool:
    tokens = tokenize((question or "").lower())
    return (
        "more" in tokens
        or "next" in tokens
        or "additional" in tokens
        or ("show" in tokens and "jobs" in tokens)
    )


def derive_query(candidate) -> str:
    if candidate.parsed_roles:
        return " ".join(candidate.parsed_roles[:2])
    if candidate.parsed_skills:
        return " ".join(candidate.parsed_skills[:5])
    return "software engineer"


def thread_summary(thread: dict) -> str:
    msgs = thread.get("messages", [])

    def clean_for_summary(text: str) -> str:
        text = normalize_spaces(text)
        text = re.sub(r"\[resume:[^\]]+\]", "", text, flags=re.IGNORECASE)
        text = normalize_spaces(text)
        return text

    user_msgs = [
        clean_for_summary(m.get("content", ""))
        for m in msgs
        if m.get("role") == "user" and clean_for_summary(m.get("content", ""))
    ]
    if not user_msgs:
        return thread.get("title", "New conversation")

    # Dynamic summary from conversation trajectory (first intent -> latest intent).
    if len(user_msgs) == 1:
        merged = user_msgs[0]
    else:
        first = user_msgs[0]
        latest = user_msgs[-1]
        merged = first if first == latest else f"{first} -> {latest}"
    words = merged.split(" ")
    if len(words) <= 14:
        return merged
    return " ".join(words[:14]) + "..."


def format_job_line(idx: int, item: dict) -> str:
    return (
        f"{idx}. {item.get('title', 'N/A')} @ {item.get('company', 'N/A')}"
        f" | {item.get('location', 'N/A')} | posted={item.get('posted_at', 'N/A')}"
    )


def build_rag_insights(
    job_catalog: list[dict],
    max_items: int = 3,
    question: str = "hiring process culture interview themes red flags",
) -> list[str]:
    company_filter = [item.get("company", "") for item in job_catalog[: max(1, max_items)] if item.get("company")]
    company_filter_slugs = {slugify(c) for c in company_filter if c}
    lines: list[str] = []
    
    def pretty_company(md: dict, fallback_slug: str = "") -> str:
        company = md.get("company", "")
        if isinstance(company, str) and company.strip():
            return company.strip()
        slug = (md.get("company_slug", "") or fallback_slug).strip()
        return slug.replace("_", " ").title() if slug else "Unknown"

    def pretty_section(section: str) -> str:
        return (section or "source").replace("_", " ")

    # Primary path: FAISS vector DB retrieval
    db, embedder = get_vector_backend()
    if db is not None and embedder is not None:
        try:
            rows = db.search(question, embedder, top_k=max_items * 6)
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
                        f"- {pretty_company(md)} • {pretty_section(md.get('section', 'chunk'))}: {row.get('text', '')}"
                    )
                return lines
        except Exception:
            # fall through to local retriever/fallback
            pass

    # Secondary path: local non-vector retriever
    evidence = rag_retriever.query(question=question, company_filter=company_filter, top_k=max_items)
    if evidence:
        for e in evidence:
            lines.append(
                f"- {e.get('company_slug', 'unknown').replace('_', ' ').title()} • {pretty_section(e.get('source', 'source'))}: {e.get('text', '')}"
            )
        return lines

    # Fallback if RAG index has no chunks
    for item in job_catalog[:max_items]:
        company = item.get("company", "")
        if not company:
            continue
        insight = insights_store.load_by_company(company)
        if insight.get("status") == "missing":
            lines.append(f"- {company}: company profile not generated yet.")
            continue
        blended = normalize_spaces(insight.get("final_blended_insight", ""))
        overview = normalize_spaces(insight.get("company_overview", ""))
        text = blended or overview or "Insight available but summary is empty."
        lines.append(f"- {company}: {text[:220]}")
    return lines


def build_general_answer(question: str, thread: dict) -> str:
    question_tokens = tokenize(question)
    context = thread.get("context", {})
    resume_text = context.get("resume_text", "")
    snippets: list[str] = []

    if resume_text:
        for sentence in resume_text.split("."):
            sent = sentence.strip()
            if sent and (question_tokens & tokenize(sent)):
                snippets.append(sent)
            if len(snippets) >= 3:
                break

    job_catalog = context.get("job_catalog", [])
    shown_jobs_count = int(context.get("shown_jobs_count", 0))

    lines = ["Here is what I found in this thread:"]
    if snippets:
        lines.append("- Resume evidence:")
        for s in snippets:
            lines.append(f"  - {s}")
    else:
        lines.append("- Resume evidence: no direct sentence-level match found for this question.")

    if job_catalog:
        lines.append(f"- Jobs in this thread: {shown_jobs_count} shown of {len(job_catalog)}")
        lines.append("- Default RAG insights:")
        lines.extend(build_rag_insights(job_catalog, max_items=3, question=question))
    else:
        lines.append("- No jobs fetched yet in this thread.")

    return "\n".join(lines)


# Session defaults
if "current_thread_id" not in st.session_state:
    threads = store.list_threads()
    if threads:
        st.session_state["current_thread_id"] = threads[0]["thread_id"]
    else:
        st.session_state["current_thread_id"] = store.create_thread("New conversation")

if "composer_prompt" not in st.session_state:
    st.session_state["composer_prompt"] = ""

st.markdown(
    """
<style>
/* Top bar */
.app-topbar {
    position: sticky;
    top: 0.25rem;
    z-index: 999;
    background: rgba(11, 16, 32, 0.92);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 0.75rem;
    padding: 0.65rem 0.9rem;
    margin: 0.15rem 0 0.8rem 0;
    backdrop-filter: blur(4px);
}
.app-topbar-title {
    font-size: 1.05rem;
    font-weight: 700;
}

/* Composer clean look */
div[data-testid="stForm"] {border: none !important; padding: 0 !important;}

/* Hide drag-drop helper text and keep uploader compact */
div[data-testid="stFileUploaderDropzoneInstructions"] {display: none !important;}
div[data-testid="stFileUploader"] > section {padding: 0 !important; border: none !important;}
div[data-testid="stFileUploaderDropzone"] {
    min-height: 3.2rem !important;
    padding: 0.25rem !important;
    border-radius: 0.75rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
div[data-testid="stFileUploaderDropzone"] button {
    width: 100% !important;
    min-height: 2.7rem !important;
    border-radius: 0.7rem !important;
    font-size: 0.95rem !important;
}

/* Input and send button same height */
div[data-testid="stTextInput"] input {
    min-height: 3.2rem !important;
    border-radius: 0.75rem !important;
}
button[kind="primary"] {
    min-height: 3.2rem !important;
    border-radius: 0.75rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="app-topbar"><div class="app-topbar-title">AI Job Intelligence Assistant</div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Conversations")
    if st.button("New Conversation", use_container_width=True):
        new_id = store.create_thread("New conversation")
        st.session_state["current_thread_id"] = new_id
        st.rerun()
    if st.button("Delete Older Conversations", use_container_width=True):
        current = st.session_state.get("current_thread_id")
        removed = store.delete_all_except(current)
        st.toast(f"Deleted {removed} older conversation(s).")
        st.rerun()

    st.markdown("---")
    all_threads = store.list_threads()
    for t in all_threads:
        tid = t["thread_id"]
        summary = thread_summary(t)
        is_current = tid == st.session_state["current_thread_id"]

        with st.container(border=True):
            st.markdown(f"**{summary}**")
            st.caption(tid)
            if st.button("Open", key=f"open_{tid}", disabled=is_current, use_container_width=True):
                st.session_state["current_thread_id"] = tid
                st.rerun()

thread_id = st.session_state["current_thread_id"]
thread_view = store.get_thread(thread_id)

title_col, action_col = st.columns([6, 2], gap="small")
with title_col:
    st.subheader("Conversation")
    st.caption(f"Thread: {thread_id}")
with action_col:
    st.markdown("&nbsp;")
    if st.button("Delete This Conversation", use_container_width=True):
        deleted = store.delete_thread(thread_id)
        if deleted:
            threads_after = store.list_threads()
            if threads_after:
                st.session_state["current_thread_id"] = threads_after[0]["thread_id"]
            else:
                st.session_state["current_thread_id"] = store.create_thread("New conversation")
        st.rerun()

messages = thread_view.get("messages", [])
if messages:
    for m in messages:
        with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
            st.write(m["content"])
else:
    st.info("No messages yet. Upload resume and/or send a prompt.")

live_response_area = st.container()

context = store.get_thread(thread_id).get("context", {})

st.markdown("---")
with st.container(border=True):
    with st.form("composer_form", clear_on_submit=True):
        up_col, prompt_col, send_col = st.columns([1.6, 6.0, 1.4], gap="small")
        with up_col:
            upload = st.file_uploader(
                "Resume",
                key="composer_upload",
                type=["pdf", "docx", "txt"],
                label_visibility="collapsed",
            )
        with prompt_col:
            prompt = st.text_input(
                "Prompt",
                key="composer_prompt",
                placeholder="Ask a question or upload resume, then press Enter...",
                label_visibility="collapsed",
            ).strip()
        with send_col:
            submitted = st.form_submit_button("➤", type="primary", use_container_width=True)

if submitted:
    if not upload and not prompt:
        st.warning("Add a prompt or upload a resume before sending.")
    else:
        shown_user = prompt if prompt else (f"[resume: {upload.name}] fetch latest jobs" if upload else "")
        if prompt:
            user_content = prompt
            if upload:
                user_content = f"[resume: {upload.name}] {prompt}"
            store.add_message(thread_id, "user", user_content, summarize_prompt(prompt))
        elif upload:
            store.add_message(
                thread_id,
                "user",
                f"[resume: {upload.name}] fetch latest jobs",
                "uploaded resume and requested latest jobs",
            )
        streamed_chunks: list[str] = []

        with live_response_area:
            if shown_user:
                with st.chat_message("user"):
                    st.write(shown_user)
            with st.chat_message("assistant"):
                stream_placeholder = st.empty()

                def on_token(delta: str):
                    streamed_chunks.append(delta)
                    stream_placeholder.markdown("".join(streamed_chunks))

                with st.spinner("AI is thinking..."):
                    result = graph.run(
                        thread_id=thread_id,
                        user_prompt=prompt,
                        upload_name=upload.name if upload else None,
                        upload_bytes=upload.getvalue() if upload else None,
                        on_token=on_token if prompt else None,
                    )

                stream_placeholder.markdown(result.assistant_text)

        store.add_message(thread_id, "assistant", result.assistant_text, "thread response")
        store.update_context(thread_id, {"last_route": result.route})
        st.rerun()
