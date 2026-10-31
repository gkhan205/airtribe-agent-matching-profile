"""
tools.py - Part A.2: Tools available to the agent

The @tool decorator turns a normal Python function into something Claude can call.
Claude decides WHEN to call a tool by reading:
  - the docstring  -> what the tool does
  - the type hints -> what arguments it takes

Some tools take `state: Annotated[dict, InjectedState]`. LangGraph fills that argument
with the current agent state when the tool runs, and Claude never sees it. That is how
compare_candidates knows the current job and shortlist without Claude re-sending them.

Tools return plain data and never crash: they return {"success": False, "error": "..."} instead.

Try a tool on its own:
    from src.agent.tools import rag_search
    rag_search.invoke({"query": "python linux", "k": 3})
"""

import json
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from src.agent.llm import get_structured_llm
from src.agent.schemas import CandidateComparison, InterviewQuestions, JobRequirements
from src.config.constants import REPORTS_DIR, RESUME_CHARS_FOR_LLM, RESUMES_DIR, ROOT
from src.file_system import fs_tools
from src.rag.job_matcher import JobMatcher, parse_cli_must, parse_job_description
from src.rag.resume_rag import ResumeRAG

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_matcher = None


def get_matcher() -> JobMatcher:
    """Return a JobMatcher, loading the FAISS index the first time only."""
    global _matcher
    if _matcher is None:
        _matcher = JobMatcher(ResumeRAG.load())
    return _matcher


def candidate_id(match: dict) -> str:
    """'data/resumes/chen_wang.pdf' -> 'chen_wang'."""
    return Path(match["resume_path"]).stem


def resolve_candidate(name_or_id: str) -> str | None:
    """Turn 'Chen Wang', 'chen_wang' or 'chen_wang.pdf' into the candidate id, or None if unknown."""
    resumes = get_matcher().rag.resumes
    key = Path(name_or_id.strip()).stem.lower().replace(" ", "_").replace("-", "_")
    if key in resumes:
        return key
    for rid, meta in resumes.items():
        if meta["name"].lower() == name_or_id.strip().lower():
            return rid
    return None


def read_resume_text(rid: str) -> str:
    """Full text of a candidate's resume (cut to RESUME_CHARS_FOR_LLM characters)."""
    path = ROOT / get_matcher().rag.resumes[rid]["resume_path"]
    result = fs_tools.read_file(str(path))
    return result["content"][:RESUME_CHARS_FOR_LLM] if result["success"] else ""


def _must_key(spec: str) -> str:
    """The part of a must-have that identifies it: 'Python:3' -> 'python', 'years:5' -> 'years'."""
    return spec.partition(":")[0].strip().lower()


def _requirement_to_spec(req) -> str:
    """Turn a Requirement found in the JD back into the 'Skill:years' text format."""
    if req.kind == "experience":
        return f"years:{req.years:g}"
    if req.kind == "skill_years":
        return f"{req.skill}:{req.years:g}"
    return req.skill


def build_must_list(jd: str, extra_must: list[str], ignored_must: list[str]) -> tuple[list[str], list[str]]:
    """Combine the JD's own must-haves with the user's changes.

    Returns (valid must-haves, skipped ones). A must-have is skipped when its skill is
    not in the taxonomy (parse_cli_must would stop the program) - the search still uses
    it semantically, it just is not a hard filter.
    """
    ignored = {_must_key(m) for m in ignored_must}
    jd_musts = [_requirement_to_spec(r) for r in parse_job_description(jd).must_haves if r.kind != "degree"]
    valid, skipped, seen = [], [], set()
    for spec in jd_musts + list(extra_must):
        key = _must_key(spec)
        if key in ignored or spec.lower() in seen:
            continue
        seen.add(spec.lower())
        try:
            parse_cli_must(spec)
            valid.append(spec)
        except (SystemExit, ValueError):
            skipped.append(spec)
    return valid, skipped


def run_match(jd: str, k: int, extra_must: list[str] | None = None, ignored_must: list[str] | None = None) -> dict:
    """Run the Milestone 2 matcher with the combined must-have list."""
    must, skipped = build_must_list(jd, extra_must or [], ignored_must or [])
    result = get_matcher().match(jd, k=k, must=must, use_jd_must=False)
    result["skipped_must"] = skipped
    return result


def extract_job_requirements(jd: str) -> dict:
    """Ask the LLM for must-have vs nice-to-have requirements (structured output)."""
    structured_llm = get_structured_llm(JobRequirements)
    reqs = structured_llm.invoke(
        "Extract the hiring requirements from this job description or recruiter request. "
        "Only list what is actually asked for.\n\n" + jd
    )
    return reqs.model_dump()


def candidate_profile(rid: str, state: dict) -> dict:
    """Everything the agent knows about one candidate: match result, round 2 analysis, round 3 decision."""
    meta = get_matcher().rag.resumes[rid]
    shortlist = state.get("shortlist") or []
    match = next((m for m in shortlist if candidate_id(m) == rid), None)
    profile = {
        "candidate_id": rid,
        "name": meta["name"],
        "headline": meta["headline"],
        "experience_years": meta["experience_years"],
        "education": meta["education"]["degree"],
        "skills": meta["skills"],
    }
    if match:
        profile.update(
            rank=shortlist.index(match) + 1,
            match_score=match["match_score"],
            score_breakdown=match["score_breakdown"],
            matched_skills=match["matched_skills"],
            missing_skills=match["missing_skills"],
            reasoning=match["reasoning"],
        )
    else:
        profile["note"] = "Not in the current shortlist"
    if rid in (state.get("analyses") or {}):
        profile["analysis"] = state["analyses"][rid]
    if rid in (state.get("decisions") or {}):
        profile["decision"] = state["decisions"][rid]
    return profile


def _job_summary(state: dict) -> str:
    reqs = state.get("requirements") or {}
    return json.dumps({"title": reqs.get("title"), "must_have": reqs.get("must_have"),
                       "nice_to_have": reqs.get("nice_to_have"), "min_years": reqs.get("min_years"),
                       "extra_must_added_by_user": state.get("extra_must", [])})


# ---------------------------------------------------------------------------
# File system tools (from Milestone 1)
# ---------------------------------------------------------------------------

@tool
def list_resumes(extension: str = "") -> dict:
    """List the resume files available, optionally filtered by extension like '.pdf'."""
    files = fs_tools.list_files(str(RESUMES_DIR), extension or None)
    return {"success": True, "count": len(files), "files": [f["name"] for f in files]}


@tool
def read_resume(candidate: str) -> dict:
    """Read the full text of one candidate's resume. `candidate` is a name or id, e.g. 'Chen Wang' or 'chen_wang'."""
    rid = resolve_candidate(Path(candidate).name)  # Path(...).name blocks '../../.env' style paths
    if not rid:
        return {"success": False, "error": f"Unknown candidate: {candidate}"}
    return {"success": True, "candidate_id": rid, "content": read_resume_text(rid)}


@tool
def search_resume(candidate: str, keyword: str) -> dict:
    """Find the lines in one candidate's resume that mention a keyword (case-insensitive)."""
    rid = resolve_candidate(Path(candidate).name)
    if not rid:
        return {"success": False, "error": f"Unknown candidate: {candidate}"}
    result = fs_tools.search_in_file(str(ROOT / get_matcher().rag.resumes[rid]["resume_path"]), keyword)
    if result["success"]:
        result["file"] = Path(result["file"]).name
        result["matches"] = result["matches"][:10]
    return result


@tool
def save_report(filename: str, content: str) -> dict:
    """Save a text or markdown report into the reports folder."""
    name = Path(filename).name or "report.md"
    result = fs_tools.write_file(str(REPORTS_DIR / name), content)
    if result["success"]:
        result["path"] = str(Path(result["path"]).relative_to(ROOT))
    return result


# ---------------------------------------------------------------------------
# RAG search tool (from Milestone 2)
# ---------------------------------------------------------------------------

@tool
def rag_search(query: str, k: int = 10, must: list[str] | None = None) -> dict:
    """Search ALL resumes for a job description or skills query and return the top k ranked candidates.

    `must` adds hard requirements in 'Skill' or 'Skill:years' format, e.g. ['Linux', 'Python:3'],
    or 'years:N' for overall experience.
    """
    result = run_match(query, k=k, extra_must=must or [])
    return {
        "success": True,
        "must_haves_applied": result["parsed_requirements"]["must_haves"],
        "skipped_unknown_skills": result["skipped_must"],
        "candidates": [
            {"candidate_id": candidate_id(m), "name": m["candidate_name"], "match_score": m["match_score"],
             "matched_skills": m["matched_skills"], "missing_skills": m["missing_skills"], "reasoning": m["reasoning"]}
            for m in result["top_matches"]
        ],
        "filtered_out": result["filtered_out"][:10],
    }


# ---------------------------------------------------------------------------
# New tools for Milestone 3
# ---------------------------------------------------------------------------

@tool
def extract_requirements(jd: str) -> dict:
    """Parse a job description into must-have vs nice-to-have requirements."""
    return extract_job_requirements(jd)


@tool
def compare_candidates(candidate_ids: list[str], state: Annotated[dict, InjectedState]) -> dict:
    """Compare 2-5 candidates head-to-head for the current job. Ids or names both work, e.g. ['chen_wang', 'Zara Ali']."""
    rids = [resolve_candidate(c) for c in candidate_ids]
    unknown = [c for c, r in zip(candidate_ids, rids) if r is None]
    if unknown:
        return {"success": False, "error": f"Unknown candidates: {unknown}"}
    profiles = [candidate_profile(r, state) for r in rids]

    structured_llm = get_structured_llm(CandidateComparison)
    comparison = structured_llm.invoke(
        "You are a hiring manager. Compare these candidates for the job below. Be specific and fair.\n\n"
        f"JOB:\n{_job_summary(state)}\n\nCANDIDATES:\n{json.dumps(profiles, indent=1)}"
    )
    table = [
        {"candidate": p["name"], "rank": p.get("rank", "-"), "match_score": p.get("match_score", "-"),
         "experience_years": p["experience_years"], "education": p["education"],
         "matched_skills": ", ".join(p.get("matched_skills", [])) or "-",
         "missing_skills": ", ".join(p.get("missing_skills", [])) or "-",
         "decision": p.get("decision", {}).get("decision", "-")}
        for p in profiles
    ]
    return {"success": True, "table": table, **comparison.model_dump()}


@tool
def generate_interview_questions(candidate_id: str, state: Annotated[dict, InjectedState]) -> dict:
    """Create 5 screening interview questions for one candidate, targeting their gaps for the current job."""
    rid = resolve_candidate(candidate_id)
    if not rid:
        return {"success": False, "error": f"Unknown candidate: {candidate_id}"}
    profile = candidate_profile(rid, state)

    structured_llm = get_structured_llm(InterviewQuestions)
    result = structured_llm.invoke(
        "Write 5 screening interview questions for this candidate. Make them specific to THIS person: "
        "verify their claimed strengths and probe their gaps for the job.\n\n"
        f"JOB:\n{_job_summary(state)}\n\nCANDIDATE:\n{json.dumps(profile, indent=1)}\n\n"
        f"RESUME:\n{read_resume_text(rid)}"
    )
    return {"success": True, "candidate": profile["name"], "questions": result.questions}


# The tools the chat agent can call when answering questions (Part B).
ALL_TOOLS = [
    list_resumes, read_resume, search_resume, save_report,
    rag_search, extract_requirements, compare_candidates, generate_interview_questions,
]
