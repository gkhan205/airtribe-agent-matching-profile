"""
test_tools.py - tools and pure logic, no LLM needed (free and fast).

Run:  pytest tests/test_tools.py -v
"""

from src.agent.nodes import explain_ranking_changes, parse_jd, requirements_to_must
from src.agent.tools import build_must_list, get_matcher, list_resumes, rag_search, read_resume, search_resume
from src.config.constants import JD_DIR


def test_list_resumes_returns_100_pdfs():
    result = list_resumes.invoke({"extension": ".pdf"})
    assert result["success"] and result["count"] == 100


def test_read_resume_by_name_and_id():
    rid = next(iter(get_matcher().rag.resumes))
    name = get_matcher().rag.resumes[rid]["name"]
    assert read_resume.invoke({"candidate": rid})["content"]
    assert read_resume.invoke({"candidate": name})["candidate_id"] == rid


def test_read_resume_missing_file_does_not_crash():
    assert read_resume.invoke({"candidate": "nobody_here"})["success"] is False


def test_read_resume_blocks_path_traversal():
    result = read_resume.invoke({"candidate": "../.env"})
    assert result["success"] is False
    assert "ANTHROPIC" not in str(result)


def test_search_resume_finds_keyword():
    rid = next(r for r, m in get_matcher().rag.resumes.items() if "Excel" in m["skills"])
    assert search_resume.invoke({"candidate": rid, "keyword": "excel"})["match_count"] > 0


def test_rag_search_must_filter():
    result = rag_search.invoke({"query": "IT systems administrator", "k": 10, "must": ["Linux"]})
    resumes = get_matcher().rag.resumes
    assert result["candidates"]
    assert all("Linux" in resumes[c["candidate_id"]]["skills"] for c in result["candidates"])


def test_unknown_skill_is_skipped_not_crashing():
    valid, skipped = build_must_list("Title\nSome job", ["Linux", "Kubernetes:2"], [])
    assert valid == ["Linux"] and skipped == ["Kubernetes:2"]


def test_ignored_must_removes_jd_requirement():
    jd = (JD_DIR / "it_systems_admin.txt").read_text()  # its must-haves: 3+ years experience, Networking
    valid, _ = build_must_list(jd, [], ["Networking"])
    assert valid == ["years:3"]


def test_requirements_to_must():
    assert requirements_to_must({"must_have": ["SQL", "3+ years experience"], "min_years": 3}) == ["SQL", "years:3"]


def test_explain_ranking_changes():
    def m(name, score):
        return {"candidate_name": name, "resume_path": f"data/resumes/{name.lower()}.pdf", "match_score": score}
    previous = [m("Ann", 80), m("Bob", 70), m("Cat", 60)]
    current = [m("Bob", 75), m("Ann", 72), m("Dan", 50)]
    changes = " ".join(explain_ranking_changes(previous, current, [{"candidate_name": "Cat", "reasons": ["missing must-have skill Linux"]}]))
    assert "Bob moved up #2 → #1 (score 70 → 75" in changes
    assert "Ann moved down #1 → #2 (score 80 → 72" in changes
    assert "Dan entered" in changes
    assert "Cat (was #3) dropped out: missing must-have skill Linux" in changes


def test_parse_jd_reads_file_by_name():
    state = parse_jd({"job_description": "teacher"})
    assert "Teacher" in state["job_description"]
