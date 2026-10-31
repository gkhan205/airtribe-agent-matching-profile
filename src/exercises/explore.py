"""
Exercise 2 - what the Milestone 2 matcher returns, and how a ranking changes when a must-have is added.

Run:  python -m src.exercises.explore   (from the project root)

Step 4 is the idea behind Part B ("re-rank and explain changes"); the agent's
rank_candidates node does the same thing with explain_ranking_changes().
"""

from src.config.constants import JD_DIR
from src.file_system.file_tools import read_file
from src.rag.job_matcher import JobMatcher
from src.rag.resume_rag import ResumeRAG

# 1. load the FAISS index and build the matcher
matcher = JobMatcher(ResumeRAG.load())

# 2. match a job description
jd = read_file(JD_DIR / "it_systems_admin.txt")
result = matcher.match(jd, k=5)
print("Result keys:", list(result.keys()))

# 3. one line per candidate
print("\nRun 1 (JD must-haves only):", result["parsed_requirements"]["must_haves"])
for rank, m in enumerate(result["top_matches"], 1):
    print(f"  #{rank} {m['candidate_name']:18s} {m['match_score']:3d}  missing: {', '.join(m['missing_skills']) or '-'}")

# 4. add a must-have and compare the two rankings
result2 = matcher.match(jd, k=5, must=["Linux"])
print("\nRun 2 (+ Linux):", result2["parsed_requirements"]["must_haves"])
for rank, m in enumerate(result2["top_matches"], 1):
    print(f"  #{rank} {m['candidate_name']:18s} {m['match_score']:3d}")

before = {m["candidate_name"]: rank for rank, m in enumerate(result["top_matches"], 1)}
after = {m["candidate_name"]: rank for rank, m in enumerate(result2["top_matches"], 1)}
print("\nChanges:")
for name, old in before.items():
    if name not in after:
        print(f"  {name} dropped out (was #{old})")
    elif after[name] != old:
        print(f"  {name}: #{old} -> #{after[name]}")
for name, new in after.items():
    if name not in before:
        print(f"  {name} is new at #{new}")
