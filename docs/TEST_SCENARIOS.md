# Test scenarios: conversation flows

Each scenario is an automated test in `tests/test_scenarios.py` (run with `pytest tests/test_scenarios.py -v`).
The tests check the agent **state** after the conversation, not the exact wording, because LLM text changes between runs.

| # | Conversation | What the agent does | What the test checks |
|---|---|---|---|
| 1 | *"Find me candidates with SQL and 3+ years experience"* | LLM extracts must-haves (SQL, 3 years) and turns them into hard filters for the RAG search | Every shortlisted candidate has SQL and ≥ 3 years of experience |
| 2 | `it_systems_admin` → *"Compare the top 3 matches side by side"* | Classified as a question; Claude calls `compare_candidates` and shows a table plus a verdict | `compare_candidates` was called with exactly the top 3 ids |
| 3 | `it_systems_admin` → *"Why did A rank higher than B?"* | Explains the gap using the score breakdown (semantic / skills / keyword / qualification), skills and experience | The answer names both candidates |
| 4 | `it_systems_admin` → *"Make Linux a must-have"* | Classified as a refinement; re-runs all 3 rounds and lists who moved, entered or dropped out, and why | `Linux` is in `extra_must`, the old shortlist is kept, everyone left has Linux, changes are explained |
| 5 | `it_systems_admin` (multi-round) | Round 1: 100 resumes → top 10. Round 2: LLM analysis of each. Round 3: hire / borderline / no-hire | 100 screened, ≤ 10 shortlisted, analysis + decision for each, borderline ones have suggestions, report file exists |
| 6 | `it_systems_admin` → *"Give me interview questions for the top candidate"* | Claude calls `generate_interview_questions`, which targets the candidate's gaps | The tool was called and returned 5 questions |
| 7 | `it_systems_admin` → *"Let's look at a different role: data/job_descriptions/teacher.txt"* → *"done, thanks"* | Classified as `new_job`: a fresh screening for Teacher, then `done` ends the graph | The JD switched to Teacher, the old shortlist was cleared, the graph reached END |

## Example run (abridged, from a real session)

```
You: it_systems_admin
  📄 Reading the job description
  🧩 Extracting must-have vs nice-to-have requirements (LLM)
  🔎 Round 1: hybrid RAG search over all resumes → 100 resumes, 10 shortlisted, 54 filtered out
  📊 Ranking candidates and comparing with the previous ranking
  🧠 Round 2: deep analysis of the top 10 (LLM, in parallel)
  ⚖️ Round 3: hire / borderline / no-hire decisions (LLM)
  📝 Writing the match report
Agent: IT Systems Administrator: screened 100 resumes → top 10 → 1 hire, 1 borderline, 8 no-hire
  | 1 | Ines Brown | 79 | 6/10 | 🟡 borderline |
  | 2 | Elena Reed | 78 | 7/10 | ✅ hire |
  ...

You: Make Linux a must-have
  🤔 Understanding your request (LLM) → refine (add ['Linux'])
  🔎 Round 1: ... → 100 resumes, 5 shortlisted, 63 filtered out
Agent: What changed:
  - ⬆️ Amara Rahman moved up #3 → #2 (score unchanged at 58; candidates above were filtered out)
  - 🆕 Leila Aziz entered at #5 (score 29)
  - ❌ Elena Reed (was #2) dropped out: missing must-have skill Linux
  ...

You: Now drop the Linux requirement and require 10+ years of experience instead
  🤔 Understanding your request (LLM) → refine (add ['years:10']) (remove ['Linux'])
Agent: What changed:
  - 🆕 Elena Reed entered at #1 (score 78)
  - ❌ Ines Brown (was #1) dropped out: 4.6 yrs experience < 10 required
  ...
```
