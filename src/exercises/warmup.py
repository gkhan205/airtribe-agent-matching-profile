"""
Exercise 3 - LangGraph by hand, in plain Python. No LLM needed.

Run:  python -m src.exercises.warmup   (from the project root)

This does manually what LangGraph automates: pass one state dictionary through
a series of functions and merge what each one returns.
"""

from typing import TypedDict

SKILLS = ["python", "react", "sql", "aws"]

RESUMES = {
    "Alice": "python react developer with aws",
    "Bob": "java and sql engineer",
    "Chen": "python sql aws data engineer",
}


class ScreeningState(TypedDict):
    job_description: str
    must_have: list[str]
    shortlist: list[str]


def extract_skills(state: ScreeningState) -> dict:
    """Return {"must_have": [...]} with every skill from SKILLS found in the job description."""
    text = state["job_description"].lower()
    return {"must_have": [skill for skill in SKILLS if skill in text]}


def shortlist(state: ScreeningState) -> dict:
    """Return {"shortlist": [...]} with the names of people who have ALL must-have skills."""
    names = [name for name, resume in RESUMES.items() if all(skill in resume for skill in state["must_have"])]
    return {"shortlist": names}


if __name__ == "__main__":
    state: ScreeningState = {"job_description": "We need a Python developer who knows AWS", "must_have": [], "shortlist": []}
    for node in (extract_skills, shortlist):   # LangGraph's edges do this loop for you
        state.update(node(state))
        print(f"after {node.__name__}: {state}")
