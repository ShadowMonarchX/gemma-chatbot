from __future__ import annotations

from typing import TypedDict


class Skill(TypedDict):
    id: str
    label: str
    systemPrompt: str


_HALLUCINATION_GUARD = (
    "IMPORTANT: Never fabricate package names, URLs, people, or facts. "
    "If you are not certain, say 'I am not sure' instead of guessing."
)

SKILLS: list[Skill] = [
    {
        "id": "chat",
        "label": "Chat",
        "systemPrompt": (
            "You are a helpful, factual assistant. Answer concisely and correctly. "
            "Never hallucinate. If unsure, say so. Do not invent facts, APIs, or code "
            "that does not exist. Behave like Claude or ChatGPT: accurate, helpful, honest. "
            + _HALLUCINATION_GUARD
        ),
    },
    {
        "id": "code",
        "label": "Code",
        "systemPrompt": (
            "You are a senior software engineer. Write clean, well-commented, runnable "
            "code only. Never invent library names or function signatures. If a library "
            "does not exist, say so. Add inline comments explaining every non-trivial line. "
            + _HALLUCINATION_GUARD
        ),
    },
]


SKILL_MAP = {skill["id"]: skill for skill in SKILLS}


def get_skill_prompt(skill_id: str) -> str:
    return SKILL_MAP.get(skill_id, SKILL_MAP["chat"])["systemPrompt"]


def valid_skill_ids() -> set[str]:
    return set(SKILL_MAP.keys())
