from typing import Literal

from pydantic import BaseModel, Field

PostType = Literal["tutorial", "opinion", "listicle", "case_study", "news"]
Tone = Literal["technical", "conversational", "formal", "playful"]


class BlogRequest(BaseModel):
    topic: str
    audience: str
    post_type: PostType = "tutorial"
    target_words: int | None = None
    tone: Tone | None = None


class SectionSpec(BaseModel):
    heading: str
    key_points: list[str] = Field(default_factory=list)
    word_budget: int = 200


class Outline(BaseModel):
    title: str
    hook: str
    thesis: str
    sections: list[SectionSpec]


class Section(BaseModel):
    heading: str
    body_md: str
    citations: list[str] = Field(default_factory=list)


class Draft(BaseModel):
    title: str
    hook: str
    sections: list[Section]
    conclusion: str


class BlogMetadata(BaseModel):
    title: str
    slug: str
    meta_description: str
    tags: list[str] = Field(default_factory=list)
    reading_time_min: int = 0
    citations: list[str] = Field(default_factory=list)
