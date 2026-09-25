from google.adk.agents.llm_agent import LlmAgent

from shared.model import get_model

MODEL = get_model()


def _specialist(name: str, kind: str, guidance: str) -> LlmAgent:
    return LlmAgent(
        model=MODEL,
        name=name,
        description=f"Writes {kind} blog posts specifically.",
        instruction=(
            f"You write {kind} blog posts. {guidance}\n"
            "Return the full Markdown document with YAML front matter at the top. "
            "No commentary, no fences wrapping the document."
        ),
        output_key="blog_md",
    )


tutorial_writer = _specialist(
    "tutorial_writer_agent",
    "tutorial",
    "Structure: problem -> setup/prereqs -> numbered steps with code -> verification "
    "-> recap. Use fenced code blocks with language tags. Each step has a one-line "
    "'why' explaining the choice.",
)

opinion_writer = _specialist(
    "opinion_writer_agent",
    "opinion",
    "Structure: bold thesis up front -> argument with evidence -> steelman the "
    "counter -> concession or refutation -> takeaway. First-person voice is fine.",
)

listicle_writer = _specialist(
    "listicle_writer_agent",
    "listicle",
    "Structure: 'N {things} for {audience}' title -> short intro -> N numbered H2 "
    "sections with parallel structure (one-sentence thesis + 2-3 sentence "
    "explanation + concrete example). Keep items roughly equal in length.",
)

case_study_writer = _specialist(
    "case_study_writer_agent",
    "case study",
    "Structure: context -> problem -> approach -> results (with numbers/metrics) -> "
    "lessons learned. Concrete > abstract. Mark invented specifics with [example].",
)

news_writer = _specialist(
    "news_roundup_writer_agent",
    "news roundup",
    "Structure: brief intro -> 4-8 curated items, each a one-line headline + 2-3 "
    "sentence commentary + source link. End with a 'what to watch' note.",
)


ROUTER_INSTRUCTION = """\
You are the blog router. The user describes a blog post they need (topic, audience,
optional post_type). Decide which specialist is best suited based on the post type,
then transfer the request to that specialist. Do NOT write the post yourself.

Routing rules:
- how-to / step-by-step / walkthrough         -> tutorial_writer_agent
- argument / hot take / commentary            -> opinion_writer_agent
- 'N things' / curated list / ranked picks    -> listicle_writer_agent
- real-world example / postmortem / project   -> case_study_writer_agent
- weekly digest / link roundup / news brief   -> news_roundup_writer_agent

If post_type is explicitly given, honor it. Otherwise infer from the topic phrasing.
"""

root_agent = LlmAgent(
    model=MODEL,
    name="blog_router_agent",
    description="Routes a blog request to the right specialist by post type.",
    instruction=ROUTER_INSTRUCTION,
    sub_agents=[
        tutorial_writer,
        opinion_writer,
        listicle_writer,
        case_study_writer,
        news_writer,
    ],
)
