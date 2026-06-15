from google.adk.agents import SequentialAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools import google_search

from shared.model import get_model, get_gemini_model
from shared.types import BlogMetadata, Outline

MODEL = get_model()

research_agent = LlmAgent(
    model=get_gemini_model(),
    name="research_agent",
    description="Gathers facts, sources, and possible angles for the blog topic.",
    instruction="""\
You are a researcher. From the user's BlogRequest (topic, audience, post_type),
compile a ResearchBundle.

Return JSON with:
- facts: list of {claim, source} (6-10 specific claims; use well-known sources;
  if uncertain, mark source as "[example]").
- sources: deduped list of source URLs/citations as strings.
- angles: 3-5 possible angles or themes the post could take.
""",
    output_key="research",
    tools=[google_search]
)

outline_agent = LlmAgent(
    model=MODEL,
    name="outline_agent",
    description="Plans the post outline from the topic + research bundle.",
    instruction="""\
Research bundle in session state under `research`:
{research}

Pick ONE angle and produce an Outline:
- title (working title)
- hook (1-2 sentences)
- thesis (the main argument or takeaway)
- sections: 3-6 SectionSpec entries (heading, key_points [3-5], word_budget).
""",
    output_schema=Outline,
    output_key="outline",
)

draft_agent = LlmAgent(
    model=MODEL,
    name="draft_agent",
    description="Drafts the full post from the outline + research bundle.",
    instruction="""\
Outline (session state `outline`):
{outline}

Research (session state `research`):
{research}

Write the full draft in Markdown:
- H1 from outline.title.
- Hook paragraph.
- One H2 per outline.sections[i].heading, written to its word_budget.
- A conclusion paragraph that returns to the thesis.

Cite facts inline as bracketed links or footnotes. Use concrete examples.
Return raw Markdown only.
""",
    output_key="draft_md",
)

edit_agent = LlmAgent(
    model=MODEL,
    name="edit_agent",
    description="Editorial pass: voice, flow, paragraph length, clarity.",
    instruction="""\
Draft in session state under `draft_md`:
{draft_md}

Editorial pass:
- Tighten openings; cut filler phrases ("in conclusion", "needless to say").
- Break paragraphs longer than ~4 sentences.
- Vary sentence length; replace passive voice where it hurts clarity.
- Keep voice consistent throughout.

Return the revised Markdown. Do not change the meaning.
""",
    output_key="edited_md",
)

seo_agent = LlmAgent(
    model=MODEL,
    name="seo_agent",
    description="Derives SEO metadata from the edited draft (metadata only, no body).",
    instruction="""\
Edited draft in session state under `edited_md`:
{edited_md}

Derive ONLY metadata. Do NOT include the body. Return a BlogMetadata JSON:
- title: refined for SEO if helpful, otherwise the H1.
- slug: kebab-case, <=60 chars.
- meta_description: <=160 chars, includes primary keyword naturally.
- tags: 3-6 topical tags.
- reading_time_min: round(word_count / 220).
- citations: deduped citation URLs found in the draft.
""",
    output_schema=BlogMetadata,
    output_key="metadata",
)

publisher_agent = LlmAgent(
    model=MODEL,
    name="publisher_agent",
    description="Assembles edited body + metadata into a publish-ready Markdown file.",
    instruction="""\
Assemble the final publish-ready Markdown file from two pieces.

Edited body (session state `edited_md`):
{edited_md}

Metadata (session state `metadata`):
{metadata}

Output format:
1. YAML front matter at the very top, between `---` delimiters, with keys in this order:
   title, slug, meta_description, tags, reading_time_min, citations.
   - Strings in double quotes when they contain colons or special chars.
   - tags as a flow-style YAML list, e.g. [agents, adk, tutorials].
   - citations as a flow-style YAML list; use [] if none.
2. ONE blank line.
3. The edited body VERBATIM. Do NOT rewrite, summarize, or re-format the body.

Return ONLY the assembled Markdown. No commentary, no code fences wrapping the document.
""",
    output_key="published_md",
)

root_agent = SequentialAgent(
    name="sequential_blog_pipeline",
    description=(
        "Research -> Outline -> Draft -> Edit -> SEO -> Publisher pipeline "
        "for blog generation."
    ),
    sub_agents=[
        research_agent,
        outline_agent,
        draft_agent,
        edit_agent,
        seo_agent,
        publisher_agent,
    ],
)
