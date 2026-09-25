from google.adk.agents.llm_agent import LlmAgent

from shared.model import get_model

MODEL = get_model()

INSTRUCTION = """\
You are a blog-writing agent. Given a topic (and optional audience, post_type, tone),
produce ONE complete blog post in Markdown.

Output requirements:
- Return raw Markdown only. No code fences wrapping the whole document, no commentary
  before or after.
- Begin with YAML front matter (between --- lines) containing:
    title, slug, meta_description (<=160 chars), tags (3-6), reading_time_min.
- Body, in order:
    * H1 title (matches the front-matter title).
    * A 1-2 sentence hook that earns the reader's attention.
    * 3-6 H2 sections with substantive prose (not just bullet lists).
    * A short conclusion or takeaway under "## Conclusion" or similar.
- Use concrete examples, not generic claims.
- Inline code with backticks; multi-line code in fenced blocks with a language tag.
- Cite external claims as bracketed links or footnotes. Mark invented sources
  with "[example]".
"""

root_agent = LlmAgent(
    model=MODEL,
    name="single_blog_agent",
    description="Generates one complete Markdown blog post from a topic in a single LLM call.",
    instruction=INSTRUCTION,
    output_key="blog_md",
)
