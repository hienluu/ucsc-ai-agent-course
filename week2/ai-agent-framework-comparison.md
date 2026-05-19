# Multi-Agent Orchestration Frameworks: A Deep Technical Comparison (May 2026)

**Bottom line up front: For a *general* multi-agent orchestration use case in 2026, LangGraph (v1.2.0, ~32.2k GitHub stars, released May 12 2026) is the strongest production-grade default — it is the only framework in this set with a battle-tested durable-execution runtime (Postgres checkpointers, super-step replay), explicit graph control flow, mature streaming, and a real ecosystem of pluggable observability (LangSmith, Langfuse, OTel).** Pick CrewAI if your problem genuinely maps to "a team of roles" and you want to prototype in a day; pick the OpenAI Agents SDK if you are already all-in on OpenAI and prefer a minimalist handoff-based runtime; pick Google ADK if you are on GCP/Gemini and want managed Sessions + Memory Bank; pick Claude Agent SDK only if you want Claude-Code-style filesystem/shell-native agents; pick Pydantic AI for type-safety-first single + simple multi-agent flows with first-class Temporal/DBOS/Prefect durable execution; and avoid Smolagents for production multi-agent — it's a brilliant code-agent library, not an orchestrator.

## TL;DR

- **LangGraph (1.2.0, ~32.2k stars)** is the best general-purpose choice for multi-agent orchestration: explicit typed graph state, durable Postgres checkpointing with per-task pending-writes recovery, native supervisor/swarm/hierarchical/pipeline patterns, five streaming modes plus `astream_events`, and the deepest observability ecosystem. Cost: a real 1–2 week learning curve and ~60+ lines vs. CrewAI's ~20 for a minimal two-agent workflow.
- **CrewAI (~51.4k stars)** wins on time-to-prototype with role/task/crew abstractions and three process modes (Sequential / Hierarchical / Consensual) plus Flows for event-driven control; one independent comparison (Let's Data Science, March 2026) found "*CrewAI gets you from idea to working prototype about 40% faster than LangGraph*." It also has documented footguns: telemetry-on-by-default, ~18% higher token overhead on a 3-agent ticket-triage benchmark (Rapid Claw 2026), no native durable checkpointing, and opaque debugging at 5+ agents.
- **The vendor SDKs are not really competitors to LangGraph/CrewAI** — they are runtimes for *one model family*. OpenAI Agents SDK (handoffs + Sessions), Google ADK (Sequential/Parallel/Loop/LlmAgent + native A2A), and Claude Agent SDK (subagents + Skills + MCP-native) each shine inside their ecosystems but each carries vendor or model-family lock-in. **Pydantic AI** and **Smolagents** sit at the lightweight end — great for narrower jobs (typed agents; code-action agents respectively) but not full multi-agent orchestrators.

## Quick Comparison Matrix

| Framework | Version (May 2026) | GitHub ★ | Orchestration Model | State / Durability | Streaming | Model Agnostic | License | Best For |
|---|---|---|---|---|---|---|---|---|
| **OpenAI Agents SDK** | 0.17.2 (May 12 2026) | 26.3k | Handoffs + agent-as-tool (code-first, imperative) | `Session` (in-mem, SQLAlchemy, Redis, MongoDB, Dapr, OpenAIConversations); no built-in durable replay | Yes (events + tokens, async iterator) | Yes via LiteLLM/Chat-Completions/Responses adapters; OpenAI-optimal | MIT | OpenAI-first apps wanting minimalism |
| **Google ADK (Python)** | 1.33.0 (May 8 2026) | 19.6k | Hierarchical agent tree + workflow agents (Sequential/Parallel/Loop) + LLM transfer | `Session` + `State` services (in-mem, Vertex Agent Engine, Firestore); `MemoryService` (in-mem, VertexAiMemoryBankService, VertexAiRagMemoryService) | Bidirectional audio/video streaming + events | Yes (Gemini-optimal, LiteLlm wrapper for others) | Apache 2.0 | GCP/Gemini, multimodal, A2A-native |
| **Claude Agent SDK** | 0.2.82 (Python, May 15 2026) | 6.9k | Orchestrator + subagents (Agent tool) with isolated context windows | Sessions w/ resume/fork; PreCompact hook; no durable replay; persistence is BYO | Async-iterator stream; tool/thinking blocks | **Claude-only** (Anthropic API, Bedrock, Vertex, Azure Foundry) | Anthropic Commercial Terms | Claude-Code-style FS/shell agents, MCP-native workflows |
| **Pydantic AI** | 1.96.0 (May 13 2026) | ~17.1k | Single agent + delegation-via-tool, programmatic hand-off, optional `pydantic-graph` state machine; durable wrappers (Temporal/DBOS/Prefect) | `WrapperAgent` proxies for durable execution; native graph persistence via pydantic-graph | Token + structured-output streaming with validation | **Most model-agnostic** — OpenAI, Anthropic, Gemini, Bedrock, Ollama, LiteLLM, Cohere, Mistral, etc. | MIT | Type-safe Python apps, durable workflows on Temporal/DBOS |
| **CrewAI** | 1.14.x stable (Apr 30 2026), 1.14.5a5 pre-release (May 12 2026) | 51.4k | Crews (Role/Goal/Backstory agents) + Tasks + Processes (Sequential, Hierarchical, Consensual); Flows for event-driven control | `memory=True` auto-creates `ShortTermMemory` (ChromaDB), `LongTermMemory` (SQLite via `LTMSQLiteStorage`), `EntityMemory` (ChromaDB); orchestrated by `ContextualMemory`; `ExternalMemory` for Mem0 | Yes (added v1.10.1 early 2026) | Yes (LiteLLM under the hood; "Without LiteLLM" path available) | MIT | Fast role-based prototyping, content/research pipelines |
| **LangGraph** | 1.2.0 (May 12 2026, 1.0 GA Oct 2025) | 32.2k | Directed StateGraph with typed state and conditional edges; supervisor / swarm / hierarchical / pipeline patterns | Best-in-class: `MemorySaver`, `SqliteSaver`, `RedisSaver`, `PostgresSaver`; super-step checkpoints, pending-writes recovery, time-travel, thread-scoped state, `Store` for long-term | Five modes: `values`, `updates`, `messages`, `custom`, `debug` + `astream_events`; subgraph streaming | Yes — any LLM (LangChain integrations or custom via `custom` stream mode) | MIT | Production-grade stateful multi-agent systems |
| **Smolagents** | 1.24.0 (Jan 16 2026) | 26.5k | `CodeAgent` (LLM writes Python actions) + `ToolCallingAgent`; `ManagedAgent` for hierarchical multi-agent | Memory implicit in agent steps; **no durable execution, no checkpointing** | `agent.run(..., stream_outputs=True)` | Yes — any LLM via Inference API, Transformers, Ollama, OpenAI/Anthropic via LiteLLM | Apache 2.0 | Code-action agents, GAIA-style benchmark tasks, HF ecosystem |

---

## Per-Framework Deep Dives

### 1. OpenAI Agents SDK (`openai-agents`, 26.3k ★, v0.17.2)

**Core abstractions.** Four primitives: `Agent` (LLM + instructions + tools + handoffs), `Tools` (function tools via `@function_tool`, hosted tools, MCP), `Handoffs` (delegation expressed as a tool call — `transfer_to_<agent>`), and `Guardrails` (input/output validators that run in parallel). Released March 2025 as the production-grade replacement for the experimental `Swarm`. New in 0.14+: **`SandboxAgent`** with `Manifest` and `UnixLocalSandboxClient` for filesystem-backed agents.

**Orchestration patterns.** Two canonical patterns documented by OpenAI itself: (a) **Manager pattern** — central agent invokes other agents as tools via `Agent.as_tool(...)`; conversation control stays with the manager; (b) **Decentralized/handoff pattern** — `handoffs=[agent_a, agent_b]` lets the LLM transfer control completely. The SDK now supports beta **`RunConfig.nest_handoff_history`** which collapses prior transcript into a single assistant summary wrapped in `<CONVERSATION HISTORY>` to avoid bloating context across nested handoffs.

**Concurrency.** Async/await throughout (`Runner.run`, `Runner.run_streamed`, `Runner.run_sync`). Parallelism mostly via the manager pattern: when the manager calls multiple `agent.as_tool` tools in one turn, the SDK runs them concurrently.

**Observability.** Built-in tracing → OpenAI Traces dashboard by default. `add_trace_processor()` / `set_trace_processors()` let you fan-out. Official adapter list: **Weights & Biases, Arize-Phoenix, MLflow, Braintrust, Pydantic Logfire, AgentOps, Scorecard, LangSmith, Maxim AI, Comet Opik, Langfuse, Langtrace, Galileo, Portkey AI, LangDB AI, Agenta, PostHog, PromptLayer, HoneyHive**, etc. OTel path is via `logfire.instrument_openai_agents()` — Langfuse's integration docs note: "*By default, the SDK doesn't emit OpenTelemetry data… Pydantic Logfire SDK has implemented an OpenTelemetry instrumentation wrapper for OpenAI Agents.*" ZDR mode disables tracing entirely.

**Memory & state.** `Session` API: `OpenAIConversationsSession` (server-side via Responses API), `RedisSession`, `SQLAlchemySession`, `MongoDBSession`, `DaprSession` (30+ state-store backends). Plus `OpenAIResponsesCompactionSession` decorator that auto-compacts history via `responses.compact` once 10+ non-user items accumulate. **No durable replay** — if a process dies mid-run, the run is lost; sessions just persist history.

**Streaming.** Native token and event streaming via `Runner.run_streamed()` and async iteration; SSE/websocket transports for OpenAI Responses models. Voice/realtime via `gpt-realtime-2`.

**Model agnosticism.** Despite the name, supports 100+ LLMs. Official paths: `set_default_openai_client` for OpenAI-compatible endpoints; `LitellmModel` from `agents.extensions.models.litellm_model` (`pip install openai-agents[litellm]`); Any-LLM adapter. **Caveat**: hosted tools (`WebSearchTool`, etc.) and full feature parity require OpenAI Responses models; tracing and pre-built OpenAI tools won't work on non-OpenAI providers without custom wiring.

**Cost.** MIT-licensed; no runtime fees. Prompt overhead minimal — the SDK adds little above raw Chat-Completions/Responses calls. Hosted feature surface (Agent Builder, Traces dashboard, Responses) is OpenAI-priced.

**DX.** Python + TypeScript both first-class (parity between `openai-agents-python` and `openai-agents-js`). Zod-based validation in TS, Pydantic in Python. Documentation is unusually clean for a 2025-era SDK.

**Footguns:**
- Default model switched to `gpt-5.4-mini` from `gpt-4.1`; pre-existing agents inherit new defaults including `reasoning.effort="none"`, `verbosity="low"` — silent behavior change.
- Handoffs are 1-way; you cannot "return" without registering a reverse handoff.
- No native durable execution / checkpointing.
- Out-of-the-box OTel is a configuration project, not a checkbox — requires Logfire-wrapper or OpenInference instrumentor.

**Verdict.** Strong fit when (a) you're on OpenAI models anyway, (b) you want the smallest possible abstraction surface, (c) the workload is "triage + specialist routing." Not the best fit for long-running, durable, or graph-structured workflows.

---

### 2. Google Agent Development Kit (ADK, `google-adk` 1.33.0, 19.6k ★ Python; ADK-Java 1.x; TypeScript/Go also shipping)

**Core abstractions.** `BaseAgent` with three subclass families:
- **`LlmAgent`** — Gemini/LLM-driven reasoning, can transfer control to peers
- **Workflow Agents** — deterministic orchestrators: `SequentialAgent`, `ParallelAgent`, `LoopAgent`
- **Custom Agents** — extend `BaseAgent` for arbitrary control flow

Plus `RemoteA2aAgent` for cross-framework agents over the A2A protocol.

**Orchestration patterns.** The 8 patterns covered in Google's own developer blog include Sequential Pipeline, Parallel Fan-Out + Synthesizer, Loop / Critic-Refiner, Hierarchical Delegation, Human-in-the-Loop, Tool-Use, LLM-driven Routing, and Multi-Agent Conversation. The framework's strongest design point is treating workflow orchestration as a first-class agent type rather than as code outside the agent system. Parent-child hierarchies share `InvocationContext` (and `temp:` state); explicit data passing uses session `state[*]`.

**Concurrency.** `ParallelAgent` runs sub-agents simultaneously, sharing session state — the docs explicitly warn: "*Although these agents operate in separate execution threads, they share the session state. To prevent race conditions, make sure each agent writes its data to a unique key.*" `LoopAgent` enables iterative refinement.

**Observability.** Native OpenTelemetry tracing (recent releases added `native OpenTelemetry agentic metrics`), plus `BigQueryAgentAnalyticsPlugin` for production analytics, plus an `adk web` developer UI with step-by-step event/state/agent inspection and a trace view. Native Cloud Trace / Cloud Monitoring on Vertex AI Agent Engine.

**Memory & state.** Three-layer model:
- **Session** = chronological event log (turns + function calls)
- **State** = transient per-conversation data
- **Memory** = personalized cross-session knowledge

Implementations: `InMemorySessionService`, **`VertexAiSessionService`** (managed via Agent Engine), Firestore-backed services. Memory: `InMemoryMemoryService`, **`VertexAiMemoryBankService`** (Gemini-extracted memories from past sessions; the underlying topic-based method was published at ACL 2025), **`VertexAiRagMemoryService`** (vector retrieval from a RAG corpus). The `PreloadMemoryTool` proactively injects memories at turn start; `LoadMemoryTool` lets the agent query memories on demand.

**Streaming.** Distinctive: bidirectional audio/video streaming via Gemini Live API. Events stream natively from `Runner.run_async()`.

**Model agnosticism.** Gemini-optimized, but model-agnostic via `LiteLlm` wrappers (`google.adk.models.lite_llm.LiteLlm`). Supports Anthropic, OpenAI, etc.

**Cost.** Apache 2.0 OSS. Runtime cost is the model spend; if you use the managed Agent Engine, **Vertex AI Agent Engine** pricing applies — Google announced lowered Agent Engine runtime pricing and "*will begin billing for additional Agent Engine services starting on January 28, 2026.*" Memory Bank is in public preview at the time of writing with free-tier quotas.

**DX.** Python is the reference language; Java reached 1.0 in April 2026 (with `LlmAgent.builder()`, A2A SDK integration, ContainerCodeExecutor, ComputerUseTool); TypeScript/Go in beta/preview. Python uses Pydantic for schemas. `adk web` and `adk run` CLIs are unusually polished for an enterprise framework. Bi-weekly release cadence.

**Ecosystem.** Native **A2A protocol** support is genuinely differentiating — your ADK agent can mount a `RemoteA2aAgent` and call a CrewAI or LangGraph agent through a JSON-RPC interface using AgentCards. Native **MCP** support. Strong tool ecosystem: `GoogleSearchTool`, `GoogleMapsTool`, `UrlContextTool`, `ComputerUseTool`, BigQuery analytics.

**Footguns:**
- Tight coupling to Vertex AI / GCP if you adopt the managed services. ADK-only mode works elsewhere but loses Memory Bank, Sessions service, Agent Engine.
- ADK 1.0.0 introduced async-by-default service interfaces — breaking change for code that used sync `BaseSessionService`/`BaseMemoryService`.
- ADK-Java lags ADK-Python: Google's Guillaume Laforge confirmed "*new features, new experiments usually start in Python, and are progressively ported to Java.*"
- `ParallelAgent` race conditions on session state are a real footgun.

**Verdict.** Best-in-class if you are on GCP/Gemini, value managed long-term memory, or need multi-modal (especially streaming audio/video) agents. The A2A native support is also worth a strong look for heterogeneous-agent deployments.

---

### 3. Claude Agent SDK (Anthropic, `claude-agent-sdk` 0.2.82, 6.9k ★ Python repo; npm `@anthropic-ai/claude-agent-sdk`)

**Core abstractions.** This is the runtime that powers Claude Code, exposed as a library. Renamed from "Claude Code SDK" on Sept 29, 2025, to signal broader use beyond coding. Core API: `query(prompt, options)` async iterator + `ClaudeSDKClient` for stateful sessions; `ClaudeAgentOptions` for configuration. Built-in tools: **Read, Write, Edit, Bash, Glob, Grep, WebSearch, WebFetch, AskUserQuestion, NotebookEdit, Agent** (the subagent-spawning tool).

**Orchestration patterns.** Single canonical pattern: **orchestrator + subagents**. Subagents are spawned via the `Agent` built-in tool; each runs in a fresh isolated context window, receives only a prompt string from the parent, returns its final message verbatim. Multiple subagents can run **concurrently**. Subagents cannot recursively spawn their own subagents (do not include `Agent` in their `tools`). Subagents are defined either programmatically via the `agents` parameter, as markdown in `.claude/agents/*.md`, or by using the built-in `general-purpose` agent.

```python
options = ClaudeAgentOptions(
    allowedTools=["Read", "Grep", "Agent"],   # Agent enables subagent spawning
    agents={
      "security-reviewer": AgentDefinition(
        description="...", systemPrompt="...", tools=["Read", "Grep"]),
      "test-runner": AgentDefinition(
        description="...", tools=["Bash"])
    })
```

The case for the orchestrator+subagents model is the strongest one Anthropic has published: per Anthropic Engineering's June 13, 2025, post "How we built our multi-agent research system," "*a multi-agent system with Claude Opus 4 as the lead agent and Claude Sonnet 4 subagents outperformed single-agent Claude Opus 4 by 90.2% on our internal research eval… multi-agent systems use about 15× more tokens than chats.*"

**Concurrency.** Parallel subagents (orchestrator decides scheduling; you define the capability, not the dispatch).

**Observability.** **No built-in observability or durable execution** — explicitly. As Augment Code's deep dive puts it: "*The Anthropic Agent SDK provides no built-in observability, no durable execution, no state persistence across sessions, and no multi-agent coordination beyond spawning subagents as tools.*" You attach observability via OTel hooks or Promptfoo-style integrations; Langfuse lists Claude Agent SDK as a supported framework. `PreCompact` hook fires before context compaction.

**Memory & state.** Sessions can be resumed or forked by `session_id`. **Context compaction is automatic** when the conversation approaches the context window. The September 2025 launch announcement explicitly framed memory and subagent coordination as "solved problems." Long-term cross-session memory is BYO (Anthropic offers a Managed Agents/Memory product, separate, with a "Dreaming" preview feature for memory deduplication).

**Streaming.** Stream-everything by default — text incrementally via `text_stream`, tool blocks, thinking blocks. Supports `eager_input_streaming` for fine-grained per-tool streaming.

**Model agnosticism.** **Claude-only**. Authentication via Anthropic API, Bedrock (`CLAUDE_CODE_USE_BEDROCK=1`), Vertex AI (`CLAUDE_CODE_USE_VERTEX=1`), Azure AI Foundry (`CLAUDE_CODE_USE_FOUNDRY=1`). LiteLLM can route to it but the SDK runtime itself only talks to Claude.

**Cost.** Anthropic Commercial Terms of Service. The CLI bundle (Claude Code) is included in the wheel — each release is ~270–340 MiB per Augment Code's analysis, which affects Docker image budgets. Token cost is the dominant cost; subagent context isolation actually *saves* tokens vs. one big context.

**DX.** Both Python (`pip install claude-agent-sdk`, Py3.10+) and TypeScript (`npm install @anthropic-ai/claude-agent-sdk`, Node 18+) are first-class. The API is small. The agent loop and tool execution are handled inside a bundled CLI binary — meaning the Python process spawns the CLI as a subprocess; this is an unusual architectural choice and per Augment Code, "*communication details between the Python layer and the bundled CLI are not publicly documented.*"

**Footguns:**
- **Model lock-in.** Switching off Claude means rewriting the agent layer.
- No durable execution; no checkpointing; no native multi-agent beyond parent→subagent spawning.
- Bundled CLI binary inflates image size and complicates CI.
- The agent loop runs *outside* your Python process — this is mostly invisible but matters for debugging.
- Subagents can't share state directly (only via the parent's prompt-string channel).
- A documented production case noted that "*running just four agents in production required ClaudeSDKClient with bypassPermissions, Docker containers, Kafka event streaming, Neo4j/Memgraph graph databases, and 15 active MCP servers*" — i.e., the SDK is a thin runtime; you build the platform around it.

**Verdict.** Best when (a) you're Claude-committed, (b) your agents primarily *act on a computer* — filesystem, shell, git, MCP, (c) you want subagent isolation for parallel exploration. Not the best fit for graph-structured workflows, durable long-running tasks, or model-agnostic teams.

---

### 4. Pydantic AI (`pydantic-ai` 1.96.0, ~17.1k ★)

**Core abstractions.** Built by the Pydantic team — whose validation layer is used inside the OpenAI SDK, Google ADK, Anthropic SDK, LangChain, LlamaIndex, CrewAI, and Instructor. Core: `Agent[AgentDepsT, OutputDataT]` (generic over dependency-injection type and output type), `RunContext`, `Tool`/`@agent.tool`/`@agent.tool_plain`, `Toolset`, `FunctionToolset`, `WrapperAgent` for durable-execution proxies. The execution model under the hood is a state machine implemented with `pydantic-graph` — `UserPromptNode` → `ModelRequestNode` → `CallToolsNode` → `End`.

**Orchestration patterns.** Four documented multi-agent patterns:
1. **Single agent** — default.
2. **Agent delegation** — parent calls a delegate inside its own tool function (`r = await delegate_agent.run(prompt, usage=ctx.usage)`); parent regains control on return.
3. **Programmatic hand-off** — application code (not the LLM) decides which agent runs next.
4. **Graph-based control flow** — use `pydantic-graph` to define explicit state machines.

Plus "Deep Agents" — third-party `pydantic-deep` / `pydantic-deepagents` extensions provide planning, file-system, sandbox, and HITL.

**Concurrency.** Tool execution mode is configurable: `parallel` (default, `asyncio.gather`), `sequential`, or `parallel_ordered_events`. Set via `ToolManager.parallel_execution_mode()` context var.

**Observability.** **First-class OpenTelemetry**. Tightly integrated with **Pydantic Logfire** (an OTel-native LLM observability platform from the same team), which is itself the de facto OTel instrumentor for the OpenAI Agents SDK as well. Any OTel-compatible backend (Langfuse, Arize, Datadog, Honeycomb) works out of the box. Pydantic AI's tracing emits standard `gen_ai.*` semantic-convention attributes.

**Memory & state.** No built-in long-term memory primitive. State carried via `deps` (dependency-injected context) and message history. **Durable execution** is the killer feature: `TemporalAgent`, `DBOSAgent`, `PrefectAgent` wrap your `Agent` and proxy model + toolset calls into Temporal activities / DBOS steps / Prefect tasks — meaning you get crash-resumable agents without rewriting your logic.

```python
from pydantic_ai.durable_exec.temporal import TemporalAgent
temporal_agent = TemporalAgent(my_agent)   # now durable
```

**Streaming.** Structured-output streaming with **immediate Pydantic validation** as fields arrive — unique in this set. Plus normal token streaming via `run_stream()` / `iter()`.

**Model agnosticism.** The most model-agnostic of all seven: "*OpenAI, Anthropic, Gemini, DeepSeek, Grok, Cohere, Mistral, Perplexity; Azure AI Foundry, Amazon Bedrock, Google Cloud, Ollama, LiteLLM, Groq, OpenRouter, Together AI, Fireworks…*"

**Cost.** MIT. Pydantic Logfire has a free tier + paid; otherwise self-host any OTel backend.

**DX.** Python only (no TS). Genuinely type-safe — `Agent[DepsT, OutputT]` generics flow through `RunResult[OutputT]`. Test ergonomics borrow from FastAPI. Capability system bundles tools/hooks/instructions/model settings into reusable units; agents can be defined in YAML/JSON. AG-UI and Vercel AI SDK integrations.

**Ecosystem.** ~17.1k stars, v1.96.0 (May 13 2026). ~443+ contributors. Active.

**Footguns:**
- Python only — no TS/JS path.
- Multi-agent story is genuinely thinner than LangGraph or CrewAI. Patterns are documented but you assemble them yourself; there is no "Crew" / "supervisor" prebuilt.
- The Pydantic team's own docs say "*vs. LangGraph: LangGraph excels at complex orchestration… Pydantic AI prioritizes simplicity and type safety for single-agent and simple multi-agent scenarios.*"

**Verdict.** Best when (a) you want Pydantic-grade type safety end-to-end, (b) you need durable execution on Temporal/DBOS/Prefect with minimal code change, (c) your multi-agent pattern is bounded (delegation or programmatic hand-off rather than 20-node graphs).

---

### 5. CrewAI (`crewai` 1.14.x stable; 1.14.5a5 pre-release on May 12 2026; 51.4k ★)

**Core abstractions.** Role-based metaphor: `Agent(role, goal, backstory, tools, ...)` + `Task(description, agent, expected_output, context=[...])` + `Crew(agents=[...], tasks=[...], process=Process.X)`. Plus **CrewAI Flows** — event-driven workflow with `@start()`, `@listen(...)` decorators and Pydantic state classes — the recommended enterprise architecture for production.

**Orchestration patterns.** Three `Process` types:
- **Sequential** — tasks run in list order; each task's output is context for the next.
- **Hierarchical** — auto-generates (or accepts) a Manager Agent that delegates to specialists and validates outputs (`manager_llm` required).
- **Consensual** — agents vote/discuss (less mature).

Plus Flows for granular event-driven control. Plus native MCP + **A2A protocol** support (added in early 2026).

**Memory & state.** When `memory=True` on a Crew, CrewAI auto-instantiates a stack of memory primitives under `crewai.memory`. Per DeepWiki's source-code analysis (deepwiki.com/crewAIInc/crewAI/7.2-memory-configuration-and-storage), "*CrewAI provides four memory types that can be configured at the crew level: short-term memory (STM), long-term memory (LTM), entity memory, and external memory… The memory systems use ChromaDB as the default vector storage backend for persistence.*" Concretely:

| Class | Default backend | Purpose |
|---|---|---|
| `ShortTermMemory` | **ChromaDB** via `RAGStorage` | Current run/session context |
| `LongTermMemory` | **SQLite** via `LTMSQLiteStorage` (default path `./memory/long_term_memory_storage.db`) | Persists across runs |
| `EntityMemory` | **ChromaDB** via `RAGStorage` | Tracks entities (people, places, concepts) |
| `ContextualMemory` | N/A (orchestrator class) | Queries STM/LTM/Entity and assembles context for the agent prompt via `build_context_for_task()` |
| `ExternalMemory` | Pluggable (e.g., **Mem0**) | External provider integration |

Mem0 integrates two ways — via `memory_config={"provider":"mem0", ...}` (replaces short-term + entity memory) or via the newer `ExternalMemory(...)` constructor (see docs.mem0.ai/integrations/crewai). Storage location is platform-app-dirs or `CREWAI_STORAGE_DIR`.

**Concurrency.** Tasks can run async (`async_execution=True`). Flows support parallel listeners. Hierarchical process delegates concurrently when the manager fans out.

**Observability.** **CrewAI Tracing** (free with `crewai login` — integrates with the **AMP / Crew Control Plane**), plus integrations with Langfuse, AgentOps, Arize Phoenix, etc. **Critical production footgun**: "*All CrewAI versions send telemetry data by default*" including agent prompts, task descriptions, execution times. **Set `telemetry=False`** on every `Crew` for production.

**Streaming.** Added in v1.10.1 (early 2026) — token and event streaming, plus A2A/MCP support.

**Model agnosticism.** Yes, via LiteLLM (default) — OpenAI, Anthropic, Bedrock, Gemini, Ollama, LM Studio, anything LiteLLM supports. Also a "Without LiteLLM" path for direct provider clients.

**Cost.** Open-source MIT. CrewAI AMP (Enterprise / Crew Control Plane) is the paid commercial offering with Tracing, deployment, unified control plane, integrations (Gmail, Slack, Salesforce). 100,000+ developers certified via learn.crewai.com.

**DX.** Python only. Pydantic-typed Flow state. ~20 lines of code for a working two-agent crew. Per Let's Data Science (March 2026): "*CrewAI gets you from idea to working prototype about 40% faster than LangGraph, according to benchmark comparisons.*"

**Ecosystem.** 51.4k stars, 7.1k forks — the largest community of any framework in this set. ~5.2M monthly PyPI downloads vs. LangGraph's ~34.5M (stars ≠ production). Processes 450M+ monthly workflows per company claims.

**Footguns:**
- **Telemetry on by default** ships agent prompts and task descriptions to CrewAI servers unless you set `telemetry=False`. Documented in independent bug reports.
- **Memory growth**: independent SJSU benchmarks measured memory usage exceeding **2 GB** for crews with 10+ agents running 50+ tasks. Recommend task batching + periodic cleanup.
- Hierarchical process is brittle if agent roles aren't precisely defined — manager misassigns tasks.
- **No native durable checkpointing** — long-running tasks risk state loss on pod crash; LangGraph is the stronger choice for resilient long-running workflows.
- Independent benchmarks: JetThoughts 2025 reports CrewAI executes ~**5.76× faster** than LangGraph on a simple QA workflow, but Pooya.blog's 2026 complex-task benchmark observes the opposite at high complexity: "*LangGraph completes 62% because its graph state machine handles failed nodes gracefully. CrewAI manages 54%.*"
- **Token overhead**: Rapid Claw's *AI Agent Framework Scorecard 2026* observes that "*CrewAI optimizes for builder velocity… You pay for that with 18% token overhead and more memory*" on a 3-agent ticket-triage benchmark.

**Verdict.** The best framework if your team thinks in human-team metaphors and your workflow is bounded (research → write → review). Don't pick it for fintech-level audit trails, mission-critical durable execution, or workflows where you need to debug exactly which agent saw what state.

---

### 6. LangGraph (`langgraph` 1.2.0, 32.2k ★; 1.0 GA October 2025)

**Core abstractions.** `StateGraph` parameterized by a typed `State` (TypedDict, Pydantic model, or dataclass), with `Nodes` (functions / agents) and `Edges` (deterministic or conditional). Compiled graphs become Pregel-style executables. Plus `create_agent` (the new LangChain 1.0 unified high-level builder, replacing `create_react_agent`), `langgraph-supervisor`, `langgraph-swarm`, and `langchain-mcp` adapters.

**Orchestration patterns.** Four canonical multi-agent patterns, all supported natively:
1. **Supervisor** — central node routes via `create_supervisor()` or tool-calling (now the LangChain-recommended pattern over the legacy supervisor library).
2. **Swarm** — peer agents handing off via `create_handoff_tool` + `create_swarm`; preserves last-active-agent in shared state.
3. **Hierarchical** — supervisor of supervisors.
4. **Pipeline** — sequential edge chain.

Plus deep agents (LangChain's `deepagents`) on top.

**Concurrency.** Native: parallel nodes run in the same super-step; LangGraph uses reducer logic to merge concurrent state updates. Subgraphs run in their own namespace and can stream separately.

**Observability.** **Best in class**. Native LangSmith tracing (no instrumentation needed if `LANGCHAIN_TRACING_V2=true`). OTel via the LangSmith bridge; native integrations with Langfuse, Arize, Datadog, etc. LangSmith provides per-node spans, time-travel debugging, replay, and dataset-driven evals.

**Memory & state.** Best-in-class durable execution. The state machine creates a **checkpoint at every super-step boundary**, plus per-task pending writes for failure recovery. Backends: `InMemorySaver`, `SqliteSaver`, `RedisSaver`, **`PostgresSaver`** (production standard, queryable history, debuggable). Plus a separate `Store` abstraction for **cross-thread / long-term memory** (`InMemoryStore`, vector-backed stores). **Time-travel debugging** via `get_state_history(config)`. Threads identified by `thread_id` in `config["configurable"]`. DBOS integration adds a second layer of durable execution for tool calls.

**Streaming.** Five modes — `values` (full state after each step), `updates` (deltas — recommended for dashboards), `messages` (LLM tokens + metadata), `custom` (user-emitted events via `StreamWriter`), `debug` (firehose). Plus `astream_events(version="v2")` for fine-grained event streaming. Subgraph streaming with namespace identifiers. `version="v3"` is in alpha (1.2 series).

**Model agnosticism.** Any LLM. LangChain integrations cover all major providers; custom streaming mode supports non-LangChain LLMs.

**Cost.** MIT. LangGraph Cloud (managed hosting with built-in checkpointing, tracing, autoscale) is the commercial path; otherwise self-host.

**DX.** **Python and TypeScript** both first-class (`langgraph` Python and `@langgraph/sdk` / `langgraph-js`). Real learning curve: independent estimates ~1–2 weeks before a team is productive. ~60+ lines for a basic multi-agent setup vs. CrewAI's ~20. The payoff is explicit control.

**Ecosystem.** 32.2k stars, 5.5k forks, ~34.5M monthly PyPI downloads — the production leader by download volume. Battle-tested at Uber, LinkedIn, Klarna, Replit, and Ally Financial since pre-1.0. Used inside LangChain 1.0 itself as the runtime for `create_agent`. Native langchain-mcp adapters for MCP. No native A2A — relies on community integrations.

**Footguns:**
- Manual `PostgresSaver` setup requires `autocommit=True` and `row_factory=dict_row` on `psycopg.connect` — otherwise checkpointing fails silently with `TypeError`.
- **State bloat** — storing large blobs (PDFs, images) in checkpointed state causes DB bloat across every step. Best practice: store URIs, not bytes.
- Supervisor pattern adds an LLM call per routing decision — latency + cost.
- The graph paradigm "clicks for some developers immediately and confuses others for weeks."
- The legacy `langgraph-supervisor` library is now deprecated in favor of in-tool supervisor; check release notes when copying old patterns.

**Verdict.** **The default recommendation for production multi-agent orchestration in 2026.** Pick it if you need durability, explicit control flow, audit trails, time-travel debugging, or production-grade streaming. The cost is genuinely a 1–2 week learning curve.

---

### 7. Smolagents (Hugging Face, `smolagents` 1.24.0, 26.5k ★)

**Core abstractions.** `CodeAgent` (writes its actions as Python code snippets executed in a sandbox) and `ToolCallingAgent` (classic JSON tool-calling). `ManagedAgent(agent, name, description)` wraps an agent so a manager can call it as a tool. `Tool` (callable), `InferenceClientModel` / `LiteLLMModel` / `TransformersModel` for LLM backends. Core library deliberately fits in ~1,000 lines.

**Orchestration patterns.** **Hierarchical only** (manager → managed agents). Documented multi-agent example: `CodeAgent` manager containing a `ToolCallingAgent` web-search subagent. There is no native supervisor library, no swarm, no graph, no checkpointing, no durable execution.

**Concurrency.** Limited; the CodeAgent can use parallel Python directly inside generated code, but framework-level parallel agent orchestration is not a first-class primitive.

**Observability.** **OpenTelemetry-native**. `pip install 'smolagents[telemetry]' openinference-instrumentation-smolagents` and any OTel backend works — Langfuse, Arize, Phoenix. From the HF docs: "*Using instrumentation to record agent runs is necessary in production. We've adopted the OpenTelemetry standard.*"

**Memory & state.** Agent steps are stored in `agent_memory` and accessible (`agent_memory.get_succinct_steps()`); no persistence layer, no checkpointing, no cross-session memory primitives. Memory is "for the current run."

**Streaming.** `agent.run(..., stream_outputs=True)` streams intermediate steps.

**Model agnosticism.** Most flexible HF-ecosystem-wise. `InferenceClientModel` (HF Inference Providers — Cerebras, Together, Fireworks, SambaNova, etc.), `TransformersModel` (local), `OllamaModel`, `LiteLLMModel` (anything LiteLLM speaks: OpenAI, Anthropic, Bedrock, Azure). Modality-agnostic: text, vision, video, audio.

**Cost.** Apache 2.0. Free.

**DX.** Python only. CodeAgent advantage is real: per Hugging Face's official blog "Our Transformers Code Agent beats the GAIA benchmark," "*We get 44.2% on the validation set: so that means Transformers Agent's ReactCodeAgent is now #1 overall, with 4 points above the second! On the test set, we get 33.3%, so we rank #2, in front of Microsoft Autogen's submission.*" On HF's internal benchmark, CodeAgent used ~30% fewer steps than ToolCallingAgent across multiple models.

**Footguns:**
- **No durable execution, no checkpointing, no production state management.** State lives in-memory for the run.
- **No native multi-agent orchestration primitives beyond `ManagedAgent`.**
- **Sandbox is not optional in production.** The built-in `LocalPythonExecutor` is "*explicitly not a security boundary*" — for production code-execution you must use E2B, Modal, Blaxel, Docker, or Pyodide+Deno WebAssembly.
- Last release: **Jan 16, 2026 (v1.24.0)** — release cadence has slowed vs. competitors that ship monthly or bi-weekly.
- Observability is OTel-only; no first-party tracing UI.

**Verdict.** Excellent for single-agent **code-action** tasks (research, web automation, data analysis, GAIA-style benchmarks). Not the right choice for production multi-agent orchestration — the abstractions are intentionally minimal.

---

## Cross-Cutting Comparisons

### A. Multi-Agent Orchestration Models

| Pattern | LangGraph | CrewAI | OpenAI Agents | ADK | Claude SDK | Pydantic AI | Smolagents |
|---|---|---|---|---|---|---|---|
| Supervisor / Manager | ✅ (`create_supervisor`, tool-calling) | ✅ (Hierarchical process w/ Manager agent) | ✅ (Manager via `agent.as_tool`) | ✅ (LlmAgent + sub_agents) | ✅ (orchestrator + Agent tool) | ⚠️ DIY via delegation | ✅ (`ManagedAgent`) |
| Swarm / Peer Handoff | ✅ (`create_swarm`, `create_handoff_tool`) | ⚠️ Consensual process | ✅ (handoffs) | ✅ (LlmAgent transfer) | ❌ subagents one-way | ⚠️ Programmatic hand-off | ❌ |
| Sequential pipeline | ✅ (edge chain) | ✅ (Sequential process) | ⚠️ DIY | ✅ (`SequentialAgent`) | ⚠️ DIY | ⚠️ DIY | ⚠️ DIY |
| Parallel fan-out | ✅ (parallel nodes, reducer) | ✅ (async_execution) | ⚠️ DIY via as_tool | ✅ (`ParallelAgent`) | ✅ (parallel subagents) | ✅ (`parallel`/`parallel_ordered_events`) | ⚠️ in-code |
| Loop / refine | ✅ (cycles) | ⚠️ DIY | ⚠️ DIY | ✅ (`LoopAgent`) | ⚠️ DIY | ✅ (graph cycles via pydantic-graph) | ⚠️ |
| Graph-structured | ✅ **first-class** | ❌ | ❌ | ⚠️ Custom agent | ❌ | ✅ (`pydantic-graph`) | ❌ |
| HITL / approval | ✅ (interrupt, resume) | ✅ (Flows HITL) | ✅ (built-in HITL) | ✅ (Tool Confirmation) | ✅ (canUseTool, permission modes) | ✅ (DeferredToolRequests / Approval) | ⚠️ DIY |
| Cross-framework A2A | Community | ✅ native | Community | ✅ **native** | ⚠️ via MCP | ✅ native | ❌ |

### B. Observability

- **Native OTel emission**: Pydantic AI (via Logfire), ADK (native `gen_ai.*` metrics), Smolagents (OpenInference instrumentor)
- **Wrapper-required OTel**: OpenAI Agents SDK (Logfire-wrapper), Claude Agent SDK (provider-level), CrewAI (third-party integrations)
- **First-party hosted UI**: LangSmith (LangChain), Pydantic Logfire, CrewAI AMP, OpenAI Traces dashboard, ADK Web UI
- **Universal OTel backend** (Langfuse, Arize, Phoenix, Datadog, Honeycomb) works with all seven, but only Pydantic AI, ADK, LangGraph, and Smolagents emit clean `gen_ai.*` semantic conventions out of the box.

### C. Memory & State

- **Durable checkpointing**: **LangGraph** (Postgres / Redis / SQLite super-step snapshots + per-task writes; time-travel; thread-scoped) — only framework with this as a first-class runtime primitive. **Pydantic AI** is a close second via Temporal/DBOS/Prefect wrappers (durability at the workflow-engine layer rather than the agent layer).
- **Session/conversation persistence**: OpenAI Agents SDK (Redis, SQLAlchemy, MongoDB, Dapr, OpenAI Conversations), ADK (Vertex AI Agent Engine Sessions, Firestore), CrewAI (memory-backed automatically), LangGraph (built-in).
- **Long-term semantic memory**: **ADK Vertex AI Memory Bank** (ACL 2025 paper method, Gemini-extracted), **CrewAI** (LongTermMemory SQLite + EntityMemory ChromaDB + Mem0 via ExternalMemory), **LangGraph** (`Store` abstraction + vector backends).
- **No native memory layer**: Claude Agent SDK, Smolagents (BYO).

### D. Streaming

| | Token streaming | Event streaming | Structured-output streaming | Multi-agent streaming |
|---|---|---|---|---|
| OpenAI SDK | ✅ | ✅ | via response_format | per-agent events |
| ADK | ✅ + audio/video | ✅ | ✅ | ✅ |
| Claude SDK | ✅ (text_stream) | ✅ (blocks) | ✅ | per-subagent |
| Pydantic AI | ✅ | ✅ | ✅ **with validation** | ✅ |
| CrewAI | ✅ (v1.10.1+) | ✅ | ✅ | per-task |
| LangGraph | ✅ | ✅ **5 modes + astream_events v2** | ✅ | ✅ **with subgraph namespaces** |
| Smolagents | ✅ | step-level | n/a | per-managed-agent |

### E. Model Agnosticism (most → least)

1. **Pydantic AI** — first-class native support for OpenAI, Anthropic, Gemini, DeepSeek, Grok, Cohere, Mistral, Perplexity, Azure AI Foundry, Bedrock, GCP, Ollama, LiteLLM, Groq, OpenRouter, Together, Fireworks
2. **LangGraph** — any LangChain integration plus custom-mode for non-LangChain LLMs
3. **CrewAI** — LiteLLM-based + direct paths
4. **Smolagents** — InferenceClient + Transformers + Ollama + LiteLLM
5. **OpenAI Agents SDK** — OpenAI-optimal; 100+ LLMs via LiteLLM extension but some features (hosted tools, tracing dashboard) degrade
6. **Google ADK** — Gemini-optimal; LiteLlm wrapper for others
7. **Claude Agent SDK** — **Claude-only**

### F. Cost & Licensing

All seven are open-source: LangGraph, LangChain, OpenAI Agents SDK, Pydantic AI, CrewAI **MIT**; ADK, Smolagents **Apache 2.0**; Claude Agent SDK is open source but governed by **Anthropic Commercial Terms** when used to power your products. Hosted/managed offerings: LangGraph Cloud, CrewAI AMP, Vertex AI Agent Engine, OpenAI Platform, Pydantic Logfire — pricing varies and shifts; ADK announced reduced Agent Engine runtime pricing effective Jan 28, 2026. Token overhead order (low → high, per published 2026 benchmarks): Smolagents (CodeAgent uses ~30% fewer steps) ≤ OpenAI SDK ≤ LangGraph ≤ CrewAI (~18% higher per Rapid Claw 2026 Scorecard) ≤ ADK (workflow agents add scaffolding).

### G. Ecosystem (May 2026 snapshot)

| Framework | GitHub ★ | Latest stable | Last release | Backing org | TS/JS |
|---|---|---|---|---|---|
| CrewAI | 51.4k | 1.14.x | Apr 30, 2026 (alpha 1.14.5a5 May 12) | CrewAI Inc | ❌ |
| LangGraph | 32.2k | 1.2.0 | May 12, 2026 | LangChain | ✅ |
| Smolagents | 26.5k | 1.24.0 | Jan 16, 2026 | Hugging Face | ❌ |
| OpenAI Agents SDK (Python) | 26.3k | 0.17.2 | May 12, 2026 | OpenAI | ✅ (separate repo) |
| ADK Python | 19.6k | 1.33.0 | May 8, 2026 | Google | ✅ (TS in beta) |
| Pydantic AI | ~17.1k | 1.96.0 | May 13, 2026 | Pydantic Services | ❌ |
| Claude Agent SDK (Py) | 6.9k | 0.2.82 | May 15, 2026 | Anthropic | ✅ |

---

## Recommendation Matrix

| Your use case | Recommendation | Why |
|---|---|---|
| **General-purpose multi-agent orchestration (your stated use case)** | **LangGraph** | Durable checkpointing, explicit graph, supervisor + swarm both native, best observability, language-agnostic, large production track record |
| Fast prototype, role-based workflow, content/research pipeline | CrewAI | ~20 LOC for a working crew; role-based mental model is fastest to communicate; Flows for production |
| OpenAI-native chat product with triage + handoffs | OpenAI Agents SDK | Minimalist API, Python + TS parity, Sessions, built-in tracing |
| GCP/Gemini stack, multimodal, voice/video | Google ADK | Gemini Live streaming, Memory Bank, Vertex AI Agent Engine, native A2A |
| Claude-Code-style agents (FS, shell, MCP-heavy) | Claude Agent SDK | Native FS/shell/Bash/Grep tools, parallel subagents, deepest MCP integration |
| Production agent with Temporal/DBOS durable execution + type safety | Pydantic AI | Only framework with first-class Temporal + DBOS + Prefect agent wrappers; structured-output streaming with validation |
| Single code-action agent / GAIA-style benchmark task | Smolagents | CodeAgent paradigm yields ~30% fewer steps and HF's submission ranked #1 GAIA validation / #2 test |
| Heterogeneous agents from multiple frameworks must interop | ADK or CrewAI (A2A native); LangGraph + langchain-mcp; OpenAI Agents (community A2A) | Pick A2A-native frameworks if you anticipate cross-framework calls |
| Regulated / fintech / healthcare with audit trail requirements | LangGraph | Time-travel + thread-scoped checkpoints + LangSmith spans give a paper trail compliance teams can audit |

---

## Final Recommendation for Multi-Agent Orchestration

**Adopt LangGraph as the primary orchestration framework.** Concretely:

1. **Start with LangGraph 1.2 + LangChain 1.0's `create_agent`** for individual ReAct-style worker agents. Use the manual supervisor pattern via tool-calling (per LangChain's own current guidance) rather than the deprecated `langgraph-supervisor` library.
2. **Use `PostgresSaver`** with a connection-pooled `psycopg` (`autocommit=True`, `row_factory=dict_row`) from day one. Do *not* ship to production on `MemorySaver`. Thread-ID format: `tenant-{id}:user-{id}:session-{id}`.
3. **Pair with LangSmith** for tracing if you're comfortable with a SaaS, or **Langfuse self-hosted** if you need OSS/data-residency control — Langfuse ingests LangGraph spans natively via OpenTelemetry.
4. **Use `langchain-mcp` adapters** so your tool layer is portable to any other framework if you ever need to migrate.
5. **For pure code-execution sub-tasks**, consider a Smolagents `CodeAgent` *inside* a LangGraph node — this gives you the GAIA-grade code-action efficiency without giving up orchestration durability.

**If your team is non-technical or your timeline is days-not-weeks**, start with **CrewAI Flows** (not raw Crews), explicitly set `telemetry=False`, and budget for a migration to LangGraph once you hit conditional-branching limits or need durable execution. The migration is a rewrite (CrewAI's role-based model doesn't 1:1 map to LangGraph nodes), not a refactor — so the "prototype-then-migrate" path costs real engineering weeks. Plan accordingly.

**Avoid:**
- Picking the OpenAI Agents SDK or Claude Agent SDK if you may need to switch model vendors within 18 months.
- Picking Smolagents as your top-level orchestrator — it is a code-agent library, not an orchestration platform.
- Building production-critical long-running agents on any framework here *without* a durable-execution story (LangGraph PostgresSaver, or Pydantic AI + Temporal/DBOS, or external orchestration via DBOS/Temporal/Prefect wrapping whatever you choose).

**Benchmarks / thresholds that would change this recommendation:**
- If your team is small (1–3 engineers), workflow is bounded (≤4 agents, no branching beyond a fixed pipeline), and time-to-market < 2 weeks → CrewAI Flows beats LangGraph on developer velocity.
- If you are >80% committed to OpenAI for the next 18 months and don't need durable replay → OpenAI Agents SDK is a simpler choice.
- If multi-modal (audio/video) is the primary capability → Google ADK with Gemini Live is unmatched.
- If model lock-in is acceptable in exchange for OS-level capability (filesystem, shell, MCP) → Claude Agent SDK.

---

## Caveats & Limitations of This Report

- **Frameworks evolve fast.** All version numbers and star counts are May 2026 snapshots; Claude Agent SDK in particular has been a moving API (V2 session-based interface added in 0.2.x alongside the generator pattern).
- **Benchmarks are situational.** JetThoughts's "5.76× faster" CrewAI vs. LangGraph figure (simple QA) and Pooya.blog's complex-task figure (LangGraph 62% vs. CrewAI 54% success) point in opposite directions because they measure different things. Treat headline numbers as suggestive — benchmark with your own workload.
- **"Multi-agent" is doing heavy lifting.** Anthropic Engineering's published research (Jun 13, 2025) measured a multi-agent system with Claude Opus 4 lead + Sonnet 4 subagents outperforming single-agent Opus 4 by 90.2% on their internal research eval, *but* "*multi-agent systems use about 15× more tokens than chats.*" Resist multi-agent for jobs a single-agent loop can handle.
- **Vendor lock-in is real.** ADK + Vertex AI Agent Engine, OpenAI Agents SDK + Responses API hosted tools, and Claude Agent SDK + Anthropic models are all designed to be *better* inside their respective ecosystems. If you adopt the managed offerings, factor switching cost into your decision.
- **No framework solves orchestration without you also solving** observability, evaluation, prompt management, cost monitoring, and secrets management. Plan to pair whichever framework you pick with Langfuse / LangSmith / Pydantic Logfire / Arize, plus an evals harness (Pydantic Evals, Promptfoo, LangSmith Evaluations, DeepEval).
- **The Claude Agent SDK CLI-binary architecture** (the agent loop runs in a bundled CLI subprocess rather than in your Python process) is not deeply documented and may surprise teams during incident response or low-level debugging.
- **CrewAI telemetry-on-by-default** is a real production-data footgun; set `telemetry=False` everywhere or upgrade to AMP where data handling is governed by enterprise terms.