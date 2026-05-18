# HW1 — Build a Simple AI Agent (google-genai)

In this homework you will complete a small AI agent built directly on the
[`google-genai`](https://github.com/googleapis/python-genai) SDK. This is the
low-level Gemini SDK — **not** Google ADK. You drive the agent loop yourself,
so you can see exactly how tool calling works under the hood.

## What you'll build

An agent that runs in a loop:

1. Sends the user's prompt to Gemini together with a list of available tools.
2. If the model decides to call a tool, your code executes it locally and
   feeds the result back to the model.
3. Steps 1–2 repeat until the model returns a plain text answer to the user.

The agent exposes **two tools** that you will implement:

| Tool          | Signature                          | What it does                      |
| ------------- | ---------------------------------- | --------------------------------- |
| `get_weather` | `get_weather(city: str) -> str`    | Return weather for a given city.  |
| `add_numbers` | `add_numbers(a: float, b: float)`  | Return the sum of two numbers.    |

## Files

```
hw1-skeleton/
├── tools.py     # Two tool stubs you need to implement
├── agent.py     # Agent loop skeleton + interactive REPL
└── README.md    # This file
```

## Setup

From the `src/` directory:

```bash
uv sync
source .venv/bin/activate
cp env.example .env       # then edit .env and add your GOOGLE_API_KEY
```

You can also set `GEMINI_MODEL` in `.env` (defaults to `gemini-2.5-flash`).

## Your tasks

1. **Implement the two tools** in `tools.py`.
   - `get_weather` can be a simple lookup table for a few cities.
   - `add_numbers` returns `a + b`.

2. **Implement the agent loop** in `agent.py` — see the `TODO` block inside
   `run_agent()`. You will need to:
   - Build a `contents` list starting with the user's message.
   - Build a `GenerateContentConfig` that registers `AVAILABLE_TOOLS` and
     **disables** the SDK's automatic function calling (so you drive the loop).
   - Loop: call `client.models.generate_content(...)`, inspect the response
     parts, execute any `function_call` parts, append the model's content
     and a `function_response` part back into `contents`, and continue until
     the model returns plain text.

3. **Try it out** with the built-in REPL:

   ```bash
   cd src/hw1-skeleton
   python agent.py
   ```

   Suggested prompts to test:
   - `What's the weather in Tokyo?`
   - `What is 17 plus 25?`
   - `Add 3 and 4, then tell me the weather in London.` *(two tool calls)*

## Hints

- The model's response is at `response.candidates[0].content`. Iterate over
  `.parts` and check `part.function_call` and `part.text`.
- To send a tool result back to the model, append a new `types.Content` with
  `role="user"` (or `"tool"` depending on SDK version) containing a
  `types.Part(function_response=types.FunctionResponse(name=..., response={"result": ...}))`.
- Cap your loop with `MAX_AGENT_STEPS` so a buggy run can't loop forever.
- Reference doc: <https://ai.google.dev/gemini-api/docs/function-calling>
