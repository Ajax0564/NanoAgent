# NanoAgent

> **Work in progress:** NanoAgent is an experimental local-model agent. Interfaces,
> tool behavior, and documentation may change as the project develops. 

NanoAgent is a small Python project for exploring how a language model can decide
when to call tools, use their results, and return a final answer. It connects to a
local OpenAI-compatible chat-completions server (such as `llama-server`) and runs
an iterative ReAct-style agent loop.

The CLI shows progress and selected tools while a request is running, then prints
the final answer. It does **not** display the model's private reasoning.

## Current capabilities

- Local model endpoint and model name are configurable from the command line.
- Native function/tool calling with a maximum of 10 agent steps per request.
- Workspace tools: list files, read supported text/spreadsheet files, and run
  Python code after approval.
- Calculator, expression plotting, web search, and date/time tools.
- Conversation memory keeps the system instructions and the latest complete
  turns. Use `/clear` to clear conversation history.
- Time-sensitive requests receive the current date and time in the
  `Asia/Kolkata` timezone. Web-search results include a retrieval time and a
  publication date when the search provider supplies one.
- An unknown tool or structurally invalid tool call can receive one correction
  attempt. Provider failures, timeouts, rate limits, and denied permissions stop
  the current run instead of triggering automatic retries.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A running local server that exposes an OpenAI-compatible
  `/v1/chat/completions` endpoint

To serve the LFM
model with `llama.cpp`, start the server like this:

```powershell
llama-server -m .\LFM3-1.7B-Q4_K_M.gguf --port 8080
```

In another terminal, install the project dependencies and start NanoAgent:

```powershell
uv sync
uv run nanoagent
```

By default, the CLI connects to `http://127.0.0.1:8080/v1/chat/completions` and
sends the model name `LFM3-1.7B`. If your server expects a different model name
or endpoint, configure them with:

```powershell
uv run nanoagent --model <server-model-name> --api_base <chat-completions-url>
```

Use `uv run nanoagent --help` to see all command-line options.

## CLI commands

- `/help` — list commands
- `/clear` — clear conversation history while keeping system instructions
- `/memory` — inspect the current message history
- `/model` — show the configured model name
- `/tools` — list registered tools
- `/skills` — show skill status
- `exit` or `quit` — leave the CLI

## Work in progress and limitations

- **Live information is not guaranteed.** Web search currently uses DDGS search
  results; it is not a dedicated live market-data or sports-score feed. A search
  retrieval time is not proof that the result itself is current.
- **Model generation is blocking.** The CLI can show progress stages, but it
  does not yet stream generated tokens from the server.
- **Tool execution is not a security sandbox.** Python execution requires user
  approval and has a timeout, but it still runs with the local process's
  permissions. Only run code you trust.
- **Agent behavior depends on the selected model and server.** Tool selection,
  argument quality, and response quality can vary.
- **This project is intended for experimentation and learning**, not unattended
  or high-stakes use.