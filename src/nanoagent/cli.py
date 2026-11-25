import argparse
import re
import time
from urllib.error import URLError
from .agent import NanoAgent
from .models import LLM
from .planning import NativeReAct
from .memory import TrimmingMemory
from .toolbox import create_native_code_tools
from .profiles import AGENT_PROFILES


try:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    Table = None

    class _PlainConsole:
        """Minimal rich.Console stand-in that strips markup."""

        _MARKUP = re.compile(r"\[/?[^\]]*\]")

        def print(self, *args, end="\n"):
            """Print string arguments after stripping Rich-style markup.

            Args:
                *args: Objects to display; only strings are written.
                end: Text appended after each string argument.

            Returns:
                None. Non-string arguments are ignored.
            """
            for arg in args:
                if isinstance(arg, str):
                    print(self._MARKUP.sub("", arg), end=end)
                # Non-strings (Rule, Table) are silently skipped

        @property
        def width(self):
            """Return the current terminal width in columns.

            Inputs:
                None.

            Returns:
                The terminal width reported by ``shutil``.
            """
            import shutil

            return shutil.get_terminal_size().columns

    console = _PlainConsole()


COMMANDS = {
    "/help": "Show available commands",
    "/multiline": (
        "Enter a multiline prompt; finish with /send or cancel with /cancel"
    ),
    "/clear": "Clear conversation memory (keeps system prompt)",
    "/memory": "Show current conversation memory",
    "/model": "Show current model name",
    "/tools": "List available tools",
    "/skills": "List available skills",
    "/agent <name>": "Switch the active agent profile",
}


def _read_multiline_query() -> str | None:
    """Collect a multiline prompt until its submit or cancel marker is entered.

    Inputs:
        None. Reads terminal lines until ``/send``, ``/cancel``, EOF, or
        Ctrl+C; blank lines inside the prompt are retained.

    Returns:
        Joined prompt text with surrounding whitespace stripped, or ``None`` if
        the input is canceled or interrupted.
    """
    console.print(
        "  Paste or type your prompt, then enter /send on its own line "
        "to submit or /cancel to discard it."
    )
    lines = []
    while True:
        try:
            line = input("... ")
        except (KeyboardInterrupt, EOFError):
            console.print("  [dim]Multiline prompt canceled.[/]")
            return None

        command = line.strip().lower()
        if command == "/send":
            return "\n".join(lines).strip()
        if command == "/cancel":
            console.print("  [dim]Multiline prompt canceled.[/]")
            return None
        lines.append(line)


def _chat_completions_url(api_base: str) -> str:
    """Append the chat-completions path when the configured URL omits it.

    Args:
        api_base: Server base URL, ``/v1`` URL, or complete chat endpoint URL.

    Returns:
        A URL ending in ``/v1/chat/completions`` or the supplied complete
        ``/chat/completions`` path.
    """
    url = api_base.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def handle_command(cmd, agent, active_agent: str | None = None):
    """Run a local CLI command and print its result.

    Args:
        cmd: Slash command entered by the user.
        agent: Active agent whose memory, model, and tools may be inspected.
        active_agent: Active profile name shown by the skills command.

    Returns:
        None. Commands may display information or clear conversation memory.
    """
    cmd = cmd.strip().lower()

    # `/help` command lists available commands
    if cmd == "/help":
        if HAS_RICH:
            table = Table(show_header=False, box=None, padding=(0, 2))
            for name, desc in COMMANDS.items():
                table.add_row(f"[bold]{name}[/]", f"[dim]{desc}[/]")
            console.print(table)
        else:
            for name, desc in COMMANDS.items():
                console.print(f"  {name:<12}  {desc}")

    # `/clear` command clears memory except system prompt
    elif cmd == "/clear":
        agent.memory.clear()
        console.print("  [dim]Memory cleared (system prompt kept).[/]")

    # `/memory` command shows current conversation memory
    elif cmd == "/memory":
        messages = agent.memory.get_messages()
        console.print(f"  [dim]{len(messages)} message(s) in memory:[/]")
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = (
                msg["content"] if isinstance(msg["content"], str) else "[multimodal]"
            )
            style = (
                "blue" if role == "user" else "green" if role == "assistant" else "dim"
            )
            console.print(f"     [{style}]{i}: {role}: {content}[/]")

    # `/model` command shows current model name
    elif cmd == "/model":
        console.print(f"  [dim]{agent.llm.model}[/]")

    # `/tools` command lists available tools with descriptions
    elif cmd == "/tools":
        for name, tool in agent.tools.registry.items():
            console.print(f"  [yellow]{name}[/] - [dim]{tool['description']}[/]")

    # `/skills` command lists available skills with descriptions
    elif cmd == "/skills":
        for name, profile in AGENT_PROFILES.items():
            active = " (active)" if name == active_agent else ""
            console.print(f"  [bold]{name}[/]{active} - [dim]{profile.description}[/]")

    # Unknown command
    else:
        console.print(
            f"  [red]Unknown command: {cmd}[/] (type [bold]/help[/] for commands)"
        )


def _progress_message(event, payload, model):
    """Format a progress event as a short, non-sensitive status message.

    Args:
        event: Progress event name emitted by the agent.
        payload: Optional response or tool-call object associated with the event.
        model: Model name shown during generation.

    Returns:
        A readable status string; private model reasoning is never included.
    """
    if event == "thinking":
        return f"Thinking with {model}..."
    if event == "response":
        tool_call = payload.tool_call
        if tool_call:
            name = tool_call.get("function", {}).get("name", "tool")
            return f"Selected tool: {name}"
        return "Preparing final answer..."
    if event == "tool_start":
        name = payload.tool_call.get("tool", "tool")
        return f"Running tool: {name}"
    if event == "tool_result":
        return "Tool finished; continuing..."
    if event == "final":
        return "Final answer ready."
    return "Working..."


def _display_rich_progress(event, payload, model, progress, used_tools):
    """Update Rich's status display and pause it during interactive tool calls.

    Args:
        event: Agent progress event being displayed.
        payload: Optional event response or tool-call object.
        model: Model name used by the progress formatter.
        progress: Active Rich status context with ``stop``, ``start``, and
            ``update`` methods.
        used_tools: Mutable list that receives the name of each started tool.

    Returns:
        None. Writes tool-start notices directly and updates the status otherwise.
    """
    if event == "response" and payload.tool_call:
        name = payload.tool_call.get("function", {}).get("name", "tool")
        console.print(f"  [cyan]Selected tool:[/] {name}")
    elif event == "tool_start":
        name = payload.tool_call.get("tool", "tool")
        used_tools.append(name)
        progress.stop()
        console.print(f"  Running tool: {name}")
        return
    elif event == "tool_result":
        progress.start()

    progress.update(_progress_message(event, payload, model))


def _kv_cache_status(llm):
    """Describe available llama.cpp prompt-context or KV-cache utilization.

    Args:
        llm: Client exposing ``slot_context`` and ``kv_cache_usage_ratio``.

    Returns:
        A human-readable usage string, or an explanatory unavailable message.
        Monitoring transport and malformed-data errors are represented in that
        message rather than raised.
    """
    slot_error = None
    usage_ratio = None
    try:
        slots = llm.slot_context()
    except (OSError, URLError, ValueError) as error:
        slot_error = str(error)
        slots = None

    if slots:
        if all(slot.prompt_tokens is not None for slot in slots):
            used_tokens = sum(slot.prompt_tokens for slot in slots)
            total_context = sum(slot.context_size for slot in slots)
            return f"KV cache context: {used_tokens:,} / {total_context:,} tokens"

    if usage_ratio is None:
        try:
            usage_ratio = llm.kv_cache_usage_ratio()
        except (OSError, URLError, ValueError) as error:
            details = [str(error)]
            if slot_error:
                details.insert(0, f"/slots: {slot_error}")
            return "KV-cache context unavailable: " + "; ".join(details)

    if usage_ratio is not None:
        return f"Current KV cache usage: {usage_ratio:.1%}"

    if slot_error:
        return (
            f"KV-cache context unavailable: /slots: {slot_error}; "
            "/metrics does not expose llamacpp:kv_cache_usage_ratio."
        )
    return (
        "KV-cache context unavailable: /slots does not provide prompt token "
        "counts, and /metrics does not expose llamacpp:kv_cache_usage_ratio."
    )


def _display_run_result(result, used_tools, elapsed_seconds, kv_cache_status):
    """Print an agent answer and its run metrics.

    Args:
        result: Final answer text.
        used_tools: Tool names recorded during the run.
        elapsed_seconds: Wall-clock duration of the request.
        kv_cache_status: Usage or unavailability summary from the monitoring API.

    Returns:
        None. Writes the result and metrics to the configured console.
    """
    tools_summary = ", ".join(used_tools) if used_tools else "none"
    console.print(f"\n  [dim]Tools used:[/] {tools_summary}")
    console.print(f"\n{result}\n")
    console.print(f"  [dim]Elapsed time: {elapsed_seconds:.2f} seconds.[/]\n")
    console.print(f"  [dim]{kv_cache_status}[/]\n")


def _create_agent(agent_name: str, llm: LLM, log_file: str) -> NanoAgent:
    """Construct an agent with the selected profile's tools and instructions.

    Args:
        agent_name: Key in ``AGENT_PROFILES``.
        llm: Shared language-model client.
        log_file: JSONL path used for run logging.

    Returns:
        A configured ``NanoAgent`` with trimming memory and a ten-step native
        planner.
    """
    profile = AGENT_PROFILES[agent_name]
    return NanoAgent(
        llm=llm,
        memory=TrimmingMemory(),
        tools=create_native_code_tools(agent_name),
        planner=NativeReAct(max_steps=10),
        log_file=log_file,
        instructions=profile.instructions,
    )


def main():
    """Parse CLI options and run the interactive NanoAgent session.

    Inputs:
        Command-line arguments for model, endpoint, API key, agent profile, and
        log path.

    Returns:
        None. Runs until the user enters ``exit``/``quit`` or sends EOF.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="LLM model name",
        default="LFM3-1.7B",
    )
    parser.add_argument(
        "-ab",
        "--api_base",
        type=str,
        help="API base URL or chat completions endpoint URL",
        default="http://127.0.0.1:8080/v1/chat/completions",
    )
    parser.add_argument("-ak", "--api_key", type=str, help="API key", default="no_key")
    parser.add_argument(
        "--agent",
        choices=tuple(AGENT_PROFILES),
        default="code_agent",
        help="Agent profile to start with",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Append prompts, model responses, tool results, and final answers to this JSONL file",
        default="nanoagent.jsonl",
    )
    args = parser.parse_args()

    # LLM
    llm = LLM(
        model=args.model,
        base_url=_chat_completions_url(args.api_base),
        api_key=args.api_key,
        think=True,
    )

    agents = {name: _create_agent(name, llm, args.log_file) for name in AGENT_PROFILES}
    active_agent = args.agent
    agent = agents[active_agent]

    # Display welcome message with model, agent, and tools
    tool_names = ", ".join(agent.tools.registry.keys())
    console.print(f"\n  [dim]Model:[/]  {args.model}")
    console.print(f"  [dim]Agent:[/]  {active_agent}")
    console.print(f"  [dim]Tools:[/]  {tool_names}")
    console.print("  [dim]Type[/] /help [dim]for commands.[/]\n")

    while True:
        # User input
        try:
            query = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("  [dim]Goodbye![/]")
            break

        # Skip empty input
        if not query:
            continue

        multiline_input = query.lower() == "/multiline"
        if multiline_input:
            query = _read_multiline_query()
            if query is None or not query:
                continue

        # Exit commands
        if not multiline_input and query.lower() in ("exit", "quit"):
            console.print("  [dim]Goodbye![/]")
            break

        # Slash commands
        if not multiline_input and query.startswith("/"):
            parts = query.split()
            if parts and parts[0].lower() == "/agent":
                if len(parts) == 1:
                    console.print(f"  Active agent: {active_agent}")
                    console.print("  Available: " + ", ".join(AGENT_PROFILES))
                elif len(parts) == 2 and parts[1].lower() in agents:
                    active_agent = parts[1].lower()
                    agent = agents[active_agent]
                    tool_names = ", ".join(agent.tools.registry.keys())
                    console.print(f"  Activated {active_agent}. Tools: {tool_names}")
                else:
                    console.print("  Usage: /agent <" + "|".join(AGENT_PROFILES) + ">")
            else:
                handle_command(query, agent, active_agent)
            console.print()
            continue

        # Run agent
        try:
            used_tools = []
            started_at = time.perf_counter()

            if HAS_RICH:
                with console.status("Thinking...", spinner="dots") as progress:

                    def display_progress(event, payload=None):
                        """Route agent progress to the Rich status renderer."""
                        _display_rich_progress(
                            event, payload, agent.llm.model, progress, used_tools
                        )

                    agent.display = display_progress
                    result = agent.run(query)
            else:

                def display_progress(event, payload=None):
                    """Record tool names and print plain-text progress updates."""
                    if event == "tool_start":
                        used_tools.append(payload.tool_call.get("tool", "tool"))
                    console.print(
                        f"  {_progress_message(event, payload, agent.llm.model)}"
                    )

                agent.display = display_progress
                result = agent.run(query)

            elapsed_seconds = time.perf_counter() - started_at
            kv_cache_status = _kv_cache_status(agent.llm)
            _display_run_result(result, used_tools, elapsed_seconds, kv_cache_status)
        except Exception as e:
            console.print(f"  [red]ERROR[/] {e}\n")


if __name__ == "__main__":
    main()
