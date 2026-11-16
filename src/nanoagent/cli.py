import argparse
import re
from .agent import NanoAgent
from .models import LLM
from .planning import NativeReAct
from .memory import TrimmingMemory
from .toolbox import create_native_code_tools


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
            for arg in args:
                if isinstance(arg, str):
                    print(self._MARKUP.sub("", arg), end=end)
                # Non-strings (Rule, Table) are silently skipped

        @property
        def width(self):
            import shutil

            return shutil.get_terminal_size().columns

    console = _PlainConsole()


COMMANDS = {
    "/help": "Show available commands",
    "/clear": "Clear conversation memory (keeps system prompt)",
    "/memory": "Show current conversation memory",
    "/model": "Show current model name",
    "/tools": "List available tools",
    "/skills": "List available skills",
}


def _chat_completions_url(api_base: str) -> str:
    """Normalize an API base URL to the OpenAI-compatible chat endpoint."""
    url = api_base.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def handle_command(cmd, agent):
    """Handle a slash command."""
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
        console.print("  No skills are currently loaded.")

    # Unknown command
    else:
        console.print(
            f"  [red]Unknown command: {cmd}[/] (type [bold]/help[/] for commands)"
        )


def _progress_message(event, payload, model):
    """Return user-facing progress without exposing model reasoning."""
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


def main():
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
    args = parser.parse_args()

    # LLM
    llm = LLM(
        model=args.model,
        base_url=_chat_completions_url(args.api_base),
        api_key=args.api_key,
        think=True,
    )

    # Tools and Skills
    tools = create_native_code_tools()

    # ReAct planner with native reasoning and tool calling (no text parsing needed)
    planner = NativeReAct(max_steps=10)

    # Coding Agent
    agent = NanoAgent(
        llm=llm,
        memory=TrimmingMemory(),
        tools=tools,
        planner=planner,
    )

    # Display welcome message with model, tools, and skills
    tool_names = ", ".join(agent.tools.registry.keys())
    console.print(f"\n  [dim]Model:[/]  {args.model}")
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

        # Exit commands
        if query.lower() in ("exit", "quit"):
            console.print("  [dim]Goodbye![/]")
            break

        # Slash commands
        if query.startswith("/"):
            handle_command(query, agent)
            console.print()
            continue

        # Run agent
        try:
            used_tools = []

            if HAS_RICH:
                with console.status("Thinking...", spinner="dots") as progress:
                    def display_progress(event, payload=None):
                        if event == "response" and payload.tool_call:
                            name = payload.tool_call.get("function", {}).get(
                                "name", "tool"
                            )
                            console.print(f"  [cyan]Selected tool:[/] {name}")
                        elif event == "tool_start":
                            used_tools.append(payload.tool_call.get("tool", "tool"))
                        progress.update(
                            _progress_message(event, payload, agent.llm.model)
                        )

                    agent.display = display_progress
                    result = agent.run(query)
            else:
                def display_progress(event, payload=None):
                    if event == "tool_start":
                        used_tools.append(payload.tool_call.get("tool", "tool"))
                    console.print(
                        f"  {_progress_message(event, payload, agent.llm.model)}"
                    )

                agent.display = display_progress
                result = agent.run(query)

            tools_summary = ", ".join(used_tools) if used_tools else "none"
            console.print(f"\n  [dim]Tools used:[/] {tools_summary}")
            console.print(f"\n{result}\n")
        except Exception as e:
            console.print(f"  [red]ERROR[/] {e}\n")


if __name__ == "__main__":
    main()
