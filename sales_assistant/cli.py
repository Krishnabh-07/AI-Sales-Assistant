#!/usr/bin/env python3
"""
AI Sales Assistant — Command-Line Interface
==========================================

Usage:
    python cli.py

Commands during chat:
    /quit   — Exit
    /new    — Start a fresh conversation session
    /leads  — Show all captured leads
    /help   — Show help

Environment:
    Copy .env.example → .env and fill in your API key before running.
"""

import json
import os
import sys
import uuid
from pathlib import Path

# ── Load .env before importing agent modules ──────────────────────────────────
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
    except ImportError:
        # dotenv not installed — env vars must be set manually
        pass
else:
    # Warn only if neither .env nor OS-level vars seem to be set
    pass

# ── Now import LangGraph / agent ──────────────────────────────────────────────
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.types import Command

from agent import create_sales_agent

# ── Constants ─────────────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent / "data"
_LEADS_FILE = _DATA_DIR / "leads.json"

BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║      🤖  TechFlow AI Sales Assistant  🤖                ║
║              Powered by LangGraph                        ║
╠══════════════════════════════════════════════════════════╣
║  Type your message and press Enter to chat.              ║
║  Special commands: /quit  /new  /leads  /help            ║
╚══════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
┌─ COMMANDS ──────────────────────────────────────────────┐
│  /quit   — Exit the assistant                           │
│  /new    — Start a brand-new conversation session       │
│  /leads  — View all leads captured this session         │
│  /help   — Show this help message                       │
└─────────────────────────────────────────────────────────┘

The AI Sales Assistant can:
  • Answer questions about TechFlow Solutions products
  • Provide detailed pricing for any product & team size
  • Recommend the right solution for your needs
  • Collect your contact details (lead qualification)
  • Request special discounts — requires manager approval

Human-in-the-Loop: If you ask for a special discount, the
assistant will pause and ask YOU (as the manager) to approve
or decline before responding to the customer.
"""


# ── Utilities ─────────────────────────────────────────────────────────────────

def print_divider(char: str = "─", width: int = 60) -> None:
    print(char * width)


def print_leads() -> None:
    """Display all captured leads from the leads.json file."""
    if not _LEADS_FILE.exists():
        print("  No leads captured yet.")
        return
    with open(_LEADS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    leads = data.get("leads", [])
    if not leads:
        print("  No leads captured yet.")
        return
    print(f"\n  📋 Captured Leads ({len(leads)} total):\n")
    for lead in leads:
        print(f"  [{lead['id']}]  {lead['name']} <{lead['email']}>")
        print(f"          Company : {lead['company']}")
        print(f"          Interest: {lead['interest']}")
        print(f"          Status  : {lead['status']}")
        print(f"          At      : {lead['captured_at'][:19]}")
        print()


# ── Interrupt Handler ─────────────────────────────────────────────────────────

def handle_discount_interrupt(interrupt_data: dict) -> Command:
    """
    Handle the human-in-the-loop pause for special discount approval.

    The graph is paused inside request_special_discount() via interrupt().
    We collect the manager's decision here and return a Command(resume=...)
    to continue graph execution.
    """
    print(interrupt_data.get("display_message", "\n[APPROVAL REQUIRED]"))
    print()

    while True:
        choice = input("  Manager — Approve this discount? [y/n]: ").strip().lower()
        if choice in ("y", "yes"):
            # Allow manager to adjust the discount percentage
            req_pct = interrupt_data.get("requested_discount_percent", 0)
            try:
                raw = input(
                    f"  Approved discount % [{req_pct}%] "
                    "(press Enter to keep requested): "
                ).strip()
                approved_pct = float(raw) if raw else req_pct
            except ValueError:
                approved_pct = req_pct

            note = input(
                "  Optional message for the customer (Enter to skip): "
            ).strip()

            return Command(resume={
                "approved": True,
                "approved_percent": approved_pct,
                "manager_note": note,
            })

        elif choice in ("n", "no"):
            reason = input(
                "  Reason for declining "
                "(Enter for default): "
            ).strip()
            return Command(resume={
                "approved": False,
                "reason": reason or "Request does not meet our current discount policy.",
            })

        else:
            print("  Please enter 'y' (approve) or 'n' (decline).")


def handle_generic_interrupt(interrupt_data: dict) -> Command:
    """Fallback for any non-discount interrupt types."""
    print("\n⚠️  The assistant requires your input to continue.")
    if "display_message" in interrupt_data:
        print(interrupt_data["display_message"])
    user_input = input("\n  Your response: ").strip()
    return Command(resume={"response": user_input})


# ── Agent Streaming ───────────────────────────────────────────────────────────

def stream_agent_response(
    agent, input_data: dict | Command, config: dict
) -> dict | None:
    """
    Stream a single agent turn and print tokens as they arrive.

    Returns the interrupt_data dict if an interrupt was triggered, else None.
    After this function returns, the caller should check for interrupts and
    call `resume_after_interrupt` if needed.
    """
    print("\n🤖 Alex: ", end="", flush=True)

    seen_content = ""
    interrupted = False

    try:
        for chunk, _metadata in agent.stream(
            input_data,
            config=config,
            stream_mode="messages",
        ):
            # Print AI message tokens as they stream in
            if isinstance(chunk, (AIMessage, AIMessageChunk)) and chunk.content:
                new_text = chunk.content[len(seen_content):]
                if new_text:
                    print(new_text, end="", flush=True)
                    seen_content = chunk.content

    except Exception as exc:
        # Unexpected error during streaming — surface it clearly
        print(f"\n\n[Streaming error: {exc}]")

    print()  # newline after streamed response

    # Check whether an interrupt was triggered during this turn
    snapshot = agent.get_state(config)
    for task in snapshot.tasks:
        for intr in task.interrupts:
            interrupted = True
            return intr.value  # return the interrupt payload

    return None  # no interrupt


def resume_after_interrupt(
    agent, interrupt_data: dict, config: dict
) -> dict | None:
    """
    Handle the interrupt, then stream the resumed response.

    Returns another interrupt_data if a second interrupt occurs, else None.
    """
    interrupt_type = interrupt_data.get("type", "")

    print_divider("═")
    if interrupt_type == "discount_approval_request":
        resume_command = handle_discount_interrupt(interrupt_data)
    else:
        resume_command = handle_generic_interrupt(interrupt_data)
    print_divider("═")

    # Stream the resumed response
    return stream_agent_response(agent, resume_command, config)


# ── Main CLI Loop ─────────────────────────────────────────────────────────────

def new_session() -> tuple[str, dict]:
    """Create a new session ID and LangGraph config."""
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    return session_id, config


def check_env() -> None:
    """Verify that the required API key is present."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider in ("google", "gemini"):
        if not os.getenv("GOOGLE_API_KEY"):
            print("❌  ERROR: GOOGLE_API_KEY is not set.")
            print("   Copy .env.example → .env and add your API key.")
            sys.exit(1)
    else:
        if not os.getenv("OPENAI_API_KEY"):
            print("❌  ERROR: OPENAI_API_KEY is not set.")
            print("   Copy .env.example → .env and add your API key.")
            sys.exit(1)


def main() -> None:
    print(BANNER)
    check_env()

    print("  🔄 Initializing Sales Assistant...", end="", flush=True)
    try:
        agent = create_sales_agent()
        print("  ✅ Ready!\n")
    except Exception as exc:
        print(f"\n❌  Failed to initialize agent:\n  {exc}")
        sys.exit(1)

    session_id, config = new_session()
    print(f"  📋 Session: {session_id}")
    print_divider()
    print("  Say hello to start chatting! (type /help for commands)\n")

    # ── Kick off with an automatic greeting ──────────────────────────────────
    greeting_input = {"messages": [HumanMessage(content="Hello!")]}
    interrupt_data = stream_agent_response(agent, greeting_input, config)
    if interrupt_data:
        interrupt_data = resume_after_interrupt(agent, interrupt_data, config)

    # ── Main conversation loop ────────────────────────────────────────────────
    while True:
        print()
        try:
            user_text = input("👤 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye! 👋\n")
            break

        if not user_text:
            continue

        # ── Slash commands ────────────────────────────────────────────────────
        if user_text.startswith("/"):
            cmd = user_text.lower().strip()

            if cmd == "/quit":
                print("\n  Goodbye! 👋\n")
                break

            elif cmd == "/new":
                session_id, config = new_session()
                print(f"\n  ✨ New session started. ID: {session_id}")
                print_divider()
                greeting_input = {"messages": [HumanMessage(content="Hello!")]}
                interrupt_data = stream_agent_response(agent, greeting_input, config)
                if interrupt_data:
                    interrupt_data = resume_after_interrupt(agent, interrupt_data, config)

            elif cmd == "/leads":
                print()
                print_divider()
                print_leads()
                print_divider()

            elif cmd == "/help":
                print(HELP_TEXT)

            else:
                print(f"  Unknown command: '{user_text}'. Type /help for available commands.")

            continue

        # ── Normal user message ───────────────────────────────────────────────
        message_input = {"messages": [HumanMessage(content=user_text)]}
        interrupt_data = stream_agent_response(agent, message_input, config)

        # Handle any interrupt that was triggered (e.g. discount approval)
        while interrupt_data is not None:
            interrupt_data = resume_after_interrupt(agent, interrupt_data, config)


if __name__ == "__main__":
    main()
