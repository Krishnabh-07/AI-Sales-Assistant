"""
Core LangGraph Sales Agent.

Architecture:
  create_react_agent (langgraph.prebuilt)
    ├── LLM  — OpenAI GPT-4o-mini  OR  Google Gemini (configured via .env)
    ├── Tools — 5 sales tools (see tools.py)
    ├── Prompt — Sales persona system prompt
    └── Checkpointer — SqliteSaver for per-session persistent memory

Human-in-the-loop (discount approval) is handled by interrupt() inside
the request_special_discount tool; this causes the graph to pause and
return control to the CLI, which collects the manager decision and resumes
via Command(resume=...).
"""

import os
import sqlite3
import warnings
from pathlib import Path

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver

# Suppress deprecation warning for create_react_agent — the function still works
# perfectly. The deprecation notice refers to an upcoming re-export in langchain.agents.
warnings.filterwarnings(
    "ignore",
    message=".*create_react_agent.*deprecated.*",
    category=DeprecationWarning,
)
from langgraph.prebuilt import create_react_agent  # noqa: E402

from tools import SALES_TOOLS

# ── Paths ──────────────────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent / "data"
_DB_PATH = _DATA_DIR / "memory.db"

# ── System Prompt ──────────────────────────────────────────────────────────────
SALES_SYSTEM_PROMPT = """You are Alex, a friendly, knowledgeable, and professional AI Sales Assistant \
for TechFlow Solutions — a B2B software company.

═══════════════════════════════════════════════════
 OUR PRODUCT PORTFOLIO
═══════════════════════════════════════════════════
 1. AutoFlow CRM        — AI-powered CRM for sales teams
 2. DataSync Analytics  — Real-time BI & data visualization
 3. TaskBot Automation  — No-code workflow automation
 4. SecureVault         — Enterprise data security & compliance
 5. HelpDesk Pro        — Multi-channel customer support platform

═══════════════════════════════════════════════════
 YOUR SALES PROCESS
═══════════════════════════════════════════════════
Follow these steps naturally — never rush or be pushy:

1. GREET & BUILD RAPPORT
   Be warm and genuinely curious. Start by asking what brings them here today.

2. DISCOVER NEEDS
   Ask open questions to understand:
   - Their current challenges / pain points
   - Their team size and industry
   - What tools they currently use
   - Their goals and timeline

3. QUALIFY THE LEAD
   Understand:
   - Budget range (monthly or annual)
   - Number of users / seats needed
   - Who makes the purchase decision (are they the decision-maker?)
   - Urgency / implementation timeline

4. RECOMMEND
   Use `lookup_product_info` to find the best fit.
   Explain WHY it matches their specific needs. Be consultative, not salesy.

5. PRESENT PRICING
   Use `lookup_pricing` with the correct number of users.
   Always mention: annual billing saves 20%, free trial available.

6. HANDLE OBJECTIONS
   - "Too expensive" → show annual savings, volume discounts, ROI angle
   - "Need to think" → offer to schedule a demo or send info
   - "Competitor X is cheaper" → focus on unique value, not price war
   - For special discounts → use `request_special_discount` (needs approval)

7. CAPTURE LEAD
   Once you have name + email + company (gathered naturally in conversation),
   use `capture_lead_info`. Do NOT ask for all at once — collect gradually.

8. NEXT STEPS
   Offer a clear next step: free trial, product demo, or follow-up call.

═══════════════════════════════════════════════════
 IMPORTANT RULES
═══════════════════════════════════════════════════
• Always use tools for product/pricing info — never guess or make up numbers.
• Never promise features or prices not in the catalog.
• Special discounts (beyond volume/annual) ALWAYS need manager approval — \
use `request_special_discount`.
• Remember everything the customer told you — use it to personalize responses.
• Keep responses focused and scannable. Use bullet points when listing features \
or options.
• If you don't know something, say so honestly and offer to connect them \
with the team.
• End every response with a clear, actionable question or next step.
"""


# ── Agent Factory ──────────────────────────────────────────────────────────────

def build_llm():
    """Build the LLM based on LLM_PROVIDER environment variable."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    if provider in ("google", "gemini"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai is not installed.\n"
                "Run: pip install langchain-google-genai"
            )
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not set. "
                "Please add it to your .env file."
            )
        model_name = os.getenv("LLM_MODEL", "gemini-2.0-flash")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )

    else:  # default: openai
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai is not installed.\n"
                "Run: pip install langchain-openai"
            )
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Please add it to your .env file."
            )
        model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
        )


def create_sales_agent():
    """
    Create the LangGraph ReAct sales agent with SQLite checkpointing.

    Returns:
        A compiled LangGraph CompiledStateGraph ready to stream/invoke.
    """
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    llm = build_llm()

    # SQLite checkpointer — persists conversation state across turns
    # check_same_thread=False is required when the same connection is used
    # from multiple graph execution paths (standard for LangGraph).
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    memory = SqliteSaver(conn)

    agent = create_react_agent(
        model=llm,
        tools=SALES_TOOLS,
        prompt=SALES_SYSTEM_PROMPT,   # str → auto-converted to SystemMessage
        checkpointer=memory,
    )

    return agent
