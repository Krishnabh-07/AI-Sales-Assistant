# 🤖 AI Sales Assistant

A conversational AI sales agent built with **LangGraph**, capable of understanding customer
needs, recommending products, presenting pricing, qualifying leads, and escalating special
discount requests to a human manager for approval — all from your terminal.

---

## What It Does

| Capability | How |
|---|---|
| Understands customer needs | Consultative questioning via LLM |
| Recommends products | `lookup_product_info` tool → product catalog |
| Presents pricing | `lookup_pricing` tool → pricing catalog |
| Qualifies leads | Conversation-driven BANT qualification |
| Captures leads | `capture_lead_info` tool → `data/leads.json` |
| Remembers conversation | SQLite checkpointing (per session) |
| Special discounts | `request_special_discount` → **Human-in-the-loop** approval |
| CLI interface | `cli.py` — run from terminal, token streaming |

---

## Project Structure

```
sales_assistant/
├── agent.py            ← LangGraph ReAct agent + LLM setup + SQLite memory
├── tools.py            ← 5 sales tools (product, pricing, leads, discount, availability)
├── cli.py              ← Terminal interface with streaming + interrupt handling
├── __init__.py
│
├── data/
│   ├── products.json   ← Product catalog (5 products)
│   ├── pricing.json    ← Pricing tiers + discount rules
│   ├── leads.json      ← Captured leads (auto-created on first capture)
│   └── memory.db       ← SQLite conversation memory (auto-created at runtime)
│
├── requirements.txt    ← Python dependencies
├── .env.example        ← Environment variable template
└── README.md           ← This file
```

> **Note:** This application lives inside the cloned LangGraph repository but is completely
> independent of it. It does **not** modify any LangGraph source files under `libs/` or `examples/`.

---

## Quickstart

### 1. Navigate to this folder

```bash
cd "C:\Users\krishnabh talukdar\Documents\Resume Projects\Real world project\AI related\AI-Sales-Assistant\sales_assistant"
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you plan to use **Google Gemini** instead of OpenAI:

```bash
pip install langchain-google-genai
```

### 4. Configure your API key

```bash
# Copy the template
copy .env.example .env     # Windows
cp .env.example .env       # macOS/Linux

# Open .env in any text editor and fill in your API key
```

Your `.env` should look like this (choose ONE provider):

```env
# For OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...your-real-key-here...
LLM_MODEL=gpt-4o-mini

# OR for Google Gemini
# LLM_PROVIDER=google
# GOOGLE_API_KEY=your-real-key-here
# LLM_MODEL=gemini-2.0-flash
```

> ⚠️ **Never commit your `.env` file to GitHub.** It's already in `.gitignore`.

### 5. Run the assistant

```bash
python cli.py
```

---

## Example Conversation

```
╔══════════════════════════════════════════════════════════╗
║      🤖  TechFlow AI Sales Assistant  🤖                ║
║              Powered by LangGraph                        ║
╚══════════════════════════════════════════════════════════╝

🤖 Alex: Hi there! I'm Alex, your TechFlow Solutions assistant.
         What brings you here today — are you looking to solve a
         specific business challenge?

👤 You: We're a 25-person SaaS startup struggling with customer support

🤖 Alex: That's a common growing pain for SaaS companies! A few quick
         questions — are your support requests coming in through multiple
         channels (email, chat, etc.), or mainly one channel?
         ...

👤 You: Can I get 30% off?

🤖 Alex: [Calling request_special_discount tool...]

═══════════════════════════════════════════════════════════
  🔔  MANAGER APPROVAL REQUIRED — Special Discount
═══════════════════════════════════════════════════════════
  Customer : Jane Smith
  Product  : HelpDesk Pro
  Discount : 30%
  Reason   : SaaS startup, early-stage, 25 users
═══════════════════════════════════════════════════════════

  Manager — Approve this discount? [y/n]: y
  Approved discount % [30%]: 20
  Optional message for the customer: Welcome offer for early-stage startups

🤖 Alex: Great news! Our manager has approved a special 20% discount
         on HelpDesk Pro for you...
```

---

## CLI Commands

| Command | Action |
|---|---|
| `/quit` | Exit the assistant |
| `/new` | Start a fresh session (new conversation thread) |
| `/leads` | View all leads captured in `data/leads.json` |
| `/help` | Show help and command list |

---

## How Human-in-the-Loop Works

When a customer asks for a special discount, the `request_special_discount` tool calls
LangGraph's `interrupt()` function. This **pauses the graph** mid-execution and surfaces
the request to the CLI. You (as the manager) see the request and enter `y` or `n`.

The graph then **resumes** with the manager's decision via `Command(resume={...})` and
the agent responds to the customer accordingly.

```
Customer asks for discount
        ↓
Agent calls request_special_discount tool
        ↓
  interrupt() ← graph pauses here
        ↓
Manager sees approval prompt in terminal
        ↓
Manager types y/n → Command(resume={...})
        ↓
Graph resumes → agent tells customer the outcome
```

This pattern uses no external services — everything runs locally.

---

## Architecture

```
cli.py  (user interface)
  │
  ├── create_sales_agent()   ←── agent.py
  │     ├── LLM (OpenAI / Gemini)
  │     ├── SqliteSaver      ←── data/memory.db   (per-session memory)
  │     └── create_react_agent (langgraph.prebuilt)
  │           ├── prompt     ←── SALES_SYSTEM_PROMPT
  │           └── tools      ←── tools.py
  │                 ├── lookup_product_info    → data/products.json
  │                 ├── lookup_pricing         → data/pricing.json
  │                 ├── capture_lead_info      → data/leads.json
  │                 ├── request_special_discount → interrupt() → CLI
  │                 └── check_availability     → data/products.json
  │
  └── stream_agent_response()   (token streaming + interrupt detection)
        └── resume_after_interrupt()  (manager input → Command(resume=...))
```

---

## Supported Products (Mock Catalog)

| Product | Category | Starting Price |
|---|---|---|
| AutoFlow CRM | CRM | \$29/user/month |
| DataSync Analytics | Business Intelligence | \$49/user/month |
| TaskBot Automation | Workflow Automation | \$19/user/month |
| SecureVault | Data Security | \$39/user/month |
| HelpDesk Pro | Customer Support | \$25/user/month |

All products include a 14–30 day free trial. Annual billing saves 20%.

---

## Switching LLM Providers

In your `.env`:

```env
# Use OpenAI (default)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini        # or: gpt-4o, gpt-4-turbo

# Use Google Gemini
LLM_PROVIDER=google
GOOGLE_API_KEY=...
LLM_MODEL=gemini-2.0-flash   # or: gemini-1.5-pro
```

Then install the extra package if using Gemini:

```bash
pip install langchain-google-genai
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `OPENAI_API_KEY is not set` | Make sure `.env` exists and has your key |
| `ModuleNotFoundError: langgraph` | Run `pip install -r requirements.txt` inside your venv |
| `ModuleNotFoundError: dotenv` | Run `pip install python-dotenv` |
| SQLite error on first run | Run `python cli.py` again — tables are created automatically |
| Agent gives wrong pricing | All pricing is in `data/pricing.json` — edit freely |

---

## Extending the Assistant

| Feature | Where to add it |
|---|---|
| New product | Add to `data/products.json` |
| New pricing tier | Add to `data/pricing.json` |
| New tool (e.g. book a demo) | Add `@tool` function to `tools.py`, add to `SALES_TOOLS` list |
| CRM integration | Call your CRM API inside `capture_lead_info` |
| Email tool | Add a `send_followup_email` tool using `smtplib` or SendGrid |
| Different LLM | Set `LLM_PROVIDER` in `.env` |

---

## License

MIT — Part of the LangGraph repository. See the root [LICENSE](../LICENSE) file.
