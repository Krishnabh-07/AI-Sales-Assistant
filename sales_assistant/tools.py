"""
Sales tools for the AI Sales Assistant.

These tools are executed by the LangGraph ToolNode inside create_react_agent.
The `interrupt()` call in `request_special_discount` triggers a human-in-the-loop
pause that requires manager approval before the graph can resume.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from langgraph.types import interrupt

# ── Paths ──────────────────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent / "data"
_LEADS_FILE = _DATA_DIR / "leads.json"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_products() -> dict:
    with open(_DATA_DIR / "products.json", encoding="utf-8") as f:
        return json.load(f)


def _load_pricing() -> dict:
    with open(_DATA_DIR / "pricing.json", encoding="utf-8") as f:
        return json.load(f)


def _load_leads() -> dict:
    if _LEADS_FILE.exists():
        with open(_LEADS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"leads": []}


def _save_leads(data: dict) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_LEADS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _find_product_in_pricing(query: str, plans: dict) -> Optional[str]:
    """Fuzzy match a product name against pricing keys."""
    q = query.lower().strip()
    for name in plans:
        if q in name.lower() or name.lower() in q:
            return name
    return None


# ── Tools ──────────────────────────────────────────────────────────────────────

@tool
def lookup_product_info(query: str) -> str:
    """Search our product catalog by name, category, or use-case.

    Use this when the customer asks what products we offer, what features a
    product has, which product suits a specific need, or wants a comparison.

    Args:
        query: A product name, category (e.g. 'CRM', 'automation'), or
               use-case keyword (e.g. 'compliance', 'support tickets').
               Pass 'all' to list every product.
    """
    data = _load_products()
    q = query.lower().strip()

    if q == "all":
        matches = data["products"]
    else:
        matches = [
            p for p in data["products"]
            if (
                q in p["name"].lower()
                or q in p["category"].lower()
                or q in p["description"].lower()
                or q in p["tagline"].lower()
                or any(q in f.lower() for f in p["features"])
                or any(q in i.lower() for i in p["ideal_for"])
            )
        ]

    if not matches:
        all_names = ", ".join(p["name"] for p in data["products"])
        return (
            f"No products found matching '{query}'.\n"
            f"Our available products are: {all_names}.\n"
            "Try searching by category (CRM, Analytics, Automation, Security, Support) "
            "or pass 'all' to see everything."
        )

    lines = [f"Found {len(matches)} matching product(s) from {data['company']}:\n"]
    for p in matches:
        lines.append(f"{'─'*50}")
        lines.append(f"  Product : {p['name']}")
        lines.append(f"  Category: {p['category']}")
        lines.append(f"  Tagline : {p['tagline']}")
        lines.append(f"  Summary : {p['description']}")
        lines.append(f"  Features: {', '.join(p['features'][:5])}{'...' if len(p['features']) > 5 else ''}")
        lines.append(f"  Best For: {', '.join(p['ideal_for'])}")
        lines.append(f"  Users   : {p['min_users']}–{p['max_users']} users supported")
        lines.append(f"  Trial   : {p['free_trial_days']}-day free trial available")
        lines.append(f"  Support : {p['support']}")

    return "\n".join(lines)


@tool
def lookup_pricing(
    product_name: str,
    plan: str = "all",
    num_users: int = 1,
    annual_billing: bool = False,
) -> str:
    """Look up pricing for a specific product and plan.

    Use this when the customer asks about cost, pricing, plans, or what they
    would pay for a given number of users.

    Args:
        product_name: Name of the product (e.g. 'AutoFlow CRM', 'HelpDesk Pro').
                      Partial matches are supported.
        plan: Which plan tier to show — 'starter', 'professional', 'enterprise',
              or 'all' to show every tier. Default is 'all'.
        num_users: Number of users for the cost calculation. Default is 1.
        annual_billing: If True, show annual billing prices (saves 20%).
    """
    data = _load_pricing()
    plans_data = data["plans"]

    matched_name = _find_product_in_pricing(product_name, plans_data)
    if not matched_name:
        available = ", ".join(plans_data.keys())
        return (
            f"Product '{product_name}' not found in our pricing catalog.\n"
            f"Available products: {available}"
        )

    product_plans = plans_data[matched_name]
    annual_pct = data["discounts"]["annual_billing_percent"]

    # Filter to requested plan
    if plan.lower() != "all" and plan.lower() in product_plans:
        display_plans = {plan.lower(): product_plans[plan.lower()]}
    else:
        display_plans = product_plans

    lines = [f"💰 Pricing for {matched_name} ({num_users} user{'s' if num_users != 1 else ''}):\n"]

    for tier_name, tier in display_plans.items():
        monthly_per_user = tier["price_per_user_monthly"]
        if annual_billing:
            effective_per_user = monthly_per_user * (1 - annual_pct / 100)
            billing_note = f"(annual billing — saves {annual_pct}%)"
        else:
            effective_per_user = monthly_per_user
            billing_note = "(monthly billing)"

        total_monthly = effective_per_user * num_users
        total_annual = total_monthly * 12

        lines.append(f"  [{tier_name.upper()} PLAN]  {billing_note}")
        lines.append(f"    Per user : ${effective_per_user:.2f}/user/month")
        lines.append(f"    Total    : ${total_monthly:.0f}/month  |  ${total_annual:.0f}/year  for {num_users} users")
        lines.append(f"    Min users: {tier['min_users']}")
        if not annual_billing:
            annual_savings = monthly_per_user * num_users * 12 * (annual_pct / 100)
            lines.append(f"    💡 Switch to annual and save ${annual_savings:.0f}/year")
        lines.append(f"    Includes : {', '.join(tier['includes'][:4])}...")
        lines.append("")

    # Volume discounts
    vol_tiers = data["discounts"]["volume_tiers"]
    applicable = [v for v in vol_tiers if num_users >= v["min_users"]]
    if applicable:
        best = max(applicable, key=lambda v: v["discount_percent"])
        lines.append(
            f"  🎉 Volume discount applies: {best['discount_percent']}% off for {best['label']}"
        )

    return "\n".join(lines)


@tool
def capture_lead_info(
    name: str,
    email: str,
    company: str,
    role: str = "",
    interest: str = "",
    budget_range: str = "",
    team_size: str = "",
    notes: str = "",
) -> str:
    """Save a qualified lead's contact information.

    Call this once you have collected the customer's name, email, and company
    through the conversation. Do NOT ask for all fields at once — gather them
    naturally during the conversation and call this tool when you have at least
    name, email, and company.

    Args:
        name: Customer's full name (required).
        email: Customer's email address (required).
        company: Customer's company or organization (required).
        role: Customer's job title or role (e.g. 'VP of Sales', 'CTO').
        interest: Product(s) they are interested in.
        budget_range: Approximate monthly/annual budget (e.g. '$500–$1000/month').
        team_size: Number of people on their team or company size.
        notes: Any other relevant notes from the conversation.
    """
    leads_data = _load_leads()

    new_lead = {
        "id": f"LEAD-{len(leads_data['leads']) + 1:04d}",
        "name": name,
        "email": email,
        "company": company,
        "role": role or "Not provided",
        "interest": interest or "Not specified",
        "budget_range": budget_range or "Not provided",
        "team_size": team_size or "Not provided",
        "notes": notes or "",
        "status": "new",
        "captured_at": datetime.now().isoformat(),
    }

    leads_data["leads"].append(new_lead)
    _save_leads(leads_data)

    return (
        f"✅ Lead captured successfully! (ID: {new_lead['id']})\n\n"
        f"  Name    : {name}\n"
        f"  Email   : {email}\n"
        f"  Company : {company}\n"
        f"  Role    : {new_lead['role']}\n"
        f"  Interest: {new_lead['interest']}\n"
        f"  Budget  : {new_lead['budget_range']}\n"
        f"  Team    : {new_lead['team_size']}\n\n"
        "Our sales team will follow up with you shortly. "
        "In the meantime, feel free to ask me anything else!"
    )


@tool
def request_special_discount(
    customer_name: str,
    product_name: str,
    requested_discount_percent: float,
    justification: str,
) -> str:
    """Request a special discount that requires manager approval.

    Use this ONLY when:
    - The customer explicitly asks for a special discount beyond standard pricing, AND
    - Standard volume discounts or annual billing savings are not sufficient.

    This will PAUSE the conversation and ask a manager to approve or decline.
    Do NOT use this for standard volume or annual discounts — those are automatic.

    Args:
        customer_name: The customer's name.
        product_name: The product the discount applies to.
        requested_discount_percent: The discount % being requested (e.g. 15.0 for 15%).
        justification: Why this customer deserves a special discount (from the conversation context).
    """
    # ── Human-in-the-loop: execution pauses here until a manager responds ──
    decision = interrupt({
        "type": "discount_approval_request",
        "display_message": (
            f"\n{'═'*58}\n"
            f"  🔔  MANAGER APPROVAL REQUIRED — Special Discount\n"
            f"{'═'*58}\n"
            f"  Customer : {customer_name}\n"
            f"  Product  : {product_name}\n"
            f"  Discount : {requested_discount_percent}%\n"
            f"  Reason   : {justification}\n"
            f"{'═'*58}"
        ),
        "customer_name": customer_name,
        "product_name": product_name,
        "requested_discount_percent": requested_discount_percent,
        "justification": justification,
    })

    if decision.get("approved"):
        approved_pct = decision.get("approved_percent", requested_discount_percent)
        note = decision.get("manager_note", "")
        return (
            f"✅ Special discount APPROVED by manager!\n"
            f"  Customer        : {customer_name}\n"
            f"  Product         : {product_name}\n"
            f"  Approved Discount: {approved_pct}%\n"
            f"  Manager Note    : {note or 'None'}\n\n"
            f"You may now offer {customer_name} a {approved_pct}% special discount "
            f"on {product_name}. Communicate this clearly and proceed to close."
        )
    else:
        reason = decision.get("reason", "Request did not meet discount policy criteria")
        return (
            f"❌ Special discount DECLINED by manager.\n"
            f"  Reason: {reason}\n\n"
            "I'm unable to offer that special discount at this time. "
            "However, I can still offer our standard savings:\n"
            "  • 20% off with annual billing\n"
            "  • Volume discounts for 50, 100, or 250+ users\n"
            "Would any of these alternatives work for you?"
        )


@tool
def check_availability(product_name: str, num_users: int) -> str:
    """Check whether a product supports a given number of users.

    Use this when the customer mentions their team size and you want to confirm
    that the product can accommodate them before recommending it.

    Args:
        product_name: The product name to check (partial match supported).
        num_users: The number of users the customer needs.
    """
    data = _load_products()
    q = product_name.lower()

    for product in data["products"]:
        if q in product["name"].lower() or product["name"].lower() in q:
            if num_users < product["min_users"]:
                return (
                    f"⚠️  {product['name']} requires at least {product['min_users']} users.\n"
                    f"You requested {num_users}. "
                    "Consider our other products that support smaller teams, "
                    "or check if adding more users fits your roadmap."
                )
            elif num_users > product["max_users"]:
                return (
                    f"⚠️  {product['name']} supports up to {product['max_users']} users.\n"
                    f"You requested {num_users}. "
                    "Please contact our Enterprise sales team for a custom deployment."
                )
            else:
                return (
                    f"✅ {product['name']} fully supports {num_users} user(s).\n"
                    f"   (Range: {product['min_users']}–{product['max_users']} users)"
                )

    all_names = ", ".join(p["name"] for p in data["products"])
    return (
        f"Product '{product_name}' not found. Available products: {all_names}"
    )


# ── Exported tool list ─────────────────────────────────────────────────────────

SALES_TOOLS = [
    lookup_product_info,
    lookup_pricing,
    capture_lead_info,
    request_special_discount,
    check_availability,
]
