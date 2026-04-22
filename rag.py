"""
rag.py — Knowledge Base Retrieval for AutoStream Agent
Simple keyword-based RAG over a local JSON knowledge base.
"""

import json
from pathlib import Path


def load_knowledge_base(path: str = "knowledge_base.json") -> dict:
    kb_path = Path(path)
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base not found at: {path}")
    with open(kb_path, "r") as f:
        return json.load(f)


def retrieve_context(query: str, kb: dict) -> str:
    """
    Retrieve relevant context from the knowledge base based on the user query.
    Uses keyword matching to find relevant sections and returns them as a
    formatted string for injection into the LLM prompt.
    """
    query_lower = query.lower()
    sections = []

    # Keywords for intent-to-section mapping
    pricing_keywords = ["price", "pricing", "cost", "plan", "basic", "pro", "month", "pay", "cheap", "expensive", "how much"]
    policy_keywords = ["refund", "cancel", "support", "trial", "policy", "policies", "guarantee"]
    company_keywords = ["what is", "autostream", "about", "platform", "creator", "youtube", "instagram", "tiktok"]

    if any(kw in query_lower for kw in pricing_keywords):
        basic = kb["pricing"]["basic"]
        pro = kb["pricing"]["pro"]
        sections.append(
            f"PRICING INFORMATION:\n"
            f"Basic Plan – {basic['price']}: {', '.join(basic['features'])}\n"
            f"Pro Plan – {pro['price']}: {', '.join(pro['features'])}"
        )

    if any(kw in query_lower for kw in policy_keywords):
        p = kb["policies"]
        sections.append(
            f"COMPANY POLICIES:\n"
            f"Refund Policy: {p['refund']}\n"
            f"Support: {p['support']}\n"
            f"Trial: {p['trial']}\n"
            f"Cancellation: {p['cancellation']}"
        )

    if any(kw in query_lower for kw in company_keywords):
        c = kb["company"]
        sections.append(
            f"ABOUT AUTOSTREAM:\n"
            f"{c['description']}\n"
            f"Supported Platforms: {', '.join(c['platforms_supported'])}"
        )

    # Default fallback — return all context if no specific match
    if not sections:
        basic = kb["pricing"]["basic"]
        pro = kb["pricing"]["pro"]
        p = kb["policies"]
        c = kb["company"]
        sections.append(
            f"ABOUT AUTOSTREAM: {c['description']}\n\n"
            f"PRICING:\n"
            f"Basic Plan – {basic['price']}: {', '.join(basic['features'])}\n"
            f"Pro Plan – {pro['price']}: {', '.join(pro['features'])}\n\n"
            f"POLICIES:\nRefund: {p['refund']} | Support: {p['support']}"
        )

    return "\n\n".join(sections)
