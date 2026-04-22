"""
tools.py — Tool Execution Layer for AutoStream Agent
Contains the mock lead capture function and any future tool integrations.
"""

import re
from datetime import datetime


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Simulates capturing a qualified lead to a CRM system.
    In production, this would POST to a CRM API (e.g. HubSpot, Salesforce).

    Args:
        name: Full name of the lead
        email: Email address of the lead
        platform: Primary content platform (YouTube, Instagram, etc.)

    Returns:
        dict with status and lead details
    """
    # Validate email format before "saving"
    email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    if not re.match(email_pattern, email):
        return {
            "status": "error",
            "message": f"Invalid email format: {email}"
        }

    lead = {
        "name": name.strip(),
        "email": email.strip().lower(),
        "platform": platform.strip(),
        "captured_at": datetime.utcnow().isoformat() + "Z",
        "source": "AutoStream AI Agent",
        "plan_interest": "Pro"
    }

    # Simulate the print output required by the assignment spec
    print(f"\n{'='*50}")
    print(f"✅ Lead captured successfully: {name}, {email}, {platform}")
    print(f"{'='*50}\n")

    return {
        "status": "success",
        "message": f"Lead successfully captured for {name}.",
        "lead": lead
    }


def validate_email(email: str) -> bool:
    """Validate email address format."""
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email.strip()))
