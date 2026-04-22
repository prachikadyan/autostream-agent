

import os
import operator
from typing import TypedDict, Optional, Annotated, List
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from rag import load_knowledge_base, retrieve_context
from tools import mock_lead_capture, validate_email

load_dotenv()

# ─────────────────────────────────────────────
# 1. STATE DEFINITION
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]   # Full conversation history
    intent: str                                       # "greeting" | "inquiry" | "high_intent"
    stage: str                                        # "chat" | "collecting_name" | "collecting_email" | "collecting_platform" | "captured"
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    retrieved_context: str
    response: str


# ─────────────────────────────────────────────
# 2. INITIALISE LLM & KNOWLEDGE BASE
# ─────────────────────────────────────────────

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.3,
    max_tokens=512
)

KB = load_knowledge_base("knowledge_base.json")

SYSTEM_PROMPT = """You are an AI sales assistant for AutoStream — a SaaS platform that provides
automated video editing tools for content creators on YouTube, Instagram, TikTok, and more.

Your personality: friendly, concise, helpful. Never be pushy. 
Answer questions accurately using the provided knowledge base context.
If a user seems interested in signing up, gently guide them through the process.

IMPORTANT RULES:
- Only use facts from the provided context. Do not invent features or prices.
- Keep responses short (2-4 sentences) unless more detail is needed.
- Never ask for personal details unless the user has shown clear intent to sign up."""


# ─────────────────────────────────────────────
# 3. NODE FUNCTIONS
# ─────────────────────────────────────────────

def classify_intent(state: AgentState) -> AgentState:
    """Classify the user's latest message into one of three intent categories."""

    last_msg = state["messages"][-1]["content"]
    current_stage = state.get("stage", "chat")

    # If already in lead-collection flow, preserve stage — don't reclassify
    if current_stage in ["collecting_name", "collecting_email", "collecting_platform"]:
        return {"intent": "high_intent", "stage": current_stage}

    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in state["messages"][-6:]
    )

    classification_prompt = f"""Classify the user's intent based on their latest message and conversation history.

Conversation:
{history_text}

Return ONLY one of these labels (no explanation):
- greeting        → casual hello, no specific question
- inquiry         → asking about features, pricing, policies, or general questions  
- high_intent     → clearly wants to sign up, try, purchase, or subscribe to a plan

Label:"""

    response = llm.invoke([HumanMessage(content=classification_prompt)])
    raw = response.content.strip().lower()

    if "high_intent" in raw:
        intent = "high_intent"
    elif "inquiry" in raw:
        intent = "inquiry"
    else:
        intent = "greeting"

    return {"intent": intent}


def retrieve_context_node(state: AgentState) -> AgentState:
    """Retrieve relevant knowledge base chunks for the user's query."""
    last_msg = state["messages"][-1]["content"]
    context = retrieve_context(last_msg, KB)
    return {"retrieved_context": context}


def generate_response(state: AgentState) -> AgentState:
    """Generate a response using LLM + retrieved context."""
    context = state.get("retrieved_context", "")
    intent = state.get("intent", "greeting")

    # Build full message history for the LLM
    lc_messages = [SystemMessage(content=SYSTEM_PROMPT)]

    if context:
        lc_messages.append(SystemMessage(
            content=f"RELEVANT KNOWLEDGE BASE CONTEXT:\n{context}"
        ))

    for msg in state["messages"][-8:]:  # Keep last 8 turns for memory
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        else:
            lc_messages.append(AIMessage(content=msg["content"]))

    # Hint for high-intent users to nudge them toward sign-up
    if intent == "high_intent":
        lc_messages.append(SystemMessage(
            content="The user is showing high intent to sign up. "
                    "Warmly acknowledge their interest and ask if they'd like to get started "
                    "by sharing a few quick details."
        ))

    ai_response = llm.invoke(lc_messages)
    reply = ai_response.content.strip()

    # If high-intent, transition stage to start collecting lead info
    new_stage = "collecting_name" if intent == "high_intent" else state.get("stage", "chat")

    return {
        "response": reply,
        "stage": new_stage,
        "messages": [{"role": "assistant", "content": reply}]
    }


def collect_lead_details(state: AgentState) -> AgentState:
    """
    Multi-turn lead collection node.
    Advances through collecting name → email → platform sequentially.
    Extracts values from user's latest message using LLM.
    """
    stage = state.get("stage", "collecting_name")
    last_msg = state["messages"][-1]["content"]

    lead_name = state.get("lead_name")
    lead_email = state.get("lead_email")
    lead_platform = state.get("lead_platform")

    reply = ""
    next_stage = stage

    # ── Stage: collecting_name ──
    if stage == "collecting_name":
        extraction = llm.invoke([HumanMessage(
            content=f"Extract only the person's name from this message. "
                    f"Return just the name, nothing else. Message: \"{last_msg}\""
        )])
        extracted = extraction.content.strip().strip('"').strip("'")

        # Basic sanity check — name should be 2-50 chars
        if 2 <= len(extracted) <= 50 and extracted.lower() not in ["unknown", "none", ""]:
            lead_name = extracted
            reply = f"Great, {lead_name}! 😊 What's your email address?"
            next_stage = "collecting_email"
        else:
            reply = "I didn't quite catch your name — could you share it again?"
            next_stage = "collecting_name"

    # ── Stage: collecting_email ──
    elif stage == "collecting_email":
        # Extract email using pattern matching
        import re
        email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", last_msg)
        if email_match and validate_email(email_match.group()):
            lead_email = email_match.group().lower()
            reply = f"Perfect! And which platform are you mainly creating for? (e.g. YouTube, Instagram, TikTok)"
            next_stage = "collecting_platform"
        else:
            reply = "That doesn't look like a valid email. Could you double-check and resend it?"
            next_stage = "collecting_email"

    # ── Stage: collecting_platform ──
    elif stage == "collecting_platform":
        known_platforms = ["youtube", "instagram", "tiktok", "twitter", "linkedin", "facebook", "x"]
        matched = next(
            (p.capitalize() for p in known_platforms if p in last_msg.lower()),
            None
        )
        if matched:
            lead_platform = matched
            next_stage = "execute_capture"
            reply = ""  # Will be set by capture node
        else:
            # Accept any platform name the user provides
            words = last_msg.strip().split()
            lead_platform = words[0].capitalize() if words else last_msg
            next_stage = "execute_capture"
            reply = ""

    return {
        "lead_name": lead_name,
        "lead_email": lead_email,
        "lead_platform": lead_platform,
        "stage": next_stage,
        "response": reply,
        "messages": [{"role": "assistant", "content": reply}] if reply else []
    }


def execute_lead_capture(state: AgentState) -> AgentState:
    """Call mock_lead_capture once all three lead fields are collected."""
    name = state.get("lead_name", "")
    email = state.get("lead_email", "")
    platform = state.get("lead_platform", "")

    result = mock_lead_capture(name, email, platform)

    if result["status"] == "success":
        reply = (
            f"🎉 You're all set, {name}! We've registered your interest in AutoStream Pro. "
            f"Check your inbox at {email} — you'll receive your free trial link within minutes. "
            f"Can't wait to see the amazing content you'll create on {platform}!"
        )
    else:
        reply = f"Something went wrong saving your details: {result['message']}. Please try again."

    return {
        "stage": "captured",
        "response": reply,
        "messages": [{"role": "assistant", "content": reply}]
    }


# ─────────────────────────────────────────────
# 4. ROUTING LOGIC
# ─────────────────────────────────────────────

def route_after_classification(state: AgentState) -> str:
    stage = state.get("stage", "chat")
    intent = state.get("intent", "greeting")

    if stage in ["collecting_name", "collecting_email", "collecting_platform"]:
        return "collect_lead_details"
    elif intent == "high_intent":
        return "retrieve_context"   # Still retrieve context, then generate will shift stage
    else:
        return "retrieve_context"


def route_after_collection(state: AgentState) -> str:
    if state.get("stage") == "execute_capture":
        return "execute_lead_capture"
    return END


def route_after_generation(state: AgentState) -> str:
    if state.get("stage") == "collecting_name":
        return END   # Response already set, wait for next user turn
    return END


# ─────────────────────────────────────────────
# 5. BUILD THE GRAPH
# ─────────────────────────────────────────────

def build_agent():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("generate_response", generate_response)
    graph.add_node("collect_lead_details", collect_lead_details)
    graph.add_node("execute_lead_capture", execute_lead_capture)

    # Entry point
    graph.add_edge(START, "classify_intent")

    # After classification: decide path
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "retrieve_context": "retrieve_context",
            "collect_lead_details": "collect_lead_details",
        }
    )

    # Knowledge retrieval always feeds into response generation
    graph.add_edge("retrieve_context", "generate_response")
    graph.add_edge("generate_response", END)

    # Lead collection: either capture or wait for next input
    graph.add_conditional_edges(
        "collect_lead_details",
        route_after_collection,
        {
            "execute_lead_capture": "execute_lead_capture",
            END: END
        }
    )

    graph.add_edge("execute_lead_capture", END)

    # Compile with memory checkpointer for multi-turn persistence
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# ─────────────────────────────────────────────
# 6. CONVERSATION RUNNER (CLI)
# ─────────────────────────────────────────────

def run_agent():
    """Interactive CLI loop for the AutoStream agent."""
    agent = build_agent()
    thread_id = "session-001"   # Single session ID — change per user in production
    config = {"configurable": {"thread_id": thread_id}}

    print("\n" + "="*60)
    print("  AutoStream AI Assistant")
    print("  Powered by Claude 3 Haiku + LangGraph")
    print("="*60)
    print("Type your message below. Press Ctrl+C to exit.\n")

    # Initial state seed
    initial_state: AgentState = {
        "messages": [],
        "intent": "greeting",
        "stage": "chat",
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "retrieved_context": "",
        "response": ""
    }

    # Warm greeting
    greeting = "Hey there! 👋 I'm the AutoStream assistant. I can help you with our plans, features, and getting started. What can I help you with today?"
    print(f"AutoStream: {greeting}\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            # Prepare input state with new user message
            input_state = {
                **initial_state,
                "messages": [{"role": "user", "content": user_input}]
            }

            # Invoke graph
            result = agent.invoke(input_state, config=config)

            # Extract and print response
            response = result.get("response", "")
            if response:
                print(f"\nAutoStream: {response}\n")

            # Update initial_state for next turn (carry over lead fields + stage)
            initial_state = {
                "messages": [],
                "intent": result.get("intent", "greeting"),
                "stage": result.get("stage", "chat"),
                "lead_name": result.get("lead_name"),
                "lead_email": result.get("lead_email"),
                "lead_platform": result.get("lead_platform"),
                "retrieved_context": "",
                "response": ""
            }

            # End session if lead captured
            if result.get("stage") == "captured":
                print("─"*60)
                print("Lead capture complete. Session ended.")
                break

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n[Error]: {e}")
            raise


if __name__ == "__main__":
    run_agent()
