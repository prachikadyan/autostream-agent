# AutoStream AI Agent
### Social-to-Lead Agentic Workflow | Built for ServiceHive × Inflx

A production-grade conversational AI agent that qualifies leads through natural dialogue — identifying intent, answering product questions via RAG, and capturing qualified leads through a tool-triggered pipeline.

---

## Project Structure

```
autostream_agent/
├── agent.py            # LangGraph agent — state machine, nodes, routing
├── rag.py              # RAG retrieval over local knowledge base
├── tools.py            # mock_lead_capture() and validation helpers
├── knowledge_base.json # AutoStream product data (pricing, policies, company info)
├── requirements.txt
├── .env.example
└── README.md
```

---

##  Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/autostream-agent.git
cd autostream-agent

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Open .env and add your Anthropic API key:
# ANTHROPIC_API_KEY=sk-ant-...
```

> Get your key at: https://console.anthropic.com

### 3. Run the Agent

```bash
python agent.py
```

### Example Conversation

```
AutoStream: Hey there!  I can help you with our plans, features, and getting started.

You: Hi, what plans do you offer?

AutoStream: AutoStream has two plans:
• Basic ($29/mo) — 10 videos/month, 720p, email support
• Pro ($79/mo) — Unlimited videos, 4K, AI captions, 24/7 support
Which fits your needs best?

You: The Pro plan looks great, I want to sign up for my YouTube channel.

AutoStream: Awesome! Let's get you started. What's your name?

You: Alex Johnson

AutoStream: Great, Alex! What's your email address?

You: alex@example.com

AutoStream: Perfect! And which platform are you mainly creating for?

You: YouTube

 Lead captured successfully: Alex Johnson, alex@example.com, YouTube

AutoStream:  You're all set, Alex! Check your inbox — your free trial link is on the way!
```

---

##  Architecture (~200 words)

### Why LangGraph?

LangGraph was chosen over AutoGen because this use case is **single-agent with deterministic state transitions** — not a multi-agent debate. LangGraph's `StateGraph` gives fine-grained control over *exactly* when nodes fire and how state evolves, which is critical for a lead-collection pipeline where premature tool invocation would break the user experience.

### How State is Managed

State is modelled as a typed dictionary (`AgentState`) with six fields: the full message history, the classified intent, the current pipeline stage (`chat` → `collecting_name` → `collecting_email` → `collecting_platform` → `captured`), and the three lead fields. A `MemorySaver` checkpointer persists this state across every conversation turn using a `thread_id`, enabling genuine multi-turn memory without an external database.

### Node Flow

```
START → classify_intent
           ↓
    [route by stage/intent]
     /              \
retrieve_context   collect_lead_details
     ↓                     ↓
generate_response   execute_lead_capture (when all 3 fields are filled)
     ↓                     ↓
    END                   END
```

The `collect_lead_details` node advances one field per turn, extracting values via LLM parsing (name) or regex (email) before moving to the next sub-stage. The `execute_lead_capture` node fires **only** after all three values are confirmed — never prematurely.

---
##  WhatsApp Deployment via Webhooks

### Overview

To deploy this agent on WhatsApp, we use the **WhatsApp Business Cloud API** (Meta) with a webhook-based event loop. Here's how it works end-to-end:

### Architecture

```
WhatsApp User
     ↓ (sends message)
Meta Webhook → POST /webhook (your server)
     ↓
FastAPI / Flask Handler
  • Parses incoming message
  • Looks up or creates session (by phone number as thread_id)
  • Invokes LangGraph agent with thread_id
     ↓
Agent produces response
     ↓
POST to Meta Graph API → /messages
     ↓
WhatsApp User receives reply
```

### Step-by-Step Integration

**1. Set up a webhook server (FastAPI example):**

```python
from fastapi import FastAPI, Request
from agent import build_agent

app = FastAPI()
agent = build_agent()

@app.post("/webhook")
async def receive_message(request: Request):
    body = await request.json()
    # Parse Meta webhook payload
    message = body["entry"][0]["changes"][0]["value"]["messages"][0]
    phone   = message["from"]
    text    = message["text"]["body"]

    # Use phone number as unique session ID
    config = {"configurable": {"thread_id": phone}}
    result = agent.invoke({"messages": [{"role": "user", "content": text}], ...}, config)

    # Send reply via Meta Graph API
    send_whatsapp_message(phone, result["response"])
    return {"status": "ok"}
```

**2. Register your webhook with Meta:**
- Go to Meta Developer Console → WhatsApp → Configuration
- Set Webhook URL to `https://yourdomain.com/webhook`
- Set a verify token and handle the `GET /webhook` verification handshake

**3. Handle session persistence:**
- Each user's phone number becomes their `thread_id`
- LangGraph's `MemorySaver` maintains state across turns automatically
- For production, swap `MemorySaver` with `PostgresSaver` for persistence across server restarts

**4. Deploy:**
- Host on Railway, Render, or AWS Lambda
- Expose via HTTPS (required by Meta)
- Use ngrok for local development testing

### Key Considerations
- **Rate limits**: Meta allows ~80 messages/second on the Business API
- **Message types**: Handle text, quick replies, and interactive buttons for better UX
- **Opt-in compliance**: Always ensure users have opted in before messaging them

---

##  Evaluation Checklist

| Requirement | Status |
|---|---|
| Intent classification (3 categories) | |
| RAG over local knowledge base | |
| State retained across 5–6 turns |  (MemorySaver + thread_id) |
| Lead collection (name + email + platform) | |
| Tool fires only after all 3 values collected |  |
| LangGraph framework |  |
| Claude 3 Haiku LLM | 
| Clean code structure | |

