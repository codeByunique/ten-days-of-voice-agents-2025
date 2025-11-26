# backend/src/sdr_agent.py
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("sdr_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

load_dotenv(".env.local")

# -------------------------
# Load FAQ JSON
# -------------------------
FAQ_PATH = Path("shared-data/day5_sdr_browserstack_faq.json")
if not FAQ_PATH.exists():
    raise FileNotFoundError(f"Unable to find FAQ file at {FAQ_PATH.resolve()}")

with open(FAQ_PATH, "r", encoding="utf-8") as f:
    FAQ_DATA = json.load(f)

FAQ_LIST = FAQ_DATA["faq"]

# -------------------------
# Lead persistence
# -------------------------
LEADS_DIR = Path("leads")
LEADS_DIR.mkdir(exist_ok=True)

ALL_LEADS = LEADS_DIR / "all_leads.json"
if not ALL_LEADS.exists():
    with open(ALL_LEADS, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)

LEAD_STATE = {}  # per room


def _get_room(context: RunContext):
    try:
        return context.room.name
    except:
        return "default"


def _get_lead(room):
    if room not in LEAD_STATE:
        LEAD_STATE[room] = {
            "name": None,
            "company": None,
            "email": None,
            "role": None,
            "use_case": None,
            "team_size": None,
            "timeline": None,
            "notes": "",
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
    return LEAD_STATE[room]


def _append_all_leads(entry):
    try:
        with open(ALL_LEADS, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        data = []

    data.append(entry)

    with open(ALL_LEADS, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# -------------------------
# FAQ Search Utility
# -------------------------
def faq_search(question: str):
    q = question.lower()

    if any(x in q for x in ["price", "pricing", "cost", "plans"]):
        return FAQ_DATA["pricing_basics"]
    if "free" in q or "trial" in q:
        return FAQ_DATA["free_tier"]
    if "who is this for" in q or "target" in q:
        return FAQ_DATA["who_is_it_for"]
    if "what does your product do" in q or "what do you do" in q:
        return FAQ_DATA["what_we_do"]

    words = [w for w in q.split() if len(w) > 2]

    best = ""
    best_score = 0
    for faq in FAQ_LIST:
        txt = (faq["q"] + " " + faq["a"]).lower()
        score = sum(1 for w in words if w in txt)
        if score > best_score:
            best_score = score
            best = faq["a"]

    return best or FAQ_DATA["overview"]


# -------------------------
# Tools
# -------------------------
@function_tool
async def search_faq_tool(context: RunContext, question: str):
    ans = faq_search(question)
    return ans


@function_tool
async def update_lead_tool(
    context: RunContext,
    name: Optional[str] = None,
    company: Optional[str] = None,
    email: Optional[str] = None,
    role: Optional[str] = None,
    use_case: Optional[str] = None,
    team_size: Optional[str] = None,
    timeline: Optional[str] = None,
    notes: Optional[str] = None,
):
    room = _get_room(context)
    lead = _get_lead(room)

    if name:
        lead["name"] = name
    if company:
        lead["company"] = company
    if email:
        lead["email"] = email
    if role:
        lead["role"] = role
    if use_case:
        lead["use_case"] = use_case
    if team_size:
        lead["team_size"] = team_size
    if timeline:
        lead["timeline"] = timeline.lower()
    if notes:
        lead["notes"] += " " + notes

    return json.dumps(lead, indent=2)


@function_tool
async def save_lead_tool(context: RunContext):
    room = _get_room(context)
    lead = _get_lead(room)

    filename = lead["name"] or "lead"
    safe = "".join(c if c.isalnum() else "_" for c in filename.lower())
    filepath = LEADS_DIR / f"{safe}_lead.json"

    lead_out = dict(lead)
    lead_out["saved_at"] = datetime.utcnow().isoformat() + "Z"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(lead_out, f, indent=2)

    _append_all_leads(lead_out)

    LEAD_STATE[room] = None
    return f"Lead saved as {filepath.name}"
    

# -------------------------
# Agent Instructions (UPDATED)
# -------------------------
class SDRAgent(Agent):
    def __init__(self) -> None:
        company_name = FAQ_DATA.get("company", "our product")
        super().__init__(
            instructions=f"""
You are a warm, professional Sales Development Representative (SDR) for {company_name}.

Your goals:
1) Greet visitors and understand what they are working on.
2) Answer questions about the product, who it is for, and basic pricing using the FAQ content only.
3) VERY IMPORTANT: In every conversation, you MUST collect these lead fields:
   - name
   - company
   - email
   - role
   - use case (what they want to use this for)
   - team size
   - timeline (now / soon / later)
4) At the end of the call, give a short verbal summary and call save_lead_tool.

Behavior:
- Start with a friendly greeting: mention BrowserStack by name and ask what brought them here.
  Example: "Hi, this is the BrowserStack sales assistant. What brings you here today?"
- As soon as you have answered 1–2 questions about the product, you MUST start asking for lead details.
  Ask the fields in this order: name, company, role, use case, team size, timeline, email.
- Whenever the user gives any of these details, CALL update_lead_tool immediately with those fields.
- Whenever the user asks a product/pricing/company question, CALL search_faq_tool with their question and base your answer ONLY on the returned text. Do not invent details.
- When the user says things like "that's all", "I'm done", "thanks", assume the call is ending:
  - Summarize: who they are, their company, role, use case, team size, timeline, and email.
  - Then CALL save_lead_tool.
  - Close politely.

Style:
- Be concise, helpful, and sales-friendly.
- Do not talk about internal tools or JSON; that is backend-only.
            """,
            tools=[search_faq_tool, update_lead_tool, save_lead_tool],
        )


# -------------------------
# Prewarm
# -------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# -------------------------
# Entrypoint
# -------------------------
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage = metrics.UsageCollector()

    @session.on("metrics_collected")
    def cb(ev: MetricsCollectedEvent):
        usage.collect(ev.metrics)

    await session.start(
        agent=SDRAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC())
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
