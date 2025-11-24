# backend/src/wellness_agent.py
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

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

# --------------------------
# Logging
# --------------------------
logger = logging.getLogger("wellness_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

load_dotenv(".env.local")

# --------------------------
# Persistence setup
# --------------------------
DATA_DIR = Path("wellness_data")  # adjusts relative to process cwd
DATA_DIR.mkdir(parents=True, exist_ok=True)
WELLNESS_FILE = DATA_DIR / "wellness_log.json"

# ensure file exists and is valid JSON array
if not WELLNESS_FILE.exists():
    with open(WELLNESS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2, ensure_ascii=False)

def _read_all_checkins():
    try:
        with open(WELLNESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _append_checkin(entry: dict):
    all_entries = _read_all_checkins()
    all_entries.append(entry)
    with open(WELLNESS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)
    logger.info(f"Appended check-in for {entry.get('timestamp')}")

def _get_last_checkin():
    all_entries = _read_all_checkins()
    if not all_entries:
        return None
    return all_entries[-1]

# --------------------------
# Helpers
# --------------------------
def _make_note_if_empty(mood: str, objectives: Optional[List[str]], note: Optional[str]) -> str:
    """
    If LLM didn't provide a note, create a short summary note automatically:
    "Mood: <mood>. Top goal: <first objective>."
    """
    if note and isinstance(note, str) and note.strip():
        return note.strip()
    top_obj = ""
    if objectives and len(objectives) > 0:
        top_obj = objectives[0].strip()
    note_parts = []
    if mood:
        note_parts.append(f"Mood: {mood.strip()}.")
    if top_obj:
        note_parts.append(f"Top goal: {top_obj}.")
    # fallback generic if nothing else
    if not note_parts:
        return "Quick check-in (no summary)."
    return " ".join(note_parts)

# --------------------------
# Tools (LLM-callable)
# --------------------------
@function_tool
async def save_checkin(
    context: RunContext,
    mood: str,
    energy: Optional[str] = None,
    objectives: Optional[List[str]] = None,
    note: Optional[str] = None,
):
    """
    Save a wellness check-in to wellness_log.json.
    Required: mood. Optional: energy, objectives (list), note.
    If 'note' is empty, auto-generate a short note summarizing mood + top objective.
    """
    ts = datetime.utcnow().isoformat() + "Z"

    # normalize inputs
    mood_val = mood.strip() if isinstance(mood, str) else str(mood)
    energy_val = (energy.strip() if isinstance(energy, str) else energy) or ""
    objs = [o.strip() for o in objectives] if objectives else []
    note_val = _make_note_if_empty(mood_val, objs, note)

    entry = {
        "timestamp": ts,
        "mood": mood_val,
        "energy": energy_val,
        "objectives": objs,
        "note": note_val,
    }
    _append_checkin(entry)
    logger.info(f"Saved check-in: mood={mood_val}, objectives={objs}, note_present={'yes' if note_val else 'no'}")
    return f"Saved check-in for {ts}."

@function_tool
async def read_summary(context: RunContext, n: Optional[int] = 3):
    """
    Return a brief summary of the last n check-ins for context.
    """
    all_entries = _read_all_checkins()
    if not all_entries:
        return "No previous check-ins found."
    recent = all_entries[-n:]
    summaries = []
    for e in recent:
        ts = e.get("timestamp", "")
        mood = e.get("mood", "")
        energy = e.get("energy", "")
        objs = e.get("objectives", [])
        note = e.get("note", "")
        summaries.append(f"{ts}: mood='{mood}', energy='{energy}', objectives={objs}, note='{note}'")
    return "\n".join(summaries)

# --------------------------
# Agent definition
# --------------------------
class WellnessAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a supportive, grounded daily health & wellness voice companion.
Your job is to perform a short daily check-in (2-6 questions), collect the user's mood, energy, and 1-3 objectives for the day, offer one or two simple actionable suggestions, and then save the check-in data.

Rules:
1) Ask about mood first: "How are you feeling today?"
2) Ask about energy: "What's your energy like today?" (or skip if user gave a clear scale)
3) Ask for 1-3 objectives: "What are 1-3 things you'd like to get done today?"
4) Offer 1 short, practical suggestion (small step, break, grounding) based on user's answers.
5) Read the most recent check-in (use read_summary) and briefly reference it: e.g. "Last time you said you were low on energy; how is today different?"
6) When you have mood and objectives (and optionally energy), CALL the tool `save_checkin` with those fields.
   - Include a short 'note' if you can: a 1-sentence summary (optional).
   - If you don't include 'note', the backend will auto-generate one.
7) After saving, recap the mood and objectives, and ask "Does that sound right?"

Be supportive and non-judgmental. Never provide medical advice or diagnoses. Keep responses short and practical.
            """,
            tools=[save_checkin, read_summary],
        )

# --------------------------
# Prewarm and session
# --------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # build session pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.0-flash-lite"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # start the session with the wellness agent
    await session.start(
        agent=WellnessAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    # when connected, proactively log previous check-in presence for debugging
    last = _get_last_checkin()
    if last:
        logger.info(f"Previous check-in found: {last.get('timestamp')} note_present={'yes' if last.get('note') else 'no'}")
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
