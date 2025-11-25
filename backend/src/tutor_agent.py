# backend/src/tutor_agent.py
import json
import logging
from pathlib import Path
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
)
from livekit.agents import function_tool, RunContext
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Logging
logger = logging.getLogger("tutor_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Load env
load_dotenv(".env.local")

# -------------------------------------------------------------
# Load Tutor Content
# -------------------------------------------------------------
CONTENT_FILE = Path("shared-data/day4_tutor_content.json")
if not CONTENT_FILE.exists():
    raise FileNotFoundError(f"Tutor content not found at {CONTENT_FILE.resolve()}")

with open(CONTENT_FILE, "r", encoding="utf-8") as f:
    TUTOR_CONTENT = {c["id"]: c for c in json.load(f)}

# -------------------------------------------------------------
# Global session state (per room)
# -------------------------------------------------------------
SESSION_STATE = {}

def get_state(room_name: str):
    if room_name not in SESSION_STATE:
        SESSION_STATE[room_name] = {
            "mode": None,
            "current_concept": None
        }
    return SESSION_STATE[room_name]

# -------------------------------------------------------------
# Tools
# -------------------------------------------------------------
@function_tool
async def set_mode(context: RunContext, room: str, mode: str):
    """
    mode = learn / quiz / teach_back
    """
    state = get_state(room)
    state["mode"] = mode
    logger.info(f"[{room}] set_mode -> {mode}")
    return f"Mode set to {mode}"

@function_tool
async def set_concept(context: RunContext, room: str, concept_id: str):
    """
    Set active concept
    """
    if concept_id not in TUTOR_CONTENT:
        logger.info(f"[{room}] set_concept failed - not found: {concept_id}")
        return f"Concept '{concept_id}' not found."
    state = get_state(room)
    state["current_concept"] = concept_id
    logger.info(f"[{room}] set_concept -> {concept_id}")
    return f"Concept set to {concept_id}"

# -------------------------------------------------------------
# Tutor Agent
# -------------------------------------------------------------
class TutorAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=f"""
You are a Teach-the-Tutor Active Recall Coach.

There are 3 modes:
- learn  → explain the concept summary (use Matthew voice)
- quiz  → ask questions (use Alicia voice)
- teach_back → ask user to explain concept and give friendly feedback (use Ken voice)

Flow:
1. Greet the user and ask which mode they want: learn / quiz / teach back.
2. Ask which concept they want (variables / loops).
3. Call set_mode + set_concept tools.

Behavior:
- LEARN mode: Explain using summary from the JSON file.
- QUIZ mode: Ask the sample_question from the JSON file.
- TEACH_BACK mode: Ask the user to explain the concept; give short qualitative feedback.
- User can switch modes anytime by saying "switch to learn/quiz/teach-back".
- Always keep tone supportive. No long lectures.

Use Murf Falcon voices:
- Matthew for learn
- Alicia for quiz
- Ken for teach_back
""",
            tools=[set_mode, set_concept],
        )

# -------------------------------------------------------------
# Prewarm
# -------------------------------------------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

# -------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------
async def entrypoint(ctx: JobContext):
    room = ctx.room.name
    logger.info(f"Starting tutor agent for room: {room}")

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash-lite"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2)
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True
    )

    usage = metrics.UsageCollector()

    @session.on("metrics_collected")
    def onmetrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    ctx.add_shutdown_callback(lambda: logger.info("Shutting down tutor agent session"))

    await session.start(
        agent=TutorAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC())
    )

    # Connect the worker to the room
    await ctx.connect()

# -------------------------------------------------------------
# CLI entry (important) - ensures uv / python invocation works
# -------------------------------------------------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
