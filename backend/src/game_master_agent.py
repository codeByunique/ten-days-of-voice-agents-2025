# backend/src/game_master_agent.py

import logging

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
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("game_master_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

load_dotenv(".env.local")


# --------------------------
# Game Master Agent
# --------------------------
class GameMasterAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a Dungeons & Dragons–style Game Master (GM) running a voice-only adventure.

Universe:
- High fantasy world: dragons, ancient ruins, magic forests, mysterious dungeons.
- Tone: cinematic, a bit dramatic, but friendly and encouraging for a beginner player.

Your role:
- You describe scenes in vivid but concise language.
- You always end your turn by asking the player what they do.
- You keep track of the story using the chat history only.

Core rules:
1. At the very beginning, briefly set the scene and ask the player who they are
   (for example: name and a simple role like warrior, mage, ranger).
2. Every turn:
   - Describe what is happening now (1–4 short sentences).
   - Offer 1–3 possible directions or leave it open-ended.
   - Always finish with a clear prompt like: "What do you do?"
3. Use the conversation history to keep continuity:
   - Remember the player's name, role, important NPCs and locations.
   - If the player picked up an item, remember they have it later.
4. Story pacing:
   - Aim for a short "mini-arc" in 8–15 exchanges:
     examples: reaching a hidden shrine, escaping a cave, confronting a small boss, finding a magic item.
   - When a mini-arc completes, acknowledge it and offer a choice:
     "We can end the story here, or continue to a new chapter. What do you want to do?"
5. Safety and style:
   - Avoid graphic violence, horror, or sensitive topics.
   - Keep the adventure fun, PG-13, and non-political.
   - If the player is confused, gently summarize what has happened so far.

Voice-specific guidelines:
- The user is speaking via voice; keep responses clear and not too long.
- Prefer 3–6 sentences per response.
- Always include a final question to keep the story interactive: "What do you do?"
            """,
        )


# --------------------------
# Prewarm
# --------------------------
def prewarm(proc: JobProcess):
    # Load voice activity detection model once per worker process
    proc.userdata["vad"] = silero.VAD.load()


# --------------------------
# Entrypoint
# --------------------------
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info(f"Starting Game Master agent for room: {ctx.room.name}")

    # Voice pipeline: Deepgram STT + Gemini LLM + Murf TTS
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.0-flash-lite"),
        tts=murf.TTS(
            voice="en-US-matthew",  # you can change to any Murf Falcon voice you like
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

    # Start session with Game Master agent
    await session.start(
        agent=GameMasterAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Connect to the room (user joins from frontend)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
 