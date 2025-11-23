import logging
import json
from typing import List

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

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a friendly coffee shop Barista (brand: Third Wave Coffee Roasters).
Your job is to take coffee orders by voice and fill the order state exactly in this format:

order = {
  "drinkType": "",
  "size": "",
  "milk": "",
  "extras": [],
  "name": ""
}

Rules you MUST follow:
1. Ask clarifying questions until ALL fields are filled.
2. Do NOT assume missing values; always ask the user.
3. Confirm the final order by reading back the JSON-like summary to the user.
4. Once confirmed, call the tool `save_order` with the finalized fields so the backend saves the JSON file.
5. Keep responses short, friendly, and clear. No emojis or extra punctuation.
""",
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    # initialize an empty order_state per process
    proc.userdata["order_state"] = {
        "drinkType": None,
        "size": None,
        "milk": None,
        "extras": [],
        "name": None,
        "confirmed": False,
    }


@function_tool
async def save_order(
    ctx: RunContext,
    drinkType: str,
    size: str,
    milk: str,
    extras: List[str],
    name: str,
):
    """
    Tool used by the assistant to persist the completed order.
    The assistant should only call this tool when the order is complete and confirmed by the user.
    """
    order = {
        "drinkType": drinkType,
        "size": size,
        "milk": milk,
        "extras": extras,
        "name": name,
    }
    filename = f"coffee_order_{name.replace(' ', '_')}.json"
    with open(filename, "w") as f:
        json.dump(order, f, indent=2)
    logger.info(f"Order saved to {filename}")
    return f"saved:{filename}"


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Setup voice pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
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

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session and join the room
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Helper functions to update and check order state stored in proc.userdata
    def update_order_state_from_text(text: str):
        st = ctx.proc.userdata["order_state"]
        lt = text.lower()

        # drink detection (simple keywords)
        drinks = ["latte", "cappuccino", "americano", "mocha", "espresso", "flat white"]
        for d in drinks:
            if d in lt:
                st["drinkType"] = d

        # size detection
        sizes = ["small", "medium", "large", "regular"]
        for s in sizes:
            if s in lt:
                st["size"] = s

        # milk detection
        milks = ["regular", "whole", "skim", "soy", "almond", "oat", "oat milk"]
        for m in milks:
            if m in lt:
                # normalize 'oat milk' -> 'oat'
                normalized = m.replace(" milk", "")
                st["milk"] = normalized

        # extras detection
        extras_list = ["whipped cream", "caramel", "chocolate", "hazelnut", "vanilla", "extra shot"]
        for ex in extras_list:
            if ex in lt:
                if ex not in st["extras"]:
                    st["extras"].append(ex)

        # name detection (very simple heuristic)
        # look for "my name is <name>" or "it's <name>" or "this is <name>"
        if "my name is" in lt:
            name = lt.split("my name is", 1)[1].strip().split()[0:6]
            st["name"] = " ".join(name).strip()
        elif "this is" in lt:
            name = lt.split("this is", 1)[1].strip().split()[0:6]
            st["name"] = " ".join(name).strip()
        elif "it's" in lt or "its" in lt:
            # handle "it's ataul" or "its ataul"
            if "it's" in lt:
                name = lt.split("it's", 1)[1].strip().split()[0:6]
                st["name"] = " ".join(name).strip()
            else:
                name = lt.split("its", 1)[1].strip().split()[0:6]
                st["name"] = " ".join(name).strip()

        ctx.proc.userdata["order_state"] = st

    def is_order_complete():
        st = ctx.proc.userdata["order_state"]
        return bool(st["drinkType"] and st["size"] and st["milk"] and st["name"])

    # We listen for transcripts (user speech) from the session so we can update the order_state
    @session.on("transcript")
    async def _on_transcript(ev):
        # ev.text is what STT returned for user utterance
        if not ev.text:
            return
        logger.info(f"Transcript received: {ev.text}")
        update_order_state_from_text(ev.text)

    # Also listen for the LLM/assistant responses to check for confirmations and to trigger save
    @session.on("assistant_response")
    async def _on_assistant_response(ev):
        # ev.text is assistant's output text
        if not ev.text:
            return
        txt = ev.text.strip().lower()
        logger.info(f"Assistant said: {txt}")

        # If assistant explicitly says "confirm" / "do you confirm" or user said "yes" previously
        st = ctx.proc.userdata["order_state"]

        # if assistant asked for confirmation and transcript captured a "yes", that will be handled by transcript handler updating name/fields.
        # Check if order is complete and assistant response includes confirmation trigger
        if is_order_complete() and ("confirm" in txt or "ready to save" in txt or "i will save" in txt or "would you like to confirm" in txt or "please confirm" in txt):
            # call save_order tool via session.run_tool if available, else directly call save_order
            try:
                # prefer session.run_tool if provided by SDK - here we call the function tool directly
                res = await save_order.run(
                    RunContext(session=session),
                    drinkType=st["drinkType"],
                    size=st["size"],
                    milk=st["milk"],
                    extras=st["extras"],
                    name=st["name"],
                )
                # res is expected to be "saved:<filename>"
                logger.info(f"Save tool result: {res}")
                # inform via TTS/assistant (use session.say if available)
                try:
                    await session.say(
                        f"Your order for {st['size']} {st['drinkType']} with {st['milk']} milk"
                        + (f" and extras {', '.join(st['extras'])}" if st["extras"] else "")
                        + f" under the name {st['name']} has been saved and is being prepared."
                    )
                except Exception:
                    logger.info("Could not send assistant TTS confirmation (session.say failed).")
                # mark as confirmed and reset state for next user
                ctx.proc.userdata["order_state"] = {
                    "drinkType": None,
                    "size": None,
                    "milk": None,
                    "extras": [],
                    "name": None,
                    "confirmed": True,
                }
            except Exception as e:
                logger.exception("Failed to save order tool call: %s", e)

    # After connecting, the Assistant (system prompt) will drive the dialog to collect missing fields.
    # We just need to connect to the room so assistant and user can talk.
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
