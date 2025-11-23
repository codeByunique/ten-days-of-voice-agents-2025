# /mnt/data/barista_agent.py
import logging
import json
from typing import List, Optional
from datetime import datetime
from pathlib import Path

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
logger = logging.getLogger("barista_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

load_dotenv(".env.local")

# --------------------------
# Orders folder (ensure exists)
# --------------------------
# Use backend/orders path (relative to working directory)
ORDERS_DIR = Path("orders")
ORDERS_DIR.mkdir(parents=True, exist_ok=True)
ALL_ORDERS_FILE = ORDERS_DIR / "all_orders.json"

# initialize all_orders if not exists
if not ALL_ORDERS_FILE.exists():
    with open(ALL_ORDERS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2, ensure_ascii=False)

# --------------------------
# Per-room orders storage (in-memory)
# --------------------------
# keys: room_name -> order dict
ORDERS = {}

def _get_room_name_from_context(context):
    # Try to extract room name from context; fallback to "default"
    try:
        room = getattr(context, "room", None)
        if room is not None:
            return getattr(room, "name", None) or "default"
    except Exception:
        pass
    return "default"

def get_order_for_context(context):
    room_name = _get_room_name_from_context(context)
    if room_name not in ORDERS:
        ORDERS[room_name] = {
            "drinkType": None,
            "size": None,
            "milk": None,
            "extras": [],
            "name": None,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
    return ORDERS[room_name], room_name

# --------------------------
# Tools to be called by LLM
# --------------------------
@function_tool
async def reset_order(context: RunContext):
    """
    Reset the in-memory order for this room/session.
    LLM should CALL this at the start of a new order.
    """
    _, room_name = get_order_for_context(context)
    ORDERS[room_name] = {
        "drinkType": None,
        "size": None,
        "milk": None,
        "extras": [],
        "name": None,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    logger.info(f"[{room_name}] reset_order called - new order started")
    return f"Order reset for room {room_name}."

@function_tool
async def update_order(
    context: RunContext,
    drinkType: Optional[str] = None,
    size: Optional[str] = None,
    milk: Optional[str] = None,
    extras: Optional[List[str]] = None,
    name: Optional[str] = None,
):
    """
    Update the per-room order. LLM should call this tool with any fields it has parsed.
    Returns the updated state as JSON string.
    """
    order, room_name = get_order_for_context(context)
    changed = False

    if drinkType:
        order["drinkType"] = drinkType.strip()
        changed = True
        logger.info(f"[{room_name}] update_order: drinkType = {order['drinkType']}")
    if size:
        order["size"] = size.strip()
        changed = True
        logger.info(f"[{room_name}] update_order: size = {order['size']}")
    if milk:
        order["milk"] = milk.strip()
        changed = True
        logger.info(f"[{room_name}] update_order: milk = {order['milk']}")
    if extras is not None:
        order["extras"] = [e.strip() for e in extras]
        changed = True
        logger.info(f"[{room_name}] update_order: extras = {order['extras']}")
    if name:
        order["name"] = name.strip()
        changed = True
        logger.info(f"[{room_name}] update_order: name = {order['name']}")

    if not changed:
        logger.info(f"[{room_name}] update_order called but no fields provided.")

    # return current order for LLM confirmation
    return json.dumps(order, indent=2, ensure_ascii=False)

def _append_to_all_orders(order_entry):
    """
    Append a plain order entry (dict) to ALL_ORDERS_FILE.
    """
    try:
        with open(ALL_ORDERS_FILE, "r", encoding="utf-8") as f:
            all_orders = json.load(f)
    except Exception:
        all_orders = []

    all_orders.append(order_entry)

    with open(ALL_ORDERS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_orders, f, indent=2, ensure_ascii=False)

@function_tool
async def save_order_to_json(context: RunContext):
    """
    Save only TWO files inside backend/orders/:
      1) all_orders.json        -> history of all orders
      2) {customername}_order.json  -> latest order of that customer (lowercase safe)
    """
    order, room_name = get_order_for_context(context)

    # validate required fields
    missing = [k for k in ("drinkType", "size", "milk", "name") if not order.get(k)]
    if missing:
        msg = f"Cannot save: missing fields {missing}"
        logger.info(f"[{room_name}] save_order_to_json: {msg}")
        return msg

    # Ensure orders folder exists
    ORDERS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------
    # 1) Save customer-specific file
    # --------------------------
    # filename is CUSTOMER NAME (lowercase, safe) e.g: ataul_order.json
    safe_name = "".join(c.lower() if c.isalnum() else "_" for c in order["name"])
    customer_file = ORDERS_DIR / f"{safe_name}_order.json"

    try:
        with open(customer_file, "w", encoding="utf-8") as f:
            json.dump(order, f, indent=2, ensure_ascii=False)
        logger.info(f"[{room_name}] Saved customer file: {customer_file}")
    except Exception as e:
        logger.exception(f"[{room_name}] Failed to write customer file {customer_file}")
        return f"Failed to save customer file: {e}"

    # --------------------------
    # 2) Update all_orders.json (history)
    # --------------------------
    entry = dict(order)
    entry["_room"] = room_name
    entry["_saved_at"] = datetime.utcnow().isoformat() + "Z"

    try:
        _append_to_all_orders(entry)
        logger.info(f"[{room_name}] Updated history file: {ALL_ORDERS_FILE}")
    except Exception as e:
        logger.exception(f"[{room_name}] Failed to append to {ALL_ORDERS_FILE}")
        return f"Failed to update history: {e}"

    # --------------------------
    # Reset state for this room after save
    # --------------------------
    ORDERS[room_name] = {
        "drinkType": None,
        "size": None,
        "milk": None,
        "extras": [],
        "name": None,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    return f"Order saved as {customer_file.name}."

# --------------------------
# BARISTA AGENT
# --------------------------
class BaristaAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a friendly coffee shop barista. The customer speaks via voice.
Your task:
1) Ask clarifying questions until you have these fields:
   - drinkType (e.g., Latte, Cappuccino, Americano)
   - size (small, medium, large)
   - milk (regular, almond, soy, oat)
   - extras (optional list: e.g., extra shot, whipped cream)
   - name (customer name)

2) At the start of a new customer's order, CALL the tool `reset_order` to ensure clean state.
3) Whenever you learn any field from the customer, CALL the tool `update_order`
   with the field(s). For example:
   - CALL update_order with {"name":"Ataul","drinkType":"Latte"}
   - CALL update_order with {"extras":["Extra Shot","Caramel Syrup"]}

4) Continue asking for missing fields until all required fields (drinkType, size, milk, name) are filled.

5) When the order is complete, CALL the tool `save_order_to_json`.
   After calling the tool, confirm the order to the customer (short friendly line)
   such as "Got it — Latte (Large) for Ataul. I'll save that now."

Keep responses short, warm and friendly. Avoid long explanations.
            """,
            tools=[reset_order, update_order, save_order_to_json],
        )

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # Logging context for room
    ctx.log_context_fields = {"room": ctx.room.name}

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

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=BaristaAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
