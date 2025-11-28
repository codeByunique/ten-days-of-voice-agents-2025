# backend/src/food_agent.py
import json
import logging
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

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("food_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

load_dotenv(".env.local")

# -------------------------
# Catalog load
# -------------------------
CATALOG_PATH = Path("shared-data/day7_food_catalog.json")
if not CATALOG_PATH.exists():
    raise FileNotFoundError(f"Catalog not found at {CATALOG_PATH.resolve()}")

with open(CATALOG_PATH, "r", encoding="utf-8") as f:
    CATALOG = json.load(f)

ITEMS = {item["id"]: item for item in CATALOG.get("items", [])}
STORE_NAME = CATALOG.get("store_name", "QuickKart Fresh")
CURRENCY = CATALOG.get("currency", "INR")
RECIPES = {r["id"]: r for r in CATALOG.get("recipes", [])}

# -------------------------
# Orders persistence (AUTO CREATE)
# -------------------------
ORDERS_DIR = Path("food_orders")   # auto create folder
ORDERS_DIR.mkdir(parents=True, exist_ok=True)

ALL_ORDERS_FILE = ORDERS_DIR / "all_orders.json"   # auto create file
if not ALL_ORDERS_FILE.exists():
    with open(ALL_ORDERS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2, ensure_ascii=False)

# per-room cart state
CARTS = {}  # room_name -> {"items": {item_id: qty}, "customer_name": str, "address": str}


def _get_room_name(context: RunContext) -> str:
    try:
        room = getattr(context, "room", None)
        if room is not None:
            return getattr(room, "name", None) or "default"
    except Exception:
        pass
    return "default"


def _get_cart(room_name: str) -> dict:
    if room_name not in CARTS:
        CARTS[room_name] = {
            "items": {},         # item_id -> quantity
            "customer_name": None,
            "address": None,
        }
    return CARTS[room_name]


def _find_item_id_by_name(name: str) -> Optional[str]:
    name_l = name.lower()
    # simple name match
    for item_id, item in ITEMS.items():
        if name_l in item["name"].lower():
            return item_id
    return None


def _recipe_items_for_phrase(phrase: str) -> List[str]:
    p = phrase.lower()
    # simple phrase mapping to recipes
    if "peanut butter sandwich" in p or "peanut butter" in p:
        return RECIPES.get("pb_sandwich", {}).get("items", [])
    if "pasta" in p:
        return RECIPES.get("pasta_for_two", {}).get("items", [])
    if "breakfast" in p:
        return RECIPES.get("simple_breakfast", {}).get("items", [])
    return []


# -------------------------
# Tools (LLM-callable)
# -------------------------
@function_tool
async def add_item_tool(
    context: RunContext,
    item_name: str,
    quantity: int = 1,
) -> str:
    """
    Add an item to the cart by human name. Quantity defaults to 1.
    """
    room = _get_room_name(context)
    cart = _get_cart(room)

    item_id = _find_item_id_by_name(item_name)
    if not item_id:
        msg = f"Item not found for name '{item_name}'."
        logger.info(f"[{room}] {msg}")
        return msg

    if quantity < 1:
        quantity = 1

    current_qty = cart["items"].get(item_id, 0)
    cart["items"][item_id] = current_qty + quantity

    item = ITEMS[item_id]
    msg = f"Added {quantity} x {item['name']} to cart."
    logger.info(f"[{room}] {msg} Cart: {cart['items']}")
    return msg


@function_tool
async def remove_item_tool(
    context: RunContext,
    item_name: str,
) -> str:
    """
    Remove an item completely from the cart by name.
    """
    room = _get_room_name(context)
    cart = _get_cart(room)

    item_id = _find_item_id_by_name(item_name)
    if not item_id:
        msg = f"Item not found for name '{item_name}'."
        logger.info(f"[{room}] {msg}")
        return msg

    if item_id in cart["items"]:
        del cart["items"][item_id]
        item = ITEMS[item_id]
        msg = f"Removed {item['name']} from the cart."
    else:
        msg = f"{item_name} is not currently in the cart."

    logger.info(f"[{room}] {msg} Cart: {cart['items']}")
    return msg


@function_tool
async def update_quantity_tool(
    context: RunContext,
    item_name: str,
    quantity: int,
) -> str:
    """
    Set a specific quantity for an item in the cart.
    If quantity <= 0, the item is removed.
    """
    room = _get_room_name(context)
    cart = _get_cart(room)

    item_id = _find_item_id_by_name(item_name)
    if not item_id:
        msg = f"Item not found for name '{item_name}'."
        logger.info(f"[{room}] {msg}")
        return msg

    if quantity <= 0:
        cart["items"].pop(item_id, None)
        item = ITEMS[item_id]
        msg = f"Removed {item['name']} from the cart."
    else:
        cart["items"][item_id] = quantity
        item = ITEMS[item_id]
        msg = f"Set {item['name']} quantity to {quantity}."

    logger.info(f"[{room}] {msg} Cart: {cart['items']}")
    return msg


@function_tool
async def list_cart_tool(context: RunContext) -> str:
    """
    Return a human-readable summary of the current cart.
    """
    room = _get_room_name(context)
    cart = _get_cart(room)

    if not cart["items"]:
        return "Your cart is currently empty."

    lines = []
    total = 0
    for item_id, qty in cart["items"].items():
        item = ITEMS[item_id]
        price = item["price"]
        subtotal = price * qty
        total += subtotal
        lines.append(
            f"{qty} x {item['name']} ({price} {CURRENCY} each) = {subtotal} {CURRENCY}"
        )

    lines.append(f"Total so far: {total} {CURRENCY}")
    summary = " | ".join(lines)
    logger.info(f"[{room}] Cart summary: {summary}")
    return summary


@function_tool
async def add_recipe_tool(
    context: RunContext,
    recipe_phrase: str,
    quantity: int = 1,
) -> str:
    """
    Add multiple items to the cart based on a simple 'ingredients for X' phrase.
    """
    room = _get_room_name(context)
    cart = _get_cart(room)

    recipe_item_ids = _recipe_items_for_phrase(recipe_phrase)
    if not recipe_item_ids:
        msg = "I am not sure which ingredients you need. Please name the items or try a simpler phrase."
        logger.info(f"[{room}] {msg}")
        return msg

    added_items = []
    for item_id in recipe_item_ids:
        if item_id not in ITEMS:
            continue
        current_qty = cart["items"].get(item_id, 0)
        cart["items"][item_id] = current_qty + quantity
        added_items.append(ITEMS[item_id]["name"])

    if not added_items:
        msg = "I could not find matching items in the catalog."
        logger.info(f"[{room}] {msg}")
        return msg

    msg = f"I've added {', '.join(added_items)} to your cart for {recipe_phrase}."
    logger.info(f"[{room}] {msg} Cart: {cart['items']}")
    return msg


@function_tool
async def set_customer_info_tool(
    context: RunContext,
    name: Optional[str] = None,
    address: Optional[str] = None,
) -> str:
    """
    Set basic customer info (name, address) for the current room/cart.
    """
    room = _get_room_name(context)
    cart = _get_cart(room)

    if name:
        cart["customer_name"] = name.strip()
    if address:
        cart["address"] = address.strip()

    logger.info(f"[{room}] Customer info updated: {cart['customer_name']}, {cart['address']}")
    return f"Customer info saved."


@function_tool
async def place_order_tool(context: RunContext) -> str:
    """
    Finalize the order: compute total, create order JSON,
    save to 'food_orders/all_orders.json' + 'food_orders/order_<timestamp>.json'.
    Then clear the cart for this room.
    """
    room = _get_room_name(context)
    cart = _get_cart(room)

    if not cart["items"]:
        return "Your cart is empty. Please add some items before placing an order."

    # Build order items
    order_items = []
    total = 0
    for item_id, qty in cart["items"].items():
        item = ITEMS[item_id]
        price = item["price"]
        subtotal = price * qty
        total += subtotal
        order_items.append(
            {
                "id": item_id,
                "name": item["name"],
                "quantity": qty,
                "price": price,
                "subtotal": subtotal,
            }
        )

    ts = datetime.utcnow().isoformat() + "Z"
    order = {
        "order_id": f"ORDER_{int(datetime.utcnow().timestamp())}",
        "store": STORE_NAME,
        "timestamp": ts,
        "currency": CURRENCY,
        "items": order_items,
        "total": total,
        "customer_name": cart.get("customer_name"),
        "address": cart.get("address"),
        "status": "placed"
    }

    # Save to all_orders.json
    try:
        with open(ALL_ORDERS_FILE, "r", encoding="utf-8") as f:
            all_orders = json.load(f)
    except Exception:
        all_orders = []

    all_orders.append(order)
    with open(ALL_ORDERS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_orders, f, indent=2, ensure_ascii=False)

    # Save individual order file
    order_file = ORDERS_DIR / f"{order['order_id'].lower()}.json"
    with open(order_file, "w", encoding="utf-8") as f:
        json.dump(order, f, indent=2, ensure_ascii=False)

    logger.info(f"[{room}] Order saved: {order_file} total={total} {CURRENCY}")

    # Clear cart for this room
    CARTS[room] = {
        "items": {},
        "customer_name": None,
        "address": None,
    }

    return f"Your order has been placed. Order total is {total} {CURRENCY}."


# -------------------------
# Agent Definition
# -------------------------
class FoodAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"""
You are a friendly food and grocery ordering assistant for a fictional store called {STORE_NAME}.
The user interacts with you by voice.

Your abilities:
- Help the user order items from the catalog (groceries, snacks, prepared food).
- Ask clarifying questions when needed (quantity, size, etc.).
- Maintain a cart in memory for the current conversation.
- Handle higher-level requests like "ingredients for a peanut butter sandwich" by calling add_recipe_tool.
- When the user is done, confirm the final cart and call place_order_tool.

Tools:
- add_item_tool: use when the user says "add X", "I want Y", or mentions a specific item and quantity.
- remove_item_tool: use when the user wants to remove an item.
- update_quantity_tool: use when the user changes quantity.
- list_cart_tool: use when the user asks what's in their cart or you want to confirm.
- add_recipe_tool: use when the user asks for ingredients for something (e.g. sandwich, pasta, breakfast).
- set_customer_info_tool: use to save the user's name or basic address when they provide it.
- place_order_tool: use when the user says things like "that's all", "place my order", or "I'm done".

Conversation style:
1) Greet the user and briefly explain: "I can help you order groceries and simple meal ingredients."
2) Ask what they would like to start with.
3) For each request, call the appropriate tool and then briefly confirm what you did.
4) If the user says something like:
   - "That's all"
   - "I'm done"
   - "Place the order"
   then:
   - Call list_cart_tool to summarize.
   - Ask for confirmation.
   - Then call place_order_tool.
5) Keep responses short, practical, and friendly.
6) Do not mention tools or JSON files directly to the user; those are internal.
            """,
            tools=[
                add_item_tool,
                remove_item_tool,
                update_quantity_tool,
                list_cart_tool,
                add_recipe_tool,
                set_customer_info_tool,
                place_order_tool,
            ],
        )


# -------------------------
# Prewarm & Entrypoint
# -------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info(f"Starting FoodAgent for room: {ctx.room.name}")

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

    usage = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    async def log_usage():
        summary = usage.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=FoodAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
