# backend/src/fraud_detection_agent.py
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
logger = logging.getLogger("fraud_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

load_dotenv(".env.local")

# -------------------------
# Fake "Database" JSON
# -------------------------
DATA_PATH = Path("shared-data/day6_fraud_cases.json")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Fraud cases file not found at {DATA_PATH.resolve()}")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    FRAUD_CASES = json.load(f)

# room_name -> active case dict
ACTIVE_CASES = {}


def _save_cases_back():
    """Write full fraud cases list back to JSON 'database'."""
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(FRAUD_CASES, f, indent=2, ensure_ascii=False)
    logger.info("Fraud cases written back to JSON DB.")


def _get_room_name(context: RunContext) -> str:
    try:
        room = getattr(context, "room", None)
        if room is not None:
            return getattr(room, "name", None) or "default"
    except Exception:
        pass
    return "default"


def _find_case_by_username(username: str):
    uname = username.strip().lower()
    for case in FRAUD_CASES:
        if case.get("userName", "").strip().lower() == uname:
            return case
    return None


# -------------------------
# Tools
# -------------------------
@function_tool
async def load_case_for_user(context: RunContext, user_name: str) -> str:
    """
    Load fraud case for given user_name from JSON DB and attach it to this room's state.
    If not found, return a message so the agent can handle it.
    """
    room = _get_room_name(context)
    case = _find_case_by_username(user_name)
    if not case:
        logger.info(f"[{room}] No fraud case found for username '{user_name}'")
        return f"No fraud case found for username {user_name}."

    ACTIVE_CASES[room] = case
    logger.info(f"[{room}] Loaded fraud case: {case['caseId']} for user {case['userName']}")
    masked = f"**** {case.get('cardEnding', 'XXXX')}"
    return (
        f"Case loaded for {case['userName']} with card ending {masked}, "
        f"amount {case['amount']} at {case['merchantName']}."
    )


@function_tool
async def verify_security_answer(context: RunContext, answer: str) -> str:
    """
    Check the user's security answer against the stored fake answer.
    Returns 'verified' or 'failed'.
    """
    room = _get_room_name(context)
    case = ACTIVE_CASES.get(room)
    if not case:
        return "no_case_loaded"

    expected = (case.get("securityAnswer") or "").strip().lower()
    given = (answer or "").strip().lower()

    if expected and given and expected == given:
        logger.info(f"[{room}] Security verification PASSED.")
        return "verified"
    else:
        logger.info(f"[{room}] Security verification FAILED. expected={expected} got={given}")
        return "failed"


@function_tool
async def update_case_status(
    context: RunContext,
    new_status: str,
    note: Optional[str] = None,
) -> str:
    """
    Update the fraud case status & outcome note and write back to DB.
    new_status: 'confirmed_safe', 'confirmed_fraud', or 'verification_failed'.
    """
    room = _get_room_name(context)
    active = ACTIVE_CASES.get(room)
    if not active:
        return "no_case_loaded"

    case_id = active.get("caseId")
    # update in in-memory full list
    updated = None
    for case in FRAUD_CASES:
        if case.get("caseId") == case_id:
            case["status"] = new_status
            case["outcomeNote"] = note or ""
            case["updatedAt"] = datetime.utcnow().isoformat() + "Z"
            updated = case
            break

    _save_cases_back()

    if updated:
        logger.info(f"[{room}] Case {case_id} updated -> {new_status}")
        return json.dumps(updated, indent=2, ensure_ascii=False)
    else:
        return "case_not_found_in_db"


# -------------------------
# Agent Persona
# -------------------------
class FraudAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a calm, professional fraud detection representative for a fictional bank called "Nexa Bank".

Your tasks for EACH CALL:
1. Introduce yourself clearly:
   - "This is the fraud monitoring department at Nexa Bank..."
   - Explain that you are calling about a suspicious transaction.
2. Ask for the customer's first name.
   - After they answer, CALL load_case_for_user with that name.
   - If the tool says no case found, apologize and end the call.
3. If a case is loaded:
   - Politely ask a **basic security question** from the case: use the 'securityQuestion' field.
   - After the user answers, CALL verify_security_answer with their answer.
   - If result = "failed":
       - Explain you cannot proceed without verification.
       - CALL update_case_status with new_status="verification_failed" and a short note.
       - End the call politely.
   - If result = "verified":
       - Reassure the customer that this is a routine check.
       - Read out the suspicious transaction details from the active case:
         - amount
         - merchantName
         - masked card (use "card ending XXXX")
         - approximate location and time (location + timestamp text)
       - Clearly ask: "Did you make this transaction? Please answer yes or no."
4. Interpret the yes/no answer:
   - If the user clearly says they DID make it:
       - Treat it as a **confirmed_safe** case.
       - CALL update_case_status with new_status="confirmed_safe" and a short explanation in the note.
       - Explain that no further action is needed and thank them.
   - If the user clearly says they DID NOT make it:
       - Treat it as **confirmed_fraud**.
       - CALL update_case_status with new_status="confirmed_fraud" and a short explanation in the note.
       - Use mock actions only:
         - Example: "We will block this card and raise a dispute for this transaction."
       - Reassure them that they will not be charged while the dispute is reviewed.
5. End the call with a short, clear summary:
   - Mention whether the transaction was marked safe or fraudulent.
   - Remind them to contact the bank if they see any other suspicious activity.

Important safety rules:
- Do NOT ask for full card numbers, PINs, passwords, or OTPs.
- Only use the non-sensitive security question from the case for verification.
- Keep your tone calm and reassuring, even if the user is confused.

Never mention JSON, files, tools, or implementation details to the user.
Only describe the bank/fraud flow in natural language.
            """,
            tools=[load_case_for_user, verify_security_answer, update_case_status],
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
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info(f"Starting FraudAgent for room: {ctx.room.name}")

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
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=FraudAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
