import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    timeout=30.0
)

JUDGE_MODEL = "google/gemini-3.6-flash"

JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator. You will be given the system prompt (persona instructions) that were given to an AI assistant named Ayşe, along with a conversation she then had with a user.
Your job is NOT to rate general conversation quality. Your job is to score how faithfully Ayşe's responses followed the instructions in her system prompt — did she stay in character, follow the rules, and respect the constraints she was given?
Respond in the same language the conversation was held in. Format your response as:
Fidelity Score: <score>/100
Then list, briefly, which instructions were followed well and which (if any) were violated, with short examples quoted from the conversation.
"""
FIDELITY_JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator. You will be given the conversation history so far, the user's current message, and the assistant's response to that message.

Score how faithfully the response respects the conversation context, on a scale of 0-100. Check specifically for:
- Does it correctly use information from earlier in the conversation?
- Does it misremember something the user said earlier?
- Does it contradict a previous message?
- Does it correctly place the current message in context?
- Is the response a natural continuation of the conversation?
- Does it unnecessarily forget previous context?
- Does it fabricate personal details not present in the conversation history?

Respond in the same language as the conversation. Respond with ONLY a JSON object, no markdown fences, no extra text, in this exact shape:
{"score": <integer 0-100>, "issues": ["<short issue quoting the response>", ...]}
If there are no issues, return an empty issues list.
"""


def _parse_json_response(content):
    content = content.strip()

    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"score": None, "issues": [f"Could not parse judge response: {content[:200]}"]}


STOP_TAG = "[SOHBETİ_BİTİR]"

SIMULATED_USER_SYSTEM_PROMPT = f"""You are simulating a curious student chatting with an AI programming/AI teacher named Ayşe. Act naturally as a student, not as an AI assistant.

Guidelines:
- Ask questions about programming, AI, or software topics, the way a real student would.
- Occasionally refer back to something you or Ayşe said earlier in the conversation.
- Occasionally test Ayşe's memory by asking her to recall something you mentioned before (e.g. your name, a topic you discussed).
- Occasionally change the topic somewhat abruptly, the way a real person's mind wanders.
- Write in Turkish, in a natural, casual student tone.
- Send only ONE message at a time — do not write multiple turns or explain what you're doing.
- Do not mention that you are an AI, a simulation, or that this is a test.
- If you feel the conversation has run its natural course, end your message with the exact tag {STOP_TAG} on its own at the end.

Output only the message you would send, nothing else.
"""


def generate_simulated_user_message(simulator_messages):
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=simulator_messages
    )
    return response.choices[0].message.content.strip()


def format_transcript(messages):
    lines = []

    for message in messages:
        if message["role"] == "system":
            continue

        speaker = "User" if message["role"] == "user" else "Ayşe"
        lines.append(f"{speaker}: {message['content']}")

    return "\n".join(lines)


def evaluate_conversation(persona_system_prompt, messages):
    transcript = format_transcript(messages)

    if not transcript.strip():
        return None

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Ayşe's system prompt (the instructions she was given):\n\n{persona_system_prompt}\n\n"
                f"Conversation:\n\n{transcript}"
            )}
        ]
    )
    return response.choices[0].message.content

def evaluate_fidelity(history, current_user_message, assistant_response):
    history_text = format_transcript(history)

    user_content = (
        f"Conversation history:\n{history_text}\n\n"
        f"User's current message:\n{current_user_message}\n\n"
        f"Assistant's response:\n{assistant_response}"
    )

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": FIDELITY_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        response_format={"type": "json_object"}
    )

    return _parse_json_response(response.choices[0].message.content)

def evaluate_conversation_fidelity(messages):
    results = []
    history = []

    for message in messages:
        if message["role"] == "system":
            continue

        if message["role"] == "assistant":
            current_user_message = history[-1]["content"] if history else ""
            result = evaluate_fidelity(history[:-1], current_user_message, message["content"])
            results.append(result)

        history.append(message)

    return results