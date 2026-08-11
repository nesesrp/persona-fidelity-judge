import json
from pathlib import Path

CHATS_FILE = Path("chats.json")
MAX_CHATS = 20


def load_chats():
    """Load saved chats from the JSON file."""
    if not CHATS_FILE.exists() or CHATS_FILE.stat().st_size == 0:
        return {}

    with open(CHATS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_chats(chats):
    """Save chats to the JSON file."""
    with open(CHATS_FILE, "w", encoding="utf-8") as file:
        json.dump(chats, file, ensure_ascii=False, indent=2)


def create_chat(chats, title="New Chat"):
    """Create a new chat."""
    if len(chats) >= MAX_CHATS:
        return None

    chat_id = str(len(chats) + 1)

    chats[chat_id] = {
        "title": title,
        "messages": []
    }

    save_chats(chats)

    return chat_id


def add_message(chats, chat_id, role, content):
    """Add a message to a chat."""
    chats[chat_id]["messages"].append({
        "role": role,
        "content": content
    })

    save_chats(chats)


def save_fidelity_results(chats, chat_id, results, average_score):
    """Save per-turn fidelity evaluation results for a chat."""
    chats[chat_id]["fidelity"] = {
        "average_score": average_score,
        "turns": results
    }

    save_chats(chats)


def get_chat(chats, chat_id):
    """Get a specific chat."""
    return chats.get(chat_id)


def list_chats(chats):
    """Display all saved chats."""
    for chat_id, chat in chats.items():
        print(f"{chat_id}. {chat['title']}")


def delete_chat(chats, chat_id):
    """Delete a specific chat."""
    if chat_id in chats:
        del chats[chat_id]
        save_chats(chats)