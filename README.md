# LLM-as-a-Judge

A sandbox for testing persona-driven chatbots with an automated LLM judge. A simulated student has a full conversation with the chatbot, which is an AI programming teacher persona, and a separate LLM then scores how faithfully the chatbot stuck to its system prompt and how consistently it used the conversation context.

## How it works

1. **Simulated user** — an LLM plays a curious student (Turkish, casual tone), asking questions, occasionally testing the chatbot's memory, and sometimes changing topics abruptly. It ends the conversation on its own by emitting a stop tag.
2. **Persona chatbot** — an LLM responds to the simulated student under a detailed system prompt: stay in character, never break persona, match the user's language, react naturally to topic changes, etc.
3. **Judge** — after the conversation ends, a separate LLM evaluates:
   - **Overall fidelity**: how well the whole conversation followed the persona's system prompt, with a 0–100 score and examples.
   - **Per-turn fidelity**: for each assistant reply, whether it correctly used prior context, avoided contradictions/fabrication, and stayed continuous — scored 0–100 with a JSON breakdown of issues.
4. Conversations and their fidelity scores are saved locally to `chats.json`.

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Orchestrates a simulated chat session end-to-end and triggers evaluation |
| `judge.py` | Prompts and functions for the simulated user, the persona LLM calls, and the fidelity judges |
| `chat_manager.py` | Loads/saves chats and fidelity results to `chats.json` |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install openai python-dotenv
```

Create a `.env` file with your [OpenRouter](https://openrouter.ai/) API key:

```
OPENROUTER_API_KEY=your_key_here
```

## Usage

```bash
python main.py
```

This starts a new simulated conversation, runs it to completion (or up to `MAX_MESSAGES` turns), prints the transcript, then prints the overall fidelity evaluation and the average per-turn fidelity score.

## Models

Configured via OpenRouter in `main.py` / `judge.py`:
- Persona chat model: `deepseek/deepseek-v4-flash-0731`
- Simulated user / judge model: `google/gemini-3.6-flash`
