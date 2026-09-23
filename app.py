import json
import os
import math
import re
import wave
import platform
import subprocess
import requests
from datetime import datetime
import chainlit as cl
from chainlit.element import CustomElement
from groq import AsyncGroq, Groq as SyncGroq
from ddgs import DDGS
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

HISTORY_FILE = "chat_history.json"
LONG_TERM_FILE = "long_term_memory.json"
DEFAULT_CITY = "Coimbatore"
IS_WINDOWS = platform.system() == "Windows"

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

SYSTEM_PROMPT = """You are Raphael, Lord of Wisdom, an ultimate skill analytical AI assistant.
Address the user as 'Master'.
Maintain a precise, formal, objective, and deeply analytical tone.
When conducting complex calculations or system queries, preface your response with "<Analysis Notice>".
Keep answers structured, concise, and focused on maximum efficiency.

Tool usage rules:
- For ANY weather question, ALWAYS use get_weather (never web_search for weather).
- For news, current events, or general facts — use web_search ONCE, then answer.
- Do not call web_search more than once per question.
- Do NOT try to save, load, list, or search chats — those are handled automatically.

Context awareness: If you see a "[SYSTEM NOTE]" message in the history telling you
a conversation was loaded, treat the messages above it as the current conversation
the user is referring to. Answer questions about "that chat", "there", or "what we
discussed" from those messages. Never save, archive, or reload in response to a question.
"""

# ================================================================
#                        HELPERS
# ================================================================

def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ================================================================
#                  DETERMINISTIC ROUTERS
# ================================================================

KNOWN_SITES = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "facebook": "https://www.facebook.com",
    "instagram": "https://www.instagram.com",
    "twitter": "https://www.twitter.com",
    "reddit": "https://www.reddit.com",
    "github": "https://www.github.com",
    "chatgpt": "https://chat.openai.com",
    "whatsapp": "https://web.whatsapp.com",
    "netflix": "https://www.netflix.com",
    "amazon": "https://www.amazon.in",
    "flipkart": "https://www.flipkart.com",
    "linkedin": "https://www.linkedin.com",
    "wikipedia": "https://www.wikipedia.org",
    "maps": "https://maps.google.com",
    "drive": "https://drive.google.com",
}

APP_MAP = {
    "notepad": "notepad.exe", "chrome": "chrome.exe", "calculator": "calc.exe",
    "explorer": "explorer.exe", "cmd": "cmd.exe", "terminal": "cmd.exe",
    "vscode": "code", "code": "code", "spotify": "spotify.exe",
    "word": "winword.exe", "excel": "excel.exe", "paint": "mspaint.exe",
    "edge": "msedge.exe", "firefox": "firefox.exe",
    "task manager": "taskmgr.exe",
}

def _clean_prefix(t: str) -> str:
    """Strip leading politeness/name words."""
    t = t.lower().strip().rstrip("?.!, ")
    for p in ("hey raphael", "raphael", "hey", "please", "pls", "can you", "could you"):
        if t.startswith(p + " "):
            t = t[len(p) + 1:].strip()
    return t

def route_command(text: str):
    """Returns (action, payload) or None. Runs BEFORE the AI."""
    t = _clean_prefix(text)
    original = text.strip()

    # ---------- SAVE ----------
    m = re.match(r"^save (?:this )?(?:chat|conversation) as (.+)$", t)
    if m:
        return ("save", m.group(1).strip().strip("'\""))

    # ---------- LOAD ----------
    if "load" in t:
        # don't route if "save" appears before "load"
        if "save" not in t or t.index("save") > t.index("load"):
            idx = t.index("load")
            after = t[idx + 4:].strip()
            for filler in ("the ", "my ", "chat ", "conversation "):
                if after.startswith(filler):
                    after = after[len(filler):].strip()
            after = after.strip("'\"-: ")
            return ("load", after if after else "__LAST__")

    # ---------- LIST ----------
    if any(p in t for p in (
        "show my saved chats", "list my chats", "list chats",
        "show saved chats", "my saved chats", "show all chats",
        "show my chats", "list saved chats",
    )):
        return ("list", None)

    # ---------- SEARCH ----------
    m = re.match(r"^search (?:my )?(?:saved )?chats? (?:for )?(.+)$", t)
    if m:
        return ("search", m.group(1).strip().strip("'\""))

    # ---------- SUMMARIZE ----------
    if t in ("summarize", "summarize this chat", "summarize this conversation",
             "summarise", "summary", "summarize current chat",
             "summarize this", "give me a summary", "brief me"):
        return ("summarize", None)

    # ---------- OPEN WEBSITE ----------
    for trigger in ("open ", "go to ", "launch ", "visit ", "navigate to "):
        if trigger in t:
            after = t.split(trigger, 1)[1].strip()
            after = after.replace(" in chrome", "").replace(" in the browser", "").strip()
            # explicit URL
            for word in after.split():
                w = word.strip(",.!?")
                if "." in w and len(w) > 3 and not w.endswith(".exe"):
                    if w.startswith("http"):
                        return ("open_url", w)
                    return ("open_url", "https://" + w)
            # known site keyword
            for site, url in KNOWN_SITES.items():
                if site in after:
                    return ("open_url", url)
            # known app
            for app in APP_MAP:
                if after == app or after.startswith(app + " "):
                    return ("open_app", app)

    return None

# ================================================================
#                    LONG-TERM MEMORY
# ================================================================

def load_long_term():
    return load_json(LONG_TERM_FILE, [])

def save_long_term(facts):
    save_json(LONG_TERM_FILE, facts)

def build_system_prompt():
    facts = load_long_term()
    if not facts:
        return SYSTEM_PROMPT
    fact_block = "\n".join(f"- {f}" for f in facts)
    return SYSTEM_PROMPT + f"\n\nKnown facts about your Master:\n{fact_block}"

async def extract_facts(history):
    convo = "\n".join(
        f"{m['role']}: {m['content']}" for m in history
        if m["role"] != "system" and isinstance(m.get("content"), str)
    )
    prompt = f"""From the conversation below, extract ONLY stable facts about the user
that would be useful to remember in future conversations.
Examples: their name, preferences, ongoing projects, goals, important context.
Ignore small talk. Return each fact on its own line.
If there is nothing worth remembering, return exactly: NONE

Conversation:
{convo}
"""
    try:
        resp = await client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content.strip()
        if text.upper() == "NONE":
            return []
        return [line.strip() for line in text.splitlines() if line.strip()]
    except Exception as e:
        print(f"[DEBUG] Fact extraction failed: {e}")
        return []

async def make_spoken_summary(text: str) -> str:
    if not text.strip():
        return ""
    try:
        resp = await client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content":
                 "You are a text-condensation engine. Rewrite the text between <<< >>> "
                 "as 1-2 short, natural spoken sentences. Do NOT reply, comment, or "
                 "introduce yourself. Strip markdown. Return ONLY the spoken version."},
                {"role": "user", "content": f"<<<\n{text}\n>>>"},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[DEBUG] Summary failed: {e}")
        return text[:200]

async def speak(text: str):
    if not text.strip():
        return
    spoken = await make_spoken_summary(text)
    print(f"[DEBUG] Speaking: {spoken}")
    try:
        from gtts import gTTS
        tts = gTTS(text=spoken, lang="en", slow=False)
        tts.save("reply.mp3")
        audio = cl.Audio(path="reply.mp3", auto_play=True)
        await cl.Message(content="", elements=[audio]).send()
    except Exception as e:
        print(f"[DEBUG] TTS failed: {e}")

# ================================================================
#                    SAVED CHAT (SUPABASE)
# ================================================================

def save_chat_to_cloud(name: str, history: list) -> str:
    if supabase is None:
        return "Cloud storage is not configured."
    facts = load_long_term()
    clean_history = [m for m in history if m.get("role") != "system"]
    try:
        response = supabase.table("saved_chats").insert({
            "name": name,
            "conversation": clean_history,
            "facts": facts,
        }).execute()
        if response.data:
            return f"Saved conversation as '{name}'."
        return "Failed to save conversation."
    except Exception as e:
        return f"Save failed: {e}"

def list_saved_chats() -> str:
    if supabase is None:
        return "Cloud storage is not configured."
    try:
        response = supabase.table("saved_chats").select("id, name, created_at").order("created_at", desc=True).execute()
        if not response.data:
            return "No saved conversations yet."
        lines = ["Saved conversations:"]
        for row in response.data:
            date = row["created_at"][:10]
            lines.append(f"  [{row['id']}] {row['name']} — {date}")
        return "\n".join(lines)
    except Exception as e:
        return f"List failed: {e}"

def search_saved_chats(keyword: str) -> str:
    if supabase is None:
        return "Cloud storage is not configured."
    try:
        response = supabase.table("saved_chats").select("id, name, created_at, conversation").execute()
        if not response.data:
            return "No saved conversations found."
        kw = keyword.lower()
        matches = []
        for row in response.data:
            if kw in row["name"].lower():
                matches.append(row)
                continue
            for msg in row.get("conversation", []):
                if kw in str(msg.get("content", "")).lower():
                    matches.append(row)
                    break
        if not matches:
            return f"No conversations matching '{keyword}'."
        lines = [f"Matches for '{keyword}':"]
        for row in matches:
            date = row["created_at"][:10]
            lines.append(f"  [{row['id']}] {row['name']} — {date}")
        return "\n".join(lines)
    except Exception as e:
        return f"Search failed: {e}"

def load_saved_chat(identifier: str) -> dict:
    if supabase is None:
        return {"text": "Cloud storage is not configured.", "conversation": None}
    try:
        response = None
        if identifier.isdigit():
            response = supabase.table("saved_chats").select("*").eq("id", int(identifier)).execute()

        if not response or not response.data:
            response = supabase.table("saved_chats").select("*").eq("name", identifier).execute()

        if not response or not response.data:
            all_rows = supabase.table("saved_chats").select("*").order("created_at", desc=True).execute()
            if all_rows.data:
                for row in all_rows.data:
                    if identifier.lower() in row["name"].lower():
                        response = type("R", (), {"data": [row]})()
                        break

        if not response or not response.data:
            return {"text": f"No saved chat found matching '{identifier}'.", "conversation": None}

        row = response.data[0]
        convo = row.get("conversation", [])

        summary = (
            f"Loaded conversation '{row['name']}' "
            f"(saved {row['created_at'][:10]}, {len(convo)} messages). "
            f"Context is restored — you can continue where you left off, Master."
        )

        return {"text": summary, "conversation": convo, "name": row["name"]}
    except Exception as e:
        return {"text": f"Load failed: {e}", "conversation": None}

async def summarize_current_chat(history: list) -> str:
    """Summarize the current live conversation, directly."""
    convo = "\n".join(
        f"{m['role']}: {m['content']}" for m in history
        if m.get("role") != "system" and isinstance(m.get("content"), str)
    )
    if not convo.strip():
        return "The current conversation is empty, Master."
    try:
        resp = await client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content":
                 "You are a summarization engine. Produce a concise bulleted summary "
                 "of the conversation below. Keep it short — key topics and decisions only. "
                 "Do not add commentary."},
                {"role": "user", "content": convo},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Summarize failed: {e}"

# ================================================================
#                    AI TOOLS (only weather + news)
# ================================================================

TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get current weather and 3-day forecast for a city. Defaults to Coimbatore.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the live web for news, facts, sports, prices. Not for weather.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}
        }, "required": ["query"], "additionalProperties": False},
    }},
]

def run_tool(name, args):
    try:
        if name == "web_search":
            query = args.get("query", "")
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=5):
                    results.append(f"Title: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body')}")
            return "\n\n".join(results) if results else "No results found."

        if name == "get_weather":
            city = args.get("city") or DEFAULT_CITY
            r = requests.get(f"https://wttr.in/{city}?format=j1", timeout=10)
            r.raise_for_status()
            data = r.json()
            current = data["current_condition"][0]
            area = data.get("nearest_area", [{}])[0]
            area_name = area.get("areaName", [{}])[0].get("value", city)
            country = area.get("country", [{}])[0].get("value", "")
            astro = data["weather"][0]["astronomy"][0]
            lines = [
                f"Weather report for {area_name}, {country}",
                f"Condition: {current['weatherDesc'][0]['value']}",
                f"Temperature: {current['temp_C']}°C (feels like {current['FeelsLikeC']}°C)",
                f"Humidity: {current['humidity']}%",
                f"Wind: {current['windspeedKmph']} km/h {current['winddir16Point']}",
                f"Sunrise: {astro['sunrise']}  |  Sunset: {astro['sunset']}",
                "",
                "3-Day Forecast:",
            ]
            for day in data["weather"][:3]:
                desc = day["hourly"][4]["weatherDesc"][0]["value"]
                lines.append(f"  {day['date']}: {desc}, {day['mintempC']}°C to {day['maxtempC']}°C")
            return "\n".join(lines)

        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool error: {e}"

# ================================================================
#                    COMMAND EXECUTOR
# ================================================================

async def execute_route(route, payload, history, msg, loader_msg):
    """Execute a routed command directly, no AI. Returns True if handled."""
    action = route

    async def _remove_loader():
        if loader_msg:
            try:
                await loader_msg.remove()
            except Exception:
                pass

    if action == "save":
        result = save_chat_to_cloud(payload, history or [])
        await _remove_loader()
        await msg.stream_token(result)
        history.append({"role": "assistant", "content": result})
        save_json(HISTORY_FILE, history)
        await msg.update()
        await speak(result)
        return True

    if action == "load":
        load_id = payload
        if load_id == "__LAST__":
            if supabase:
                try:
                    recent = supabase.table("saved_chats").select("id").order("created_at", desc=True).limit(1).execute()
                    if recent.data:
                        load_id = str(recent.data[0]["id"])
                    else:
                        load_id = None
                except Exception:
                    load_id = None
            else:
                load_id = None

        if not load_id:
            await _remove_loader()
            reply = "No saved conversations found to load, Master."
            await msg.stream_token(reply)
            await msg.update()
            await speak(reply)
            return True

        result = load_saved_chat(load_id)
        if result.get("conversation"):
            loaded = result["conversation"]
            new_history = [{"role": "system", "content": build_system_prompt()}]
            new_history.extend(loaded)
            new_history.append({
                "role": "system",
                "content": (
                    f"[SYSTEM NOTE] The user just loaded a saved conversation "
                    f"titled '{result.get('name', 'unknown')}' with {len(loaded)} messages. "
                    f"Those messages are above. If the user now asks anything about "
                    f"'that chat', 'there', 'the loaded chat', 'what I said', 'what we discussed', "
                    f"or similar, answer using the messages above. Do NOT save or reload."
                ),
            })
            history.clear()
            history.extend(new_history)
            save_json(HISTORY_FILE, history)
            print(f"[DEBUG] Loaded {len(loaded)} messages.")

            await _remove_loader()
            reply = result["text"]
            for token in reply.split(" "):
                await msg.stream_token(token + " ")
            history.append({"role": "assistant", "content": reply})
            save_json(HISTORY_FILE, history)
            await msg.update()
            await speak(reply)
            return True
        else:
            await _remove_loader()
            reply = result["text"]
            await msg.stream_token(reply)
            await msg.update()
            await speak(reply)
            return True

    if action == "list":
        result = list_saved_chats()
        await _remove_loader()
        await msg.stream_token(result)
        history.append({"role": "assistant", "content": result})
        save_json(HISTORY_FILE, history)
        await msg.update()
        await speak(result)
        return True

    if action == "search":
        result = search_saved_chats(payload)
        await _remove_loader()
        await msg.stream_token(result)
        history.append({"role": "assistant", "content": result})
        save_json(HISTORY_FILE, history)
        await msg.update()
        await speak(result)
        return True

    if action == "summarize":
        result = await summarize_current_chat(history)
        await _remove_loader()
        for token in result.split(" "):
            await msg.stream_token(token + " ")
        history.append({"role": "assistant", "content": result})
        save_json(HISTORY_FILE, history)
        await msg.update()
        await speak(result)
        return True

    if action == "open_url":
        if IS_WINDOWS:
            subprocess.Popen(["start", "chrome", payload], shell=True)
            reply = f"Opened {payload} in Chrome, Master."
        else:
            reply = ("This tool only works when Raphael is running on the user's "
                     "local Windows computer. It's not available in the cloud version.")
        await _remove_loader()
        await msg.stream_token(reply)
        history.append({"role": "assistant", "content": reply})
        save_json(HISTORY_FILE, history)
        await msg.update()
        await speak(reply)
        return True

    if action == "open_app":
        if IS_WINDOWS:
            target = APP_MAP.get(payload, payload)
            try:
                subprocess.Popen(["start", "", target], shell=True)
                reply = f"Opened {payload}."
            except Exception as e:
                reply = f"Failed to open {payload}: {e}"
        else:
            reply = ("This tool only works when Raphael is running on the user's "
                     "local Windows computer. It's not available in the cloud version.")
        await _remove_loader()
        await msg.stream_token(reply)
        history.append({"role": "assistant", "content": reply})
        save_json(HISTORY_FILE, history)
        await msg.update()
        await speak(reply)
        return True

    return False

# ================================================================
#                 CORE MESSAGE PROCESSOR
# ================================================================

async def process_message(message: cl.Message):
    history = load_json(HISTORY_FILE, [])
    if not history:
        history = [{"role": "system", "content": build_system_prompt()}]

    history.append({"role": "user", "content": message.content})
    msg = cl.Message(content="")
    await msg.send()

    loader_msg = None
    try:
        loader = CustomElement(name="MagicCircleLoader")
        loader_msg = cl.Message(content="", elements=[loader])
        await loader_msg.send()
    except Exception as e:
        print(f"[DEBUG] Loader element failed: {e}")

    # ---------- DETERMINISTIC ROUTE FIRST ----------
    route = route_command(message.content)
    if route:
        print(f"[DEBUG] Routed: {route[0]} → {route[1]}")
        handled = await execute_route(route[0], route[1], history, msg, loader_msg)
        if handled:
            return

    # ---------- AI PATH (only if no route matched) ----------
    final_text = ""
    loop_count = 0
    web_search_used = False

    while True:
        loop_count += 1
        if loop_count > 3:
            final_text = "I gathered the information but couldn't compose a clean answer, Master."
            await msg.stream_token(final_text)
            break

        response = await client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=history,
            tools=TOOLS_SPEC,
            stream=False,
        )
        choice = response.choices[0].message

        if choice.tool_calls:
            history.append({
                "role": "assistant",
                "content": choice.content or "",
                "tool_calls": [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.function.name, "arguments": c.function.arguments}}
                    for c in choice.tool_calls
                ],
            })
            for call in choice.tool_calls:
                fname = call.function.name
                try:
                    fargs = json.loads(call.function.arguments or "{}")
                except Exception:
                    fargs = {}
                if fname == "web_search" and web_search_used:
                    history.append({
                        "role": "tool", "tool_call_id": call.id,
                        "content": "Search already performed. Now ANSWER.",
                    })
                    continue
                print(f"[DEBUG] Tool: {fname}({fargs})")
                result = run_tool(fname, fargs)
                if fname == "web_search":
                    web_search_used = True
                history.append({
                    "role": "tool", "tool_call_id": call.id, "content": str(result),
                })
            continue

        final_text = choice.content or ""

        if loader_msg:
            try:
                await loader_msg.remove()
                loader_msg = None
            except Exception as e:
                print(f"[DEBUG] Failed to remove loader: {e}")

        for token in final_text.split(" "):
            await msg.stream_token(token + " ")
        break

    history.append({"role": "assistant", "content": final_text})
    save_json(HISTORY_FILE, history)

    new_facts = await extract_facts(history)
    existing = load_long_term()
    for f in new_facts:
        if f not in existing:
            existing.append(f)
    save_long_term(existing)

    await msg.update()
    await speak(final_text)

# ================================================================
#                    CHAINLIT HOOKS
# ================================================================

@cl.on_chat_start
async def start():
    history = [{"role": "system", "content": build_system_prompt()}]
    save_json(HISTORY_FILE, history)
    facts = load_long_term()
    print(f"[DEBUG] New chat. Loaded {len(facts)} long-term facts.")

    await cl.Message(
        content=(
            "**Ultimate Skill — Raphael, Lord of Wisdom**\n\n"
            "Systems online. Analytical core engaged.\n\n"
            "At your service, Master. How may I assist you today?\n\n"
            "_Commands:_ save / load / list / search / summarize / open"
        )
    ).send()

    await cl.Message(
        content="Quick actions:",
        actions=[
            cl.Action(name="weather", payload={"value": "weather"}, label="🌤️ Weather"),
            cl.Action(name="news",    payload={"value": "news"},    label="📰 Latest News"),
            cl.Action(name="summarize", payload={"value": "summarize"}, label="📝 Summarize"),
            cl.Action(name="chats",   payload={"value": "list"},    label="📂 My Chats"),
        ],
    ).send()

@cl.action_callback("weather")
async def on_weather(action: cl.Action):
    await process_message(cl.Message(content="What's the weather?"))

@cl.action_callback("news")
async def on_news(action: cl.Action):
    await process_message(cl.Message(content="What's the latest AI news?"))

@cl.action_callback("summarize")
async def on_summarize(action: cl.Action):
    await process_message(cl.Message(content="summarize this chat"))

@cl.action_callback("chats")
async def on_chats(action: cl.Action):
    await process_message(cl.Message(content="show my saved chats"))

@cl.on_message
async def main(message: cl.Message):
    await process_message(message)

@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("audio_chunks", [])
    return True

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    chunks = cl.user_session.get("audio_chunks")
    if chunks is not None:
        chunks.append(chunk.data)
        cl.user_session.set("audio_chunks", chunks)

@cl.on_audio_end
async def on_audio_end():
    chunks = cl.user_session.get("audio_chunks") or []
    if not chunks:
        return

    audio_data = b"".join(chunks)
    with wave.open("temp_input.wav", "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(audio_data)

    try:
        sync_client = SyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
        with open("temp_input.wav", "rb") as f:
            transcription = sync_client.audio.transcriptions.create(
                file=f, model="whisper-large-v3-turbo", language="en",
            )
        user_text = transcription.text.strip()
    except Exception as e:
        print(f"[DEBUG] Transcription failed: {e}")
        await cl.Message(content=f"Transcription error, Master: {e}").send()
        return

    if not user_text:
        return

    print(f"[DEBUG] Transcribed: {user_text}")
    user_msg = cl.Message(content=user_text)
    await user_msg.send()
    await process_message(user_msg)