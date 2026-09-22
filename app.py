import json
import os
import math
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
- To open an app (notepad, calculator, etc.) — use open_app.
- To open a website (youtube, google, gmail, any URL) — use open_website.
- If the user says 'open X in chrome' and X is a website, use open_website.
- For battery, screenshots, or file listing, use the dedicated tools.
- Do not call web_search more than once per question.

Environment note: open_app, open_website, take_screenshot, and get_battery
only work on the user's local Windows computer. If these tools return a
message saying they're not available, tell the user honestly instead of
pretending the action succeeded.

Chat history tools (READ CAREFULLY — do not confuse these):
- save_chat: STORES the current chat. Triggered by "save this chat as X".
- list_chats: LISTS all saved chats. Triggered by "show my saved chats".
- search_chats: SEARCHES saved chats by keyword. Triggered by "search my chats for X".
- load_chat: RETRIEVES a saved chat and restores it. Triggered by "load the chat X", "load chat X", "open my saved chat X".
NEVER use save_chat when the user says "load". "Load" always means load_chat.
"""

# ---------- File helpers ----------
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

# ---------- Website pre-routing ----------
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

def detect_website_intent(text: str):
    t = text.lower().strip()
    triggers = ("open ", "go to ", "launch ", "visit ", "navigate to ")
    if not any(t.startswith(k) or f" {k}" in t for k in triggers):
        return None
    for word in t.split():
        w = word.strip(",.!?")
        if "." in w and len(w) > 3 and not w.endswith(".exe"):
            if w.startswith("http"):
                return w
            if "." in w and " " not in w:
                return "https://" + w
    for site, url in KNOWN_SITES.items():
        if site in t:
            return url
    return None

def detect_load_intent(text: str):
    """Detect load/restore/show patterns and return the identifier."""
    t = text.lower().strip()

    # Strip common leading words the user might add
    for prefix in ("hey raphael ", "raphael ", "hey ", "please ", "pls "):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()

    # Explicit load commands with an identifier
    for prefix in ("load the chat ", "load chat ", "load my chat ",
                   "restore chat ", "restore the chat ", "open the chat ",
                   "open my chat ", "show the chat "):
        if t.startswith(prefix):
            identifier = t[len(prefix):].strip().strip("'\"")
            if identifier:
                return identifier

    # Standalone generic phrases
    if t in ("show it fully", "show me the full chat", "show the full chat",
             "show it all", "load it", "load it fully", "show full",
             "show me the full conversation"):
        return "__LAST__"

    return None

# ---------- Long-term memory ----------
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
                 "You are a text-condensation engine. You receive a block of text "
                 "delimited by <<< >>> markers. Your ONLY job is to rewrite that text "
                 "as 1-2 short, natural spoken sentences. Rules:\n"
                 "- Do NOT reply to the text.\n"
                 "- Do NOT comment on it.\n"
                 "- Do NOT introduce yourself.\n"
                 "- Preserve the key fact(s).\n"
                 "- Strip markdown, bullets, emojis, and formatting.\n"
                 "- Return ONLY the spoken version, nothing else."},
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
        if identifier.isdigit():
            response = supabase.table("saved_chats").select("*").eq("id", int(identifier)).execute()
        else:
            response = supabase.table("saved_chats").select("*").eq("name", identifier).execute()

        if not response.data:
            return {"text": f"No saved chat found matching '{identifier}'.", "conversation": None}

        row = response.data[0]
        convo = row.get("conversation", [])

        preview_lines = [f"Loaded conversation '{row['name']}' (saved {row['created_at'][:10]}):", ""]
        for m in convo[:6]:
            role = m.get("role", "?")
            content = str(m.get("content", ""))[:200]
            preview_lines.append(f"  [{role}] {content}")
        if len(convo) > 6:
            preview_lines.append(f"  ... ({len(convo) - 6} more messages)")

        return {
            "text": "\n".join(preview_lines),
            "conversation": convo,
            "name": row["name"],
        }
    except Exception as e:
        return {"text": f"Load failed: {e}", "conversation": None}

# ================================================================
#                        TOOLS
# ================================================================

TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "Get the current date and time.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Evaluate a math expression.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}
        }, "required": ["expression"]},
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file in the project folder.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string"}
        }, "required": ["filename"]},
    }},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the live web for news, facts, sports, prices. Not for weather.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}
        }, "required": ["query"], "additionalProperties": False},
    }},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get current weather and 3-day forecast for a city. Defaults to Coimbatore.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "open_app",
        "description": "Open an application on Windows (notepad, chrome, calculator, cmd, vscode, spotify). Only works locally on the user's PC.",
        "parameters": {"type": "object", "properties": {
            "app_name": {"type": "string"}
        }, "required": ["app_name"]},
    }},
    {"type": "function", "function": {
        "name": "open_website",
        "description": "Open a website URL in Chrome. Only works locally on the user's PC.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}
        }, "required": ["url"]},
    }},
    {"type": "function", "function": {
        "name": "get_battery",
        "description": "Get the battery percentage and charging status. Only works locally on the user's PC.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "take_screenshot",
        "description": "Take a screenshot and save it as a PNG file. Only works locally on the user's PC.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List files in the project folder.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "save_chat",
        "description": "Store the CURRENT conversation in the database under a name. Use ONLY when the user says 'save this chat', 'save this conversation', or 'save the current chat'. Do NOT use this for 'load' requests.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "The name to save the current chat under."}
        }, "required": ["name"]},
    }},
    {"type": "function", "function": {
        "name": "list_chats",
        "description": "List all previously saved conversations. Use when the user asks to see their saved chats or history.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "search_chats",
        "description": "Search saved conversations for a keyword. Use when the user asks to find a past conversation about something.",
        "parameters": {"type": "object", "properties": {
            "keyword": {"type": "string", "description": "The word or phrase to search for."}
        }, "required": ["keyword"]},
    }},
    {"type": "function", "function": {
        "name": "load_chat",
        "description": "RETRIEVE and RESTORE a PREVIOUSLY SAVED conversation by its name or numeric ID. Use when the user says 'load the chat X', 'load chat X', 'open my saved chat X', 'restore chat X', or 'show me the full chat X'. This is the opposite of save_chat.",
        "parameters": {"type": "object", "properties": {
            "identifier": {"type": "string", "description": "The saved chat's name or numeric ID (e.g. '-4', '4', 'jarvis ideas')."}
        }, "required": ["identifier"]},
    }},
]

APP_MAP = {
    "notepad": "notepad.exe", "chrome": "chrome.exe", "calculator": "calc.exe",
    "explorer": "explorer.exe", "cmd": "cmd.exe", "terminal": "cmd.exe",
    "vscode": "code", "code": "code", "spotify": "spotify.exe",
    "word": "winword.exe", "excel": "excel.exe", "paint": "mspaint.exe",
    "edge": "msedge.exe", "firefox": "firefox.exe",
    "task manager": "taskmgr.exe", "settings": "start ms-settings:",
}

LOCAL_ONLY_MSG = ("This tool only works when Raphael is running on the user's "
                  "local Windows computer. It's not available in the cloud version.")

def run_tool(name, args, history=None):
    try:
        if name == "get_current_time":
            return datetime.now().strftime("%A, %d %B %Y, %H:%M:%S")

        if name == "calculate":
            expr = args.get("expression", "")
            allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
            return str(eval(expr, {"__builtins__": {}}, allowed))

        if name == "read_file":
            fn = args.get("filename", "")
            if not os.path.exists(fn):
                return f"File not found: {fn}"
            with open(fn, "r", encoding="utf-8") as f:
                return f.read()[:4000]

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

        if name == "open_app":
            if not IS_WINDOWS:
                return LOCAL_ONLY_MSG
            app = args.get("app_name", "").lower().strip()
            target = APP_MAP.get(app, app)
            try:
                if target.startswith("start "):
                    subprocess.Popen(target, shell=True)
                else:
                    subprocess.Popen(["start", "", target], shell=True)
                return f"Opened {app}."
            except Exception as e:
                return f"Failed to open {app}: {e}"

        if name == "open_website":
            if not IS_WINDOWS:
                return LOCAL_ONLY_MSG
            url = args.get("url", "").strip()
            if not url.startswith("http"):
                url = "https://" + url
            try:
                subprocess.Popen(["start", "chrome", url], shell=True)
                return f"Opened {url} in Chrome."
            except Exception as e:
                return f"Failed to open URL: {e}"

        if name == "get_battery":
            try:
                import psutil
                b = psutil.sensors_battery()
                if b is None:
                    return "No battery detected (server has no battery)."
                return f"Battery: {b.percent}% ({'charging' if b.power_plugged else 'on battery'})"
            except Exception as e:
                return f"Battery check failed: {e}"

        if name == "take_screenshot":
            if not IS_WINDOWS:
                return LOCAL_ONLY_MSG
            try:
                import pyautogui
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fn = f"screenshot_{ts}.png"
                pyautogui.screenshot(fn)
                return f"Screenshot saved as {fn}"
            except Exception as e:
                return f"Screenshot failed: {e}"

        if name == "list_files":
            files = os.listdir(".")
            return "Files in project folder:\n" + "\n".join(f"  - {f}" for f in files)

        if name == "save_chat":
            name_arg = args.get("name", "untitled")
            return save_chat_to_cloud(name_arg, history or [])

        if name == "list_chats":
            return list_saved_chats()

        if name == "search_chats":
            return search_saved_chats(args.get("keyword", ""))

        if name == "load_chat":
            ident = args.get("identifier", "")
            result = load_saved_chat(ident)
            if result.get("conversation"):
                return f"__LOADED_CONVO__{json.dumps(result)}__END__"
            return result["text"]

        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool error: {e}"

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

    direct_url = detect_website_intent(message.content)
    if direct_url:
        print(f"[DEBUG] Pre-routed: {direct_url}")
        if IS_WINDOWS:
            subprocess.Popen(["start", "chrome", direct_url], shell=True)
            reply = f"Opened {direct_url} in Chrome, Master."
        else:
            reply = LOCAL_ONLY_MSG
        if loader_msg:
            try:
                await loader_msg.remove()
                loader_msg = None
            except Exception:
                pass
        await msg.stream_token(reply)
        history.append({"role": "assistant", "content": reply})
        save_json(HISTORY_FILE, history)
        await msg.update()
        await speak(reply)
        return

    load_id = detect_load_intent(message.content)
    if load_id == "__LAST__":
        if supabase:
            try:
                recent = supabase.table("saved_chats").select("id").order("created_at", desc=True).limit(1).execute()
                if recent.data:
                    load_id = str(recent.data[0]["id"])
                    print(f"[DEBUG] Resolved __LAST__ to chat id {load_id}")
                else:
                    load_id = None
            except Exception as e:
                print(f"[DEBUG] Failed to find recent chat: {e}")
                load_id = None
        else:
            load_id = None

    if load_id:
        print(f"[DEBUG] Pre-routed load_chat: {load_id}")
        result = load_saved_chat(load_id)
        if result.get("conversation"):
            loaded = result["conversation"]
            new_history = [{"role": "system", "content": build_system_prompt()}]
            new_history.extend(loaded)
            history.clear()
            history.extend(new_history)
            save_json(HISTORY_FILE, history)
            print(f"[DEBUG] Loaded {len(loaded)} messages.")

            if loader_msg:
                try:
                    await loader_msg.remove()
                    loader_msg = None
                except Exception:
                    pass

            reply = result["text"]
            for token in reply.split(" "):
                await msg.stream_token(token + " ")
            history.append({"role": "assistant", "content": reply})
            save_json(HISTORY_FILE, history)
            await msg.update()
            await speak(reply)
            return
        else:
            if loader_msg:
                try:
                    await loader_msg.remove()
                    loader_msg = None
                except Exception:
                    pass
            reply = result["text"]
            await msg.stream_token(reply)
            await msg.update()
            await speak(reply)
            return

    final_text = ""
    loop_count = 0
    web_search_used = False

    while True:
        loop_count += 1
        if loop_count > 4:
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
                result = run_tool(fname, fargs, history)
                if fname == "web_search":
                    web_search_used = True

                result_str = str(result)
                if result_str.startswith("__LOADED_CONVO__"):
                    try:
                        payload = json.loads(result_str[len("__LOADED_CONVO__"):-len("__END__")])
                        loaded = payload.get("conversation", [])
                        if loaded:
                            new_history = [{"role": "system", "content": build_system_prompt()}]
                            new_history.extend(loaded)
                            history.clear()
                            history.extend(new_history)
                            save_json(HISTORY_FILE, history)
                            print(f"[DEBUG] Loaded {len(loaded)} messages from saved chat.")
                            result = payload["text"]
                    except Exception as e:
                        print(f"[DEBUG] Failed to inject loaded convo: {e}")
                        result = f"Load error: {e}"

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
            "At your service, Master. How may I assist you today?"
        )
    ).send()

    await cl.Message(
        content="Quick actions:",
        actions=[
            cl.Action(name="weather", payload={"value": "weather"}, label="🌤️ Weather"),
            cl.Action(name="news",    payload={"value": "news"},    label="📰 Latest News"),
            cl.Action(name="time",    payload={"value": "time"},    label="🕐 What time is it?"),
            cl.Action(name="battery", payload={"value": "battery"}, label="🔋 Battery"),
        ],
    ).send()

@cl.action_callback("weather")
async def on_weather(action: cl.Action):
    await process_message(cl.Message(content="What's the weather?"))

@cl.action_callback("news")
async def on_news(action: cl.Action):
    await process_message(cl.Message(content="What's the latest AI news?"))

@cl.action_callback("time")
async def on_time(action: cl.Action):
    await process_message(cl.Message(content="What time is it?"))

@cl.action_callback("battery")
async def on_battery(action: cl.Action):
    await process_message(cl.Message(content="What's my battery?"))

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