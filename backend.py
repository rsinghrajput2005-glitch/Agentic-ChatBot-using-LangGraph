import os
import re
import math
import hmac
import hashlib
import secrets
import sqlite3
from typing import Any, TypedDict, Annotated

import requests
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# Hosted embeddings: no torch / local model, so startup is fast and RAM usage is low.
# Needs GOOGLE_API_KEY in your .env (free key from https://aistudio.google.com/apikey)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_ROOT = os.path.join(BASE_DIR, "faiss_db")
USERS_DB_PATH = os.path.join(BASE_DIR, "users.db")
CHAT_DB_PATH = os.path.join(BASE_DIR, "chatbot.db")


def user_index_path(user_id: str) -> str:
    """Each user gets a private FAISS index folder."""
    return os.path.join(FAISS_ROOT, user_id)


def has_pdf(user_id: str) -> bool:
    return os.path.exists(user_index_path(user_id))


def ingest_rag_document(file_path: str, user_id: str) -> int:
    """Index a PDF into this user's private FAISS index (replaces their previous one)."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(user_index_path(user_id))
    return len(chunks)


def get_retriever(user_id: str):
    path = user_index_path(user_id)
    if not os.path.exists(path):
        return None
    vector_store = FAISS.load_local(
        folder_path=path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})


@tool
def rag_tool(query: str, config: RunnableConfig) -> str:
    """
    Retrieve relevant information from the PDF document.

    Use this tool when the user asks factual or conceptual questions
    that may be answered using the stored PDF documents.

    Args:
        query: The question or search query used to retrieve PDF content.
    """
    # Only ever search the PDF belonging to the user who is chatting
    user_id = (config or {}).get("configurable", {}).get("user_id")
    if not user_id:
        return "No PDF document is currently available."

    retriever = get_retriever(user_id)

    if retriever is None:
        return "No PDF document is currently available."

    documents = retriever.invoke(query)

    if not documents:
        return "No relevant information was found in the PDF."

    formatted = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")
        formatted.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )
    return "\n\n".join(formatted)


@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """
    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"


search_tool = TavilySearch(max_results=5, topic="general", search_depth="advanced")


@tool
def get_current_weather(location: str) -> str:
    """
    Get the current real-time weather for a given city or location.

    Args:
        location: City or location name, for example:
                  "Dhaka", "London, UK", or "New York, US".

    Returns:
        A formatted current weather report.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return (
            "Weather API key is missing. "
            "Set the OPENWEATHER_API_KEY environment variable."
        )

    try:
        geo_response = requests.get(
            "https://api.openweathermap.org/geo/1.0/direct",
            params={"q": location, "limit": 1, "appid": api_key},
            timeout=10,
        )
        geo_response.raise_for_status()
        locations: list[dict[str, Any]] = geo_response.json()

        if not locations:
            return f"Could not find the location: {location}"

        latitude = locations[0]["lat"]
        longitude = locations[0]["lon"]
        resolved_name = locations[0].get("name", location)
        country = locations[0].get("country", "")
        state = locations[0].get("state", "")

        weather_response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "lat": latitude,
                "lon": longitude,
                "appid": api_key,
                "units": "metric",
            },
            timeout=10,
        )
        weather_response.raise_for_status()
        data = weather_response.json()

        temperature = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        pressure = data["main"]["pressure"]
        description = data["weather"][0]["description"]
        wind_speed = data.get("wind", {}).get("speed", "N/A")
        visibility_meters = data.get("visibility")
        visibility_km = (
            round(visibility_meters / 1000, 1) if visibility_meters is not None else "N/A"
        )

        parts = [resolved_name]
        if state:
            parts.append(state)
        if country:
            parts.append(country)
        display_location = ", ".join(parts)

        return (
            f"Current weather in {display_location}:\n"
            f"- Condition: {description.title()}\n"
            f"- Temperature: {temperature}°C\n"
            f"- Feels like: {feels_like}°C\n"
            f"- Humidity: {humidity}%\n"
            f"- Pressure: {pressure} hPa\n"
            f"- Wind speed: {wind_speed} m/s\n"
            f"- Visibility: {visibility_km} km"
        )

    except requests.Timeout:
        return "The weather service request timed out. Please try again."
    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response is not None else "unknown"
        if status_code == 401:
            return "The OpenWeather API key is invalid or inactive."
        return f"Weather API returned an HTTP error: {status_code}"
    except requests.RequestException as error:
        return f"Could not connect to the weather service: {error}"
    except (KeyError, TypeError, ValueError) as error:
        return f"Unexpected weather API response: {error}"


tools = [calculator, search_tool, get_current_weather, rag_tool]
tool_llm = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM_PROMPT = (
    "You are a helpful Agentic Chatbot with access to several tools.\n\n"
    "Tool usage instructions:\n"
    "- Use `rag_tool` for ANY question about the PDF or document: summaries, "
    "'first question', explanations, solutions, page content, and so on. "
    "You cannot see the PDF yourself, so you MUST call `rag_tool` first. "
    "Never say that no PDF is available unless `rag_tool` itself returns that. "
    "For broad requests like a summary, call `rag_tool` several times with different "
    "queries (e.g. 'introduction', 'main topics', 'conclusion', 'questions') before answering. "
    "For a specific question number, search for that wording (e.g. 'Question 1', 'Q1').\n"
    "- Use `search_tool` for current events, recent information, or information "
    "that requires an internet search.\n"
    "- Use `calculator` for mathematical calculations. Do not calculate complex "
    "expressions manually when the calculator is available.\n"
    "- Use `get_current_weather` when the user asks about current weather for a location.\n\n"
    "Answer general questions directly when no tool is required. "
    "Do not invent information from the document; base PDF answers only on what "
    "`rag_tool` returns. If `rag_tool` says no PDF is available, then ask the user to upload one. "
    "After receiving a tool result, provide a clear and helpful final answer."
)


def chat_node(state: ChatState, config: RunnableConfig):
    user_id = (config or {}).get("configurable", {}).get("user_id")
    pdf_status = (
        "\n\nStatus: a PDF is currently indexed and available through `rag_tool`."
        if user_id and has_pdf(user_id)
        else "\n\nStatus: no PDF has been indexed yet."
    )
    messages = [SystemMessage(content=SYSTEM_PROMPT + pdf_status), *state["messages"]]
    response = tool_llm.invoke(messages)
    return {"messages": [response]}


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

conn = sqlite3.connect(database=CHAT_DB_PATH, check_same_thread=False)
checkpoint = SqliteSaver(conn)

chatbot = graph.compile(checkpointer=checkpoint)


def retrieve_all_threads(user_id: str) -> list[str]:
    """Thread ids belonging to this user only, ordered oldest -> newest."""
    prefix = f"{user_id}__"
    seen = []
    for cp in checkpoint.list(None):  # newest checkpoint first
        tid = cp.config["configurable"]["thread_id"]
        if tid.startswith(prefix) and tid not in seen:
            seen.append(tid)
    return seen[::-1]


def delete_thread(thread_id: str) -> None:
    """Permanently delete all saved checkpoints for a conversation."""
    try:
        checkpoint.delete_thread(thread_id)  # available in recent langgraph versions
    except AttributeError:
        # Fallback for older versions: delete rows directly
        with checkpoint.lock:
            cur = conn.cursor()
            cur.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            cur.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
            conn.commit()
            cur.close()


# ---------------- simple user accounts ----------------
users_conn = sqlite3.connect(USERS_DB_PATH, check_same_thread=False)
users_conn.execute(
    "CREATE TABLE IF NOT EXISTS users ("
    "username TEXT PRIMARY KEY, salt BLOB NOT NULL, pw_hash BLOB NOT NULL)"
)
users_conn.commit()


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)


def create_user(username: str, password: str) -> tuple[bool, str]:
    """Returns (ok, username_or_error_message)."""
    username = username.strip().lower()
    if not re.fullmatch(r"[a-z0-9]{3,20}", username):
        return False, "Username must be 3-20 letters or numbers (no spaces or symbols)."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    salt = secrets.token_bytes(16)
    try:
        users_conn.execute(
            "INSERT INTO users (username, salt, pw_hash) VALUES (?, ?, ?)",
            (username, salt, _hash_password(password, salt)),
        )
        users_conn.commit()
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    return True, username


def verify_user(username: str, password: str) -> str | None:
    """Returns the normalized username if the credentials are correct, else None."""
    username = username.strip().lower()
    row = users_conn.execute(
        "SELECT salt, pw_hash FROM users WHERE username = ?", (username,)
    ).fetchone()
    if row is None:
        return None
    salt, stored = row
    if hmac.compare_digest(_hash_password(password, salt), stored):
        return username
    return None