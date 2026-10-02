import os
import math
import sqlite3
from typing import Any, TypedDict, Annotated

import requests
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
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
DB_PATH = os.path.join(BASE_DIR, "faiss_db")
CHAT_DB_PATH = os.path.join(BASE_DIR, "chatbot.db")


def ingest_rag_document(file_path: str) -> int:
    """Index a PDF into FAISS (replaces the previous index). Returns chunk count."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH)
    return len(chunks)


def get_retriever():
    if not os.path.exists(DB_PATH):
        return None
    vector_store = FAISS.load_local(
        folder_path=DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the PDF document.

    Use this tool when the user asks factual or conceptual questions
    that may be answered using the stored PDF documents.

    Args:
        query: The question or search query used to retrieve PDF content.
    """
    # Load lazily so a newly uploaded PDF is picked up without restarting
    retriever = get_retriever()

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
    "- Use `rag_tool` for questions about the uploaded PDF or document. "
    "Always retrieve relevant document content before answering PDF-related questions.\n"
    "- Use `search_tool` for current events, recent information, or information "
    "that requires an internet search.\n"
    "- Use `calculator` for mathematical calculations. Do not calculate complex "
    "expressions manually when the calculator is available.\n"
    "- Use `get_current_weather` when the user asks about current weather for a location.\n\n"
    "Answer general questions directly when no tool is required. "
    "Do not invent information from the uploaded document. "
    "If the user asks about a PDF but no document is available, ask them to upload a PDF. "
    "After receiving a tool result, provide a clear and helpful final answer."
)


def chat_node(state: ChatState):
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
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


def retrieve_all_threads() -> list[str]:
    """Thread ids ordered oldest -> newest."""
    seen = []
    for cp in checkpoint.list(None):  # newest checkpoint first
        tid = cp.config["configurable"]["thread_id"]
        if tid not in seen:
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