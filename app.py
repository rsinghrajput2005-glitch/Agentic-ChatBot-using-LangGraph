import os
import uuid
import tempfile

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage

from backend import (
    chatbot,
    ingest_rag_document,
    retrieve_all_threads,
    delete_thread,
    DB_PATH,
)

st.set_page_config(page_title="Agentic Chatbot", page_icon="🤖", layout="wide")

TOOL_LABELS = {
    "rag_tool": "📄 Searching your PDF",
    "tavily_search": "🌐 Searching the web",
    "calculator": "🧮 Calculating",
    "get_current_weather": "⛅ Fetching weather",
}


# ---------- helpers ----------
def load_messages(thread_id: str):
    state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", []) if state and state.values else []


def to_ui_history(messages):
    history = []
    for m in messages:
        if isinstance(m, HumanMessage):
            history.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage) and m.content:
            history.append({"role": "assistant", "content": m.content})
    return history


def thread_title(thread_id: str) -> str:
    for m in load_messages(thread_id):
        if isinstance(m, HumanMessage):
            text = str(m.content)
            return text[:28] + ("…" if len(text) > 28 else "")
    return "New chat"


def new_chat():
    tid = str(uuid.uuid4())
    st.session_state.thread_id = tid
    st.session_state.history = []
    if tid not in st.session_state.threads:
        st.session_state.threads.append(tid)


def open_chat(tid: str):
    st.session_state.thread_id = tid
    st.session_state.history = to_ui_history(load_messages(tid))


def remove_chat(tid: str):
    delete_thread(tid)
    if tid in st.session_state.threads:
        st.session_state.threads.remove(tid)
    if tid == st.session_state.thread_id:
        if st.session_state.threads:
            open_chat(st.session_state.threads[-1])
        else:
            new_chat()


def remove_all_chats():
    for tid in list(st.session_state.threads):
        delete_thread(tid)
    st.session_state.threads = []
    new_chat()


# ---------- session state ----------
if "indexed_file" not in st.session_state:
    st.session_state.indexed_file = None
if "history" not in st.session_state:
    st.session_state.history = []
if "threads" not in st.session_state:
    st.session_state.threads = retrieve_all_threads()

if "thread_id" not in st.session_state:
    saved_id = st.query_params.get("thread_id")
    if saved_id:  # page refresh: URL keeps the thread
        if saved_id not in st.session_state.threads:
            st.session_state.threads.append(saved_id)
        open_chat(saved_id)
    elif st.session_state.threads:  # fresh start: resume latest chat
        open_chat(st.session_state.threads[-1])
    else:
        new_chat()

# keep the URL in sync with the active thread
st.query_params["thread_id"] = st.session_state.thread_id


# ---------- sidebar ----------
with st.sidebar:
    st.title("🤖 Agentic Chatbot")

    if st.button("➕ New chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.divider()
    st.subheader("📄 PDF knowledge")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded and st.session_state.indexed_file != uploaded.name:
        with st.spinner("Indexing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            try:
                n_chunks = ingest_rag_document(tmp_path)
                st.session_state.indexed_file = uploaded.name
                st.success(f"Indexed {uploaded.name} ({n_chunks} chunks)")
            finally:
                os.remove(tmp_path)
    elif st.session_state.indexed_file:
        st.caption(f"Active: {st.session_state.indexed_file}")
    elif os.path.exists(DB_PATH):
        st.caption("Using previously indexed PDF")
    else:
        st.caption("No PDF indexed yet")

    st.divider()
    st.subheader("💬 Conversations")
    for tid in reversed(st.session_state.threads):
        label = thread_title(tid)
        is_current = tid == st.session_state.thread_id
        col_open, col_del = st.columns([5, 1])
        with col_open:
            if st.button(
                ("● " if is_current else "") + label,
                key=f"thread_{tid}",
                use_container_width=True,
            ):
                open_chat(tid)
                st.rerun()
        with col_del:
            if st.button("🗑", key=f"del_{tid}", help="Delete this chat"):
                remove_chat(tid)
                st.rerun()

    if st.session_state.threads:
        with st.expander("⚠️ Danger zone"):
            confirm = st.checkbox("I want to delete ALL chats")
            if st.button("Delete all chats", disabled=not confirm):
                remove_all_chats()
                st.rerun()


# ---------- main chat ----------
st.header("Chat")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask anything — PDF, web, weather, math…")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        tool_status = st.empty()

        def token_stream():
            for chunk, meta in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                if isinstance(chunk, ToolMessage):
                    label = TOOL_LABELS.get(chunk.name, f"🔧 {chunk.name}")
                    tool_status.caption(f"{label} — done")
                elif (
                    isinstance(chunk, AIMessageChunk)
                    and meta.get("langgraph_node") == "chat_node"
                ):
                    for tc in chunk.tool_call_chunks or []:
                        if tc.get("name"):
                            label = TOOL_LABELS.get(tc["name"], f"🔧 {tc['name']}")
                            tool_status.caption(f"{label}…")
                    if chunk.content:
                        yield chunk.content

        try:
            reply = st.write_stream(token_stream())
        except Exception as e:
            reply = f"⚠️ Something went wrong: {e}"
            st.error(reply)

    st.session_state.history.append({"role": "assistant", "content": reply})