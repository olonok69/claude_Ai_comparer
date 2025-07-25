# Updated services/chat_service.py

import streamlit as st
from config import SERVER_CONFIG
import uuid
from langchain_core.messages import HumanMessage, AIMessage

# Session state initialization
def init_session():
    defaults = {
        "params": {},
        "current_chat_id": None,
        "current_chat_index": 0,
        "history_chats": get_history(),
        "messages": [],
        "client": None,
        "agent": None,
        "tools": [],
        "prompts": [],  # Add prompts support
        "resources": [],  # Add resources support
        "tool_executions": [],
        "servers": SERVER_CONFIG['mcpServers'],
        "conversation_memory": []  # Add conversation memory tracking
    }
    
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_history():
    if "history_chats" in st.session_state and st.session_state["history_chats"]:
        return st.session_state["history_chats"]
    else:
        chat_id = str(uuid.uuid4())
        new_chat = {'chat_id': chat_id,
                    'chat_name': 'New chat',
                    'messages': []}
        st.session_state["current_chat_index"] = 0
        st.session_state["current_chat_id"] = chat_id
    return [new_chat]

def get_current_chat(chat_id):
    """Get messages for the current chat."""
    for chat in st.session_state["history_chats"]:
        if chat['chat_id'] == chat_id:
            return chat['messages']
    return []

def _append_message_to_session(msg: dict) -> None:
    """
    Append *msg* to the current chat's message list **and**
    keep history_chats in-sync.
    """
    chat_id = st.session_state["current_chat_id"]
    
    # Avoid duplicating messages
    if st.session_state["messages"] and len(st.session_state["messages"]) > 0:
        last_message = st.session_state["messages"][-1]
        # Check if this is a duplicate message
        if (last_message.get("role") == msg.get("role") and 
            last_message.get("content") == msg.get("content") and
            last_message.get("tool") == msg.get("tool")):
            return  # Don't add duplicate
    
    st.session_state["messages"].append(msg)
    
    for chat in st.session_state["history_chats"]:
        if chat["chat_id"] == chat_id:
            chat["messages"] = st.session_state["messages"]     # same list
            # Only rename chat based on user messages, not tool messages
            if (chat["chat_name"] == "New chat" and 
                msg["role"] == "user" and 
                "content" in msg):                 
                chat["chat_name"] = " ".join(msg["content"].split()[:5]) or "Empty"
            break

def create_chat():
    """Create a new chat session."""
    chat_id = str(uuid.uuid4())
    new_chat = {'chat_id': chat_id,
                'chat_name': 'New chat',
                'messages': []}
    
    st.session_state["history_chats"].append(new_chat)
    st.session_state["current_chat_index"] = 0
    st.session_state["current_chat_id"] = chat_id
    st.session_state["messages"] = []  # Clear current messages
    st.session_state["conversation_memory"] = []  # Clear conversation memory
    return new_chat

def switch_chat(chat_id: str):
    """Switch to a different chat and load its context."""
    if chat_id == st.session_state.get("current_chat_id"):
        return  # Already on this chat
        
    st.session_state["current_chat_id"] = chat_id
    st.session_state["messages"] = get_current_chat(chat_id)
    
    # Update current chat index
    for i, chat in enumerate(st.session_state["history_chats"]):
        if chat["chat_id"] == chat_id:
            st.session_state["current_chat_index"] = i
            break

def delete_chat(chat_id: str):
    """Delete a chat from history."""
    if not chat_id: # protection against accidental call
        return

    # 1) Remove from session_state.history_chats
    st.session_state["history_chats"] = [
        c for c in st.session_state["history_chats"]
        if c["chat_id"] != chat_id
    ]

    # 2) Switch current_chat to another one or create new
    if st.session_state["current_chat_id"] == chat_id:
        if st.session_state["history_chats"]:            # if chats still exist
            first = st.session_state["history_chats"][0]
            st.session_state["current_chat_id"] = first["chat_id"]
            st.session_state["current_chat_index"] = 0
            st.session_state["messages"] = first["messages"]
        else:                                            # if all deleted → new empty
            new_chat = create_chat()
            st.session_state["messages"] = new_chat["messages"]
        
        # Clear conversation memory when switching/deleting chats
        st.session_state["conversation_memory"] = []
    return

def get_conversation_summary(max_messages: int = 10):
    """Get a summary of recent conversation for context."""
    if not st.session_state.get("messages"):
        return "This is the start of a new conversation."
    
    recent_messages = st.session_state["messages"][-max_messages:]
    summary_parts = []
    
    for msg in recent_messages:
        if msg["role"] == "user":
            summary_parts.append(f"User: {msg['content'][:100]}...")
        elif msg["role"] == "assistant" and "content" in msg and msg["content"]:
            summary_parts.append(f"Assistant: {msg['content'][:100]}...")
    
    return "\n".join(summary_parts)

def get_clean_conversation_memory():
    """Get conversation memory with only user/assistant content messages for LLM compatibility."""
    conversation_messages = []
    
    if st.session_state.get('current_chat_id'):
        messages = get_current_chat(st.session_state['current_chat_id'])
        
        for msg in messages:
            if msg["role"] == "user" and "content" in msg:
                conversation_messages.append(HumanMessage(content=msg["content"]))
            elif (msg["role"] == "assistant" and 
                  "content" in msg and 
                  msg["content"] and 
                  "tool" not in msg):  # Only regular assistant messages, not tool messages
                conversation_messages.append(AIMessage(content=msg["content"]))
    
    return conversation_messages