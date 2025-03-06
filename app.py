

!pip install -qU 'langgraph==0.2.45' 'langchain-google-genai==2.0.4' streamlit

import os
import streamlit as st
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, InjectedState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.tool import ToolMessage
from collections.abc import Iterable
from random import randint


# --- Setup ---
import getpass  # Import getpass

# Check if GOOGLE_API_KEY is already set in the environment
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# If not set, prompt the user to enter it securely
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = getpass.getpass("Enter your Google API Key: ")

# Set the GOOGLE_API_KEY environment variable
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")


# --- State and Instructions ---
class OrderState(TypedDict):
    messages: Annotated[list, add_messages]
    order: list[str]
    finished: bool

BARISTABOT_SYSINT = (
    "system",
    "You are a BaristaBot, an interactive cafe ordering system. [Instructions as before]",
)
WELCOME_MSG = "Welcome to the BaristaBot cafe. How may I serve you today?"

# --- Tools ---
@tool
def get_menu() -> str:
    """Provide the latest up-to-date menu."""
    return """
    MENU:
    [Menu items as before]
  """

@tool
def add_to_order(drink: str, modifiers: Iterable[str]) -> str:
    """Adds the specified drink to the customer's order."""

@tool
def confirm_order() -> str:
    """Asks the customer if the order is correct."""

@tool
def get_order() -> str:
    """Returns the users order so far."""

@tool
def clear_order():
    """Removes all items from the user's order."""

@tool
def place_order() -> int:
    """Sends the order to the barista for fulfillment."""
    return randint(1, 5)  # ETA in minutes

# --- Nodes ---
def human_node(state: OrderState) -> OrderState:
    """Get user input through Streamlit."""
    user_input = st.text_input("User:", key="user_input")
    if user_input in {"q", "quit", "exit", "goodbye"}:
        state["finished"] = True
    return state | {"messages": [("user", user_input)]}


def chatbot_with_welcome_msg(state: OrderState) -> OrderState:
    """The chatbot itself."""
    if state["messages"]:
        new_output = llm_with_tools.invoke([BARISTABOT_SYSINT] + state["messages"])
    else:
        new_output = AIMessage(content=WELCOME_MSG)
    return {"order": [], "finished": False} | state | {"messages": [new_output]}

def order_node(state: OrderState) -> OrderState:
    """The ordering node."""
    tool_msg = state.get("messages", [])[-1]
    order = state.get("order", [])
    outbound_msgs = []
    order_placed = False
    # ... (rest of the order_node logic as before) ...

def maybe_route_to_tools(state: OrderState) -> str:
    """Route between chat and tool nodes."""
    # ... (rest of the maybe_route_to_tools logic as before) ...

def maybe_exit_human_node(state: OrderState) -> Literal["chatbot", "__end__"]:
    """Route to the chatbot, unless it looks like the user is exiting."""
    if state.get("finished", False):
        return END
    else:
        return "chatbot"


# --- Graph Setup ---
auto_tools = [get_menu]
tool_node = ToolNode(auto_tools)
order_tools = [add_to_order, confirm_order, get_order, clear_order, place_order]
llm_with_tools = llm.bind_tools(auto_tools + order_tools)

graph_builder = StateGraph(OrderState)
graph_builder.add_node("chatbot", chatbot_with_welcome_msg)
graph_builder.add_node("human", human_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("ordering", order_node)

graph_builder.add_conditional_edges("chatbot", maybe_route_to_tools)
graph_builder.add_conditional_edges("human", maybe_exit_human_node)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("ordering", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile()

# --- Streamlit App ---
if "state" not in st.session_state:
    st.session_state.state = {"messages": []}

if st.session_state.user_input:  
    st.session_state.state = graph.invoke(st.session_state.state, config={"recursion_limit": 100})
    st.session_state.user_input = ""  # Clear input after processing

# Display conversation
st.text_area("Conversation:", value="\n".join([msg.content for msg in st.session_state.state["messages"]]), height=300)

# Display order if any
if st.session_state.state.get("order"):
    st.write("**Current Order:**")
    for item in st.session_state.state["order"]:
        st.write(f"- {item}")
