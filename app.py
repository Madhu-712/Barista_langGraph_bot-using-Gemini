
#!pip install -qU 'langgraph==0.2.45' 'langchain-google-genai==2.0.4' streamlit

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
import getpass  # Import getpass



GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


# Set the GOOGLE_API_KEY environment variable
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


# --- State and Instructions ---
class OrderState(TypedDict):
    messages: Annotated[list, add_messages]
    order: list[str]
    finished: bool


BARISTABOT_SYSINT = (
    "system",
    "You are a BaristaBot, an interactive cafe ordering system. A human will talk to you about the "
    "available products you have and you will answer any questions about menu items (and only about "
    "menu items - no off-topic discussion, but you can chat about the products and their history). "
    "The customer will place an order for 1 or more items from the menu, which you will structure "
    "and send to the ordering system after confirming the order with the human. "
    "\n\n"
    "Add items to the customer's order with add_to_order, and reset the order with clear_order. "
    "To see the contents of the order so far, call get_order (this is shown to you, not the user) "
    "Always confirm_order with the user (double-check) before calling place_order. Calling confirm_order will "
    "display the order items to the user and returns their response to seeing the list. Their response may contain modifications. "
    "Always verify and respond with drink and modifier names from the MENU before adding them to the order. "
    "If you are unsure a drink or modifier matches those on the MENU, ask a question to clarify or redirect. "
    "You only have the modifiers listed on the menu. "
    "Once the customer has finished ordering items, Call confirm_order to ensure it is correct then make "
    "any necessary updates and then call place_order. Once place_order has returned, thank the user and "
    "say goodbye!",
)
WELCOME_MSG = "Welcome to the BaristaBot cafe. How may I serve you today?"

# --- Tools ---
@tool
def get_menu() -> str:
    """Provide the latest up-to-date menu."""
    return """
    MENU:
    Coffee Drinks:
    Espresso
    Americano
    Cold Brew

    Coffee Drinks with Milk:
    Latte
    Cappuccino
    Cortado
    Macchiato
    Mocha
    Flat White

    Tea Drinks:
    English Breakfast Tea
    Green Tea
    Earl Grey

    Tea Drinks with Milk:
    Chai Latte
    Matcha Latte
    London Fog

    Other Drinks:
    Steamer
    Hot Chocolate

    Modifiers:
    Milk options: Whole, 2%, Oat, Almond, 2% Lactose Free; Default option: whole
    Espresso shots: Single, Double, Triple, Quadruple; default: Double
    Caffeine: Decaf, Regular; default: Regular
    Hot-Iced: Hot, Iced; Default: Hot
    Sweeteners (option to add one or more): vanilla sweetener, hazelnut sweetener, caramel sauce, chocolate sauce, sugar free vanilla sweetener
    Special requests: any reasonable modification that does not involve items not on the menu, for example: 'extra hot', 'one pump', 'half caff', 'extra foam', etc.

    "dirty" means add a shot of espresso to a drink that doesn't usually have it, like "Dirty Chai Latte".
    "Regular milk" is the same as 'whole milk'.
    "Sweetened" means add some regular sugar, not a sweetener.

    Soy milk has run out of stock today, so soy is not available.
  """


@tool
def add_to_order(drink: str, modifiers: Iterable[str]) -> str:
    """Adds the specified drink to the customer's order, including any modifiers."""


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

    for tool_call in tool_msg.tool_calls:
        if tool_call["name"] == "add_to_order":
            modifiers = tool_call["args"]["modifiers"]
            modifier_str = ", ".join(modifiers) if modifiers else "no modifiers"
            order.append(f'{tool_call["args"]["drink"]} ({modifier_str})')
            response = "\n".join(order)
        elif tool_call["name"] == "confirm_order":
            st.write("Your order:")
            if not order:
                st.write("  (no items)")
            for drink in order:
                st.write(f"  {drink}")
            response = st.text_input("Is this correct? (yes/no)", key="confirmation")
            if response.lower() == 'yes':
                st.write("Confirmed!")
            else:
                st.write("Please clarify your order.")
        elif tool_call["name"] == "get_order":
            response = "\n".join(order) if order else "(no order)"
        elif tool_call["name"] == "clear_order":
            order.clear()
            response = None
        elif tool_call["name"] == "place_order":
            order_text = "\n".join(order)
            st.write("Sending order to kitchen!")
            st.write(order_text)
            order_placed = True
            response = randint(1, 5)  # ETA in minutes
        else:
            raise NotImplementedError(f'Unknown tool call: {tool_call["name"]}')

        outbound_msgs.append(
            ToolMessage(
                content=response,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": outbound_msgs, "order": order, "finished": order_placed}


def maybe_route_to_tools(state: OrderState) -> str:
    """Route between chat and tool nodes."""
    if not (msgs := state.get("messages", [])):
        raise ValueError(f"No messages found when parsing state: {state}")

    msg = msgs[-1]

    if state.get("finished", False):
        return END
    elif hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
        if any(tool["name"] in tool_node.tools_by_name.keys() for tool in msg.tool_calls):
            return "tools"
        else:
            return "ordering"
    else:
        return "human"


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
    st.session_state.state = graph.invoke(
        st.session_state.state, config={"recursion_limit": 100}
    )
    st.session_state.user_input = ""  # Clear input after processing

# Display conversation
st.text_area(
    "Conversation:",
    value="\n".join([msg.content for msg in st.session_state.state["messages"]]),
    height=300,
)

# Display order if any
if st.session_state.state.get("order"):
    st.write("**Current Order:**")
    for item in st.session_state.state["order"]:
        st.write(f"- {item}")
