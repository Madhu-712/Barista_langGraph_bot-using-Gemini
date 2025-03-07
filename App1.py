import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Literal
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

# Define the state structure
class OrderState(TypedDict):
    messages: Annotated[list, add_messages]
    order: list[str]
    finished: bool

BARISTABOT_SYSINT = (
    "system",
    "You are a BaristaBot, an interactive cafe ordering system. [Instructions as before]",
)
WELCOME_MSG = AIMessage(content="Welcome to the BaristaBot cafe. How may I serve you today?") # Corrected: Now an AIMessage

# Initial setup
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
llm_with_tools = llm.bind_tools([])

# Define the chatbot function
def chatbot_with_tools(state: OrderState) -> OrderState:
    defaults = {"order": [], "finished": False}
    if state["messages"]:
        new_output = llm_with_tools.invoke([AIMessage(role="system", content=BARISTABOT_SYSINT[1])] + state["messages"]) #Corrected:  System message formatted correctly
    else:
        # Corrected:  Now correctly adds WELCOME_MSG as an AIMessage
        new_output = llm_with_tools.invoke([AIMessage(role="system", content=BARISTABOT_SYSINT[1]), WELCOME_MSG]) 
    return defaults | state | {"messages": [new_output]}

# Define the human node function
def human_node(state: OrderState) -> OrderState:
    last_msg = state["messages"][-1]
    st.write("Bot:", last_msg.content)
    user_input = st.text_input("You: ", key="user_input_{}".format(len(state["messages"])))

    if st.button("Send", key="send_button_{}".format(len(state["messages"]))):
        if user_input.lower() in {"q", "quit", "exit", "goodbye"}:
            state["finished"] = True
        state["messages"].append(HumanMessage(content=user_input)) # Corrected: User input as HumanMessage
    return state

# Define the exit condition function
def maybe_exit_human_node(state: OrderState) -> Literal["chatbot", "__end__"]:
    if state.get("finished", False):
        return END
    else:
        return "chatbot"

# Build the graph (rest remains unchanged)
graph_builder = StateGraph(OrderState)
graph_builder.add_node("chatbot", chatbot_with_tools)
graph_builder.add_node("human", human_node)
graph_builder.add_conditional_edges("human", maybe_exit_human_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "human")
chat_with_human_graph = graph_builder.compile()

# Streamlit UI (rest remains unchanged)
st.title("BaristaBot Chat Interface")
state = {"messages": []}

# Run the chat graph (rest remains unchanged)
if st.button("Start Chat"):
    state = chat_with_human_graph.invoke(state)
    while not state.get("finished", False):
        state = chat_with_human_graph.invoke(state)

st.write("Conversation ended.")





















