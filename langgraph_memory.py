import os   
import dotenv
dotenv.load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph


from langgraph.checkpoint.memory import MemorySaver

memory_saver = MemorySaver() #기억저장소, 인메모리, 질문을 기억함

class State(TypedDict):
    messages : Annotated[list, add_messages]

tool = TavilySearchResults(max_results=2)
tools = [tool]
tool_node = ToolNode(tools)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=API_KEY)
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State) -> State:
    result = llm_with_tools.invoke(state["messages"])
    return {"messages": [result]}

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge("tools","chatbot")
graph_builder.add_conditional_edges("chatbot",tools_condition)

graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=memory_saver)

config = {"configurable" : {"thread_id" : "1"}} # 이 쓰레드 아이디 안에서는 대화 맥락 기억함

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit","q","quit"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages" : ("user",user_input)},config):
        for value in event.values():
            print("Assistant: ", value["messages"][-1].content)

#메시지 턴 확인
snapshot = graph.get_state(config)
print(snapshot)

# 기억할 메시지 수 제한하기
def filter_messages(messages: list) -> list:
    return messages[-2:]

def chatbot(state: State) -> State:
    messages = filter_messages(state["messages"]) # 최근 2개의 메시지만 가져옴
    result = llm_with_tools.invoke(messages)
    return {"messages": [result]}

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge("tools","chatbot")
graph_builder.add_conditional_edges("chatbot",tools_condition)

graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=memory_saver)


from langchain_core.messages import HumanMessage


config = {"configurable" : {"thread_id" : "20"}} # 이 쓰레드 아이디 안에서는 대화 맥락 기억함
input_message = HumanMessage(content = "hi! I'm bob dan i like soccer")
for event in graph.stream({"messages" : [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

input_message = HumanMessage(content="what's my name?")
for event in graph.stream({"messages" : [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

input_message = HumanMessage(content="what's my name?")
for event in graph.stream({"messages" : [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

# 메시지를 2개까지만 기억하게 해뒀기때문에 여기서부터 모름
input_message = HumanMessage(content="what's my favorite?")
for event in graph.stream({"messages" : [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()