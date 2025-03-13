import os
import dotenv
dotenv.load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')


from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langgraph.prebuilt import ToolNode

@tool
def get_weather(location: str) -> str:
    """Call to get the weather"""
    if location in ["서울", "인천"]:
        return f"오늘 서울과 인천의 날씨는 15도입니다."
    else:
        return f"오늘 {location}의 날씨는 20도입니다."

@tool
def get_coldest_cities():
    """Get a list of coldest cities"""
    return "서울,고성" 

tools= [get_weather, get_coldest_cities]
tool_node = ToolNode(tools)

from langchain_openai import ChatOpenAI

model_with_tools = ChatOpenAI(
    model="gpt-4o-mini",
 ).bind_tools(tools) #모델이 도구선정, 입력값 선정까지 알아서 함


#Conditional Edge 생성
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState

def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    # 마지막 메시지가 도구 호출이면 도구 호출 노드로 이동
    if last_message.tool_calls:
        return "tools"
    # 도구 호출이 없으면 END로 이동
    return END


def call_model(state:MessagesState) :
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue
)

workflow.add_edge("tools", "agent") #tools노드는 무조건 agent로 응답을 보내기 때문에 conditional edge가 필요없음

app = workflow.compile()

# final_state = app.invoke({"messages": [HumanMessage(content="서울 날씨 알려줘")]})
# print(final_state["messages"][-1].content) #서울 15도 입니다.

for chunk in app.stream({"messages": [HumanMessage(content="가장 추운 날씨 알려줘")]}, stream_mode="values"): #중간 결과물 전부 출력
    print(chunk["messages"][-1].pretty_print())
