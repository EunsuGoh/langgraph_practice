import os
import dotenv
dotenv.load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')


from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages : Annotated[list, add_messages]

from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode,tools_condition

tool = TavilySearchResults(max_results=2)
tools = [tool]

tool_node = ToolNode(tools) #AI가 도구를 사용할 수 있게 노드로 묶어줌

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=API_KEY)

llm_with_tools = model.bind_tools(tools)

def chatbot(state: State) -> State:
    result = llm_with_tools.invoke(state["messages"])
    return {"messages": [result]}

from langgraph.graph import StateGraph



graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge("tools","chatbot")
graph_builder.add_conditional_edges("chatbot",tools_condition) #tools conditions 라이브러리는 랭그래프에서 컨디셔널 엣지를 자동화해줌

graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()

#웹검색이 필요한 질문
print(graph.invoke({"messages": {"role" : "user","content" : "지금 한국 대통령 누구야?"}}))
