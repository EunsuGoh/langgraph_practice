import os
import dotenv
dotenv.load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

# from typing import TypedDict

# class State(TypedDict):
#     counter: int
#     alphabet: list[str]

# graph_builder = StateGraph(State)

from typing import Annotated # 해당 값에 메타데이터 넣어줌
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
import operator

class State(TypedDict):
    counter: int 
    alphabet: Annotated[list[str], operator.add] #챗봇의 기억 유지

def node_a(state: State) -> State:
    state["counter"] = state["counter"] + 1
    state["alphabet"] = ["Hello"]
    return state

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot",node_a) # 함수를 노드로 지정
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# 그래프 시각화
# from IPython.display import Image, display
# display(Image(graph.get_graph(xray=True).draw_mermaid_png()))

# 그래프 실행
#초기상태 정의
initial_state = {"counter": 0, "alphabet": []}

state = initial_state

for _ in range(3):
    state = graph.invoke(state)
    print(state)


