from typing import Annotated

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages : Annotated[list[dict], add_messages] # 메시지 추가

graph_builder = StateGraph(State)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def chatbot(state: State) -> State:
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")

graph = graph_builder.compile()

# 매번 State로 메시지 선언하지 말라고 MessageState 라는 타입 사용
# 단순한 형태의 챗봇에서만 MessageState 사용
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage

graph_builder = StateGraph(MessagesState)

# # 챗봇 실행
# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ["exit", "quit","q"]:
#         break
#     for event in graph.stream({"messages": ("user", user_input)}): # 대화가 한 턴씩 보임
#         for value in event.values():
#             print("Assistant: ", value["messages"][-1].content) # state안에 메시지를 쌓을 것이기 때문에, 가장 최신 메시지를 가져옴

# MessageState 업데이트하기
class State(MessagesState):
    counter : int

graph_builder = StateGraph(State)

# 챗봇 노드 변경
def chatbot(state: State) -> State:
    state["counter"] = state.get("counter", 0) + 1
    return {
        "messages": [llm.invoke(state["messages"])],
        "counter": state["counter"]
    }

graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")

graph = graph_builder.compile()

initial_state = {"messages": [HumanMessage(content="Hello")], "counter": 0}

result = graph.invoke(initial_state)
print(f"final state: {result}")

state = initial_state
for _ in range(3):
    state = graph.invoke(state)
    print(f"Counter: {state['counter']}")
    print(f"Last message : {state['messages'][-1].content}")
    print("-"*3)
