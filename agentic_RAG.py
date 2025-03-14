import os   
import dotenv
dotenv.load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
# print(docs_list)

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 100, chunk_overlap = 50
)

doc_split = text_splitter.split_documents(docs_list)
chroma = Chroma()

#Add to vector DB
vertorstore = chroma.from_documents(
    documents=doc_split,
    collection_name="rag_chroma",
    embedding=OpenAIEmbeddings()
)

#벡터 DB를 리트리버로서 선언
retriever = vertorstore.as_retriever()

# Retriever를 도구로 저장
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search snd return information about Lilien Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs"
)
tools = [retriever_tool]

#AgentState 선언
from typing import Annotated, Sequence, TypedDict,Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage],add_messages] #BaseMessage 를 시퀀셜하게 저장할것이며, add_messages를 통해 메시지를 누적시킴

#문서 관련성 검토 함수 정의
from langchain import hub
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import tools_condition

def grade_documents(state) -> Literal["generate","rewrite"]: #정해진 두 개 중 하나의 값만 출력해야함
    """
    Determines whether the retrieved documents are relevant to the question,

    Args : 
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """
    print("---CHECK RELEVANCE---")

    #Data model
    class grade(BaseModel): #파이던틱 구조를 상속 / 사용자 정의 데이터 모델
        """Binary score for relevance check"""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'") #yes, no 둘 중 하나의 값만 들어오도록 엄격하게 제어(타입 에러 감지용도)

    #그레이더 역할을 수행할 LLM
    model = ChatOpenAI(temperature=0,model='gpt-4o-mini', streaming=True)
    #LLM Tool binding
    llm_with_tool = model.with_structured_output(grade) #모델의 리턴이 grade class과 같이 출력(바이너리 스코어)

    #prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n{context}\n\n
        Here is ther user question : {question}\n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevent. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context","question"]
    )

    #chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question":question, "context":docs})

    score = scored_result.binary_score

    if score =="yes":
        print("---DECISION: DOCS RELEVANT")
        return "generate"
    
    else:
        print("---DECISION: DOCS NOT RELEVANT")
        print(score)
        return "rewrite"
    

#사용자와 상호작용하는 에이전트 함수 정의
def agent(state):
    """
    Invokes the agent model to generate a response bsed on the current state. Given the question, it will decide to retrieve the retriever tool, or simply end.

    Args : 
        state(messages): The current state
    
    Returns:
        dict: The updated state with the agent response appended to messages.
    """

    print("---CALL AGENT---")
    messages = state["messages"]
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o-mini")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages":[response]}

#질문 재작성 함수 정의
def rewrite(state):
    """
    Transform the query to produce a better question.

    Arge:
        State(messages) : The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f"""
    Look at the input and try to reason about the underlying semantic intent or meaning.\n
    Here is the initial question:
    \n ------- \n
    {question}
    \n ------- \n
    Formulate an improved question: """
        )
    ]

    #grader
    model = ChatOpenAI(temperature=0, model = "gpt-4o-mini", streaming=True)
    response = model.invoke(msg)
    return {"messages":[response]}


# 답변 함수
def generate(state):
    """
    Generate answer

    Args : 
        state(messages) : The current state
    
    Returns :
        dict: The updated state with re-phrased question
    """

    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    #Prompt 
    prompt = hub.pull('rlm/rag-prompt')

    #LLM
    llm = ChatOpenAI(temperature=0, model = "gpt-4o-mini", streaming=True)

    #Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    #Chain
    rag_chain = prompt | llm | StrOutputParser()

    #Run
    response = rag_chain.invoke({"context":docs,"question":question})
    return {"messages":{response}}


print("*"*20 + "Prompt[rlm/rag-prompt]"+"*"*20)
prompt = hub.pull("rlm/rag-prompt").pretty_print() 
print(prompt)#프롬프트 어떻게 생겼느지 프린트

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

#Define a new graph
workflow = StateGraph(AgentState)

#Define the nodes we will cycle between
workflow.add_node("agent", agent)
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate",generate)

workflow.add_edge(START,"agent")

#1. 검색이 필요한지 판단
workflow.add_conditional_edges(
    "agent",
    tools_condition, #에이전트의 리턴이 tool_calling을 포함하면 리트리브 툴을 콜하고, 아닌 경우 END
    {
        "tools":"retrieve",
        END:END
    }
)

#2. 검색해온 문서가 적절한지 판단
workflow.add_conditional_edges(
    "retrieve",
    grade_documents
)

workflow.add_edge("generate",END)
workflow.add_edge("rewrite","agent")

#Compile
graph = workflow.compile()


#Graph 실행하기
query_1 = "agent memory가 무엇인가요?"
query_2 = "Lilian Weng은 agent memory를 어떤 것에 비유했나요?"

import pprint

# Query 1 질문 : 정상적인 답변
inputs = {"messages": [("user", query_1)]}

for output in graph.stream(inputs,{"recursion_limit":10}): #inputs : state / 두번째 딕셔너리는 config (recursion limit 은 노드들 반복 총 제한횟수 - 무한루프 제한)
    for key, value in output.items():
        print(f"Output from node '{key}") 
        print("---")
        print(value)

# Query 2 질문 : 블로그에 없는 질의, Not relevant 출력 후 노드 반복 횟수 제한에 의해 종료됨
inputs = {"messages": [("user", query_2)]}

for output in graph.stream(inputs,{"recursion_limit":10}): #inputs : state / 두번째 딕셔너리는 config (recursion limit 은 노드들 반복 총 제한횟수 - 무한루프 제한)
    for key, value in output.items():
        print(f"Output from node '{key}")
        print("---") 
        print(value)