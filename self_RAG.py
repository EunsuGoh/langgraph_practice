import os   
import dotenv
dotenv.load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

#WebBaseLoader 이용, 웹페이지 내 텍스트를 크로마 DB에 저장 후 Retriever 로 만들기
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 250, chunk_overlap = 0
)

doc_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()

#문서 관련성 검토 체인 작성
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

#데이터모델
class GradeDocuments(BaseModel):
    """Binary score for relavance check on retrieved documents"""

    binary_score :str = Field(
        description="Documents are elevant to the question, 'yes' or 'no'"
    )

llm = ChatOpenAI(model = "gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

#prompt
system = """You are a grader assassing relevance of a retrieved document to a user question. \n
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
Giva a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human","Retrieved document: \n\n {document} \n\n User question: {question}")
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
question = "agent memory"
docs = retriever.get_relevant_documents(question)
doc_txt = docs[1].page_content
# 문서 관련성 함수가 잘 동작하고 있는지 확인
# print(retrieval_grader.invoke({"question":question, "document":doc_txt})) # answer : 'yes'

#답변생성 Chain 생성
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

#prompt
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name = "gpt-4o-mini", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = prompt | llm | StrOutputParser()

generation = rag_chain.invoke({"context":docs, "question":question})
#답변생성 체크
# print(generation)

#환각 검토 chain 생성
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer"""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

llm = ChatOpenAI(model_name = "gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

#prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
Giva a binary score 'yes' or 'no'. 'yes' means that the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human","Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
# hallucination_grader.invoke({"documents" : docs, "generation":generation}) #yes

#답변 적절성 검토 chain
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question"""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

llm = ChatOpenAI(model_name = "gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

#prompt
system = """You are a grader assessing whether an answer addresses / resolves a question. \n
Giva a binary score 'yes' or 'no'. 'yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human","User question: \n\n {question} \n\n LLM generation: {generation}")
    ]
)
answer_grader = answer_prompt | structured_llm_grader
# answer_grader.invoke({"question" : question, "generation":generation}) #yes

#질문 재작성 chain
llm = ChatOpenAI(model_name = "gpt-4o-mini", temperature=0)

#prompt
system = """You are a question re-writer that converts an input question to a better version that is optimized \n
for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning"""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human","Here is the initial question: \n\n {question} \n Formulate an improved question.")
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
# print(question_rewriter.invoke({"question" : question})) # What are the key concepts and applications of agent memory in artificial intelligence?

#Graph 정의하기
from typing import List
from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
    Represents the state of out graph

    Attributes :
        question: question
        generation : LLM generation
        documents : list of documents
    """
    question: str
    generation: str
    documents: List[str]

#위에서 정의한 chain들을 노드로 정의
def retrieve(state):
    print("---Retrieve---")
    question = state["question"]

    #Retrieval
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question":question}

def generate(state):
    print("---Generate---")
    question = state["question"]
    documents = state["documents"]

    #RAG generation
    generation = rag_chain.invoke({"context":documents, "question":question})
    return {"documents": documents, "question":question, "generation":generation}

def grade_documents(state):
    print("---Check documents relevance to question---")
    question = state["question"]
    documents = state["documents"]

    #Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"documents": d.page_content, "question":question}
        )
        grade = score.binary_score
        if grade == "yes":
            print("--- Grade : Document relevant---")
            filtered_docs.append(d) #연관성있는 문서만 추가
        else:
            print("--- Grade : Document not relevant---")
            continue
    return {"documents": filtered_docs, "question":question}

def transform_query (state):
    print("---Transform query---")
    question = state["question"]
    documents = state["documents"]

    better_question = question_rewriter.invoke({"question":question})
    return {"documents": documents, "question":better_question}

# 이제 분기 노드들 작성

# 분기 노드 1 : 문서 연관성이 없을 때 질문을 재작성하도록 유도하는 노드
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.
    """

    print("---Assess Graded documents---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # 모든 문서가 연관성이 없음
        # 질문을 재작성
        print("---Decision : All documents are not relevant to question, transform query---")
        return "transform_query"
    else:
        # 연관성 있는 문서가 있음
        # LLM 응답 작성
        print("---Decision : LLM generation---")
        return "generate"

#분기 노드 2 : 환각 검토 + 답변 적절성 검토 노드드
def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.
    """
    print("---Check Hallucination---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({
        "documents":documents, "generation":generation
    })

    grade = score.binary_score

    if grade == "yes": #환각현상이 없을 때(팩트에 기반한 답변이 생성됐을때)
        print("---Decision : Generation is grounded in documents---")
        print("---Grade Generation vs Quesion---")
        score = answer_grader.invoke({"documents":documents, "generation":generation})
        grade = score.binary_score
        if grade == "yes": #답변이 적절할 때
            print("---Decision : Generation Addresses question---")
            return "useful"
        else : #답변이 부적절할때
            print("---Decision : Generation does not Addresses question---")
            return "not useful"
    else: #환각현상이 있을 때때
        print("---Decision : Generation is not grounded in documents, Re-try---")
        return "not supported"

#그래프 구축
from langgraph.graph import END, StateGraph, START
workflow = StateGraph(GraphState)

workflow.add_node("retrieve",retrieve)
workflow.add_node("grade_documents",grade_documents)
workflow.add_node("generate",generate)
workflow.add_node("transform_query",transform_query)

#엣지 추가
workflow.add_edge(START,"retrieve")
workflow.add_edge("retrieve","grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query":"transform_query",
        "generate":"generate"
    }
)
workflow.add_edge("transform_query","retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported":"generate",
        "useful":END,
        "not useful":"transform_query"
    }
)

#Compile
app = workflow.compile()

# # 그래프 시각화
# # from Ipython.display import Image, display
# import matplotlib.pyplot as plt
# try:
#     # display(Image(app.get_graph(xray=True).draw_mermaid_png()))
#     plt.imshow(app.get_graph(xray=True).draw_mermaid_png())
# except Exception:
#     pass

from pprint import pprint

inputs = {"question":"Explain how the different types of agent memory work?"}
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"Node '{key}' : ")
        # 옵셔널 : 각 노드 별 스테이트 풀 출력하기
        # pprint.ppint(value["keys"], indent=2, width=80, depth=None)
    print("\n---\n")

print(value["generation"])