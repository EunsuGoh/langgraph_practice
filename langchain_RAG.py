import os
import warnings
import dotenv
dotenv.load_dotenv()
warnings.filterwarnings("ignore")
API_KEY = os.getenv('OPENAI_API_KEY')


#PDF 파일 로드
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./RAG.pdf")
# pages = loader.load_and_split()
pages  = loader.load()

# print(pages[0])
# for page in pages:
#     print("page type: ",type(page.page_content))
#     print("page content: ",page.page_content)

# #Web 파일 로드
# from langchain_community.document_loaders import WebBaseLoader
# import bs4

# loader = WebBaseLoader("https://www.espn.com/",
#                        bs_kwargs=dict(
#                            parse_only=bs4.SoupStrainer(
#                                class_ = "headlineStack top-headlines"
#                            )
#                        ))
# #ssl 오류 방지
# loader.requests_kwargs = {"verify": False}

# data = loader.load()
# # print(data)

# #Text splitter
# from langchain_text_splitters import CharacterTextSplitter

# text_splitter = CharacterTextSplitter(
#     separator="\n", #활성화 하지 않으면 500자씩 안끊길 수 있음, 어떤 세퍼레이터가 적당한지 모르니 리컬시브 세퍼레이터를 사용함
#     chunk_size=500,
#     chunk_overlap=100,
#     length_function=len
# )

# chunks = text_splitter.split_documents(pages)

# # print(chunks[0])
# # print([len(chunk.page_content) for chunk in chunks])

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, # 세퍼레이터 항목 없이도 500자 이내로 끊김
    chunk_overlap=100,
    length_function=len,
    # is_separator_regex=False
)   

chunks = text_splitter.split_documents(pages)

# # print(chunks[0])
# # print([len(chunk.page_content) for chunk in chunks])


# Embedding
from langchain_openai import OpenAIEmbeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=API_KEY
)
embeddings = embedding_model.embed_documents([i.page_content for i in chunks])
# print(len(embeddings),len(embeddings[0]))


# 벡터 스토어 저장
from langchain_community.vectorstores import FAISS
db = FAISS.from_documents(chunks,embedding_model)

#Retriver 생성
retriever = db.as_retriever()
query = "what is RAG?"
#유사 문서 검색
# print(retriever.invoke(query))


#langchain에 RAG 구현
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

model = ChatOpenAI(model = "gpt-4o",
                   api_key = API_KEY)
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

print(rag_chain.invoke("How to consist RAG?"))
