from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_template("다음 한글 문장을 프랑스어로 번역해줘 {sentence} French sentence : (print from here)")

llm = ChatOpenAI(model="gpt-4o-mini")

output_parser = StrOutputParser()

runnable_chain = {"sentence":RunnablePassthrough()} | prompt | llm | output_parser
print(runnable_chain.invoke({"sentence":"안녕하세요"}))

# 들어온 입력값에 대해 변수 만들기
runnable_pass = RunnablePassthrough.assign(
    sentence = lambda x: x["sentence"]
)

runnable_chain = runnable_pass | prompt | llm | output_parser

print(runnable_chain.invoke({"sentence":"안녕하세요"}))

 
