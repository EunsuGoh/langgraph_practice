from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

#prompt template 생성
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

#llm 생성
llm = ChatOpenAI(model="gpt-4o-mini")

#chain 생성
chain = prompt | llm | StrOutputParser()

#chain 실행
print(chain.invoke({"topic": "chickens"}))
