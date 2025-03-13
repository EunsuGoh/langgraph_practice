import os
import dotenv
from langchain_openai import ChatOpenAI
dotenv.load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')



# 랭체인 없이 llm 사용하기
# from openai import OpenAI

# client = OpenAI()
# response = client.chat.completions.create(
#     model='gpt-4o-mini',
#     messages=[
#         {
#             'role': 'user',
#             'content': '2002년 월드컵 4강 국가 알려줘'
#         }
#     ]
# )

# print(response)

# 랭체인 사용하기
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model='gpt-4o-mini')
print(chat.invoke('자기소개 한번 해봐라.'))

