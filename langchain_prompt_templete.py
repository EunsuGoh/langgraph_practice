from langchain.prompts import PromptTemplate

prompt=PromptTemplate(
    template='''
                너는 요리사야. 내가 가진 재료들을 갖고 만들 수 있는 요리를 {number}추천하고    
                요리 이름과 재료를 알려줘.
                재료는 다음과 같아:
                {ingredients}
    ''',

)

prompt

print(prompt.invoke({'number': 3, 'ingredients': '치즈, 피자, 토마토'}))