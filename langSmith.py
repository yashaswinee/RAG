#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os 

os.environ['LANGSMITH_TRACING']='true'
os.environ['LANGSMITH_ENDPOINT']='https://api.smith.langchain.com'
os.environ['LANGSMITH_API_KEY']= os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_PROJECT']='sherlock text'
os.environ['OPENAI_API_KEY']= os.getenv('OPENAI_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

print(os.environ['TASTY_TOAST'])


# In[3]:


import getpass
import os
from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")


# In[4]:


from langchain_core.prompts import ChatPromptTemplate

systemTemplate = "Translate from english to {language}"
promptTemplate = ChatPromptTemplate.from_messages(
    [('system', systemTemplate), ('user', "{text}")]
) 

prompt = promptTemplate.invoke( {'language': 'hindi', 'text': 'Hello! How are you?'})

prompt.to_messages()

response = model.invoke(prompt)
response.content

print(response.content)



# In[6]:


from langsmith import traceable

def retriever():
    with open("smallContent.txt", "r") as file:
        data = file.read()
    return data


@traceable
def rag(question):
    docs = retriever()
    systemTemplate = f"Answer the users question using only the provided information below: {docs}"

    promptTemplate = ChatPromptTemplate.from_messages([
        ('system', systemTemplate), ('user', question)
    ])
    
    prompt = promptTemplate.invoke({})
    response = model.invoke(prompt)
    return response.content

response = rag('Give short summary of story')

print(response)

    

