#!/usr/bin/env python
# coding: utf-8

# In[16]:


get_ipython().system(' pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain sentence-transformers')


# In[2]:


import getpass
import os

os.environ['LANGCHAIN_TRACING_V2'] = 'True'
os.environ['LANGSMITH_ENDPOINT']= 'https://api.smith.langchain.com'
os.environ['LANGSMITH_API_KEY']=os.environ['LANGSMITH_API_KEY_2']

os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_API_KEY']
os.environ['GEMINI_API_KEY'] = os.environ['GEMINI_API_KEY']


# In[3]:


question = 'Summarise the content given'
with open('smallContent.txt', 'r') as f:
    data = f.read()

# to approximate tokens
import tiktoken

def count_tokens(content, encoding_name='cl100k_base'):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(content))
    return num_tokens

tokens = count_tokens(question)
print(f"question: {tokens}")

tokens = count_tokens(data)
print(f'content: {tokens}')


# In[4]:


# split
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=10)
splits = text_splitter.create_documents([data])
print(splits[1])


# In[5]:


# embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

model_name = 'intfloat/e5-large'
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},  
    encode_kwargs={'normalize_embeddings': True}
)
print(embeddings)


# In[6]:


# retriever
from langchain_community.vectorstores import Chroma

persist_directory = './chroma_e5_db'

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,  
    persist_directory=persist_directory
)



# In[7]:


# Retrieval and Generation

retriever = vectorstore.as_retriever(kwargs=3)

# 1. Retrieval
from langchain import hub

prompt = hub.pull("rlm/rag-prompt")

# LLM
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=os.environ["GEMINI_API_KEY"],
)


# In[8]:


# post processing - generation
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
rag_chain.invoke("Who is Porlock?")

