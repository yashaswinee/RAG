#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os


os.environ['GEMINI_API_KEY'] = os.environ['GEMINI_API_KEY']

os.environ['LANGSMITH_API_KEY']= os.environ['LANGSMITH_API_KEY']
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGSMITH_ENDPOINT']='https://api.smith.langchain.com'
os.environ['LANGSMITH_PROJECT']='RAG-fusion'

from langsmith import traceable
print(os.environ['TASTY_TOAST'])


# In[10]:


with open('smallContent.txt', 'r') as f:
    data = f.read()


# In[11]:


# splits

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=10)
splits = text_splitter.create_documents([data])
print(splits[1])


# In[12]:


# embed
from langchain_community.embeddings import HuggingFaceEmbeddings

model_name = 'intfloat/e5-large'
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)


# In[5]:


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


# In[6]:


# template
from langchain.prompts import ChatPromptTemplate

template = """You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines. Original question: {question}"""

from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(template)
 
# gives 6 questions based on user input
@traceable 
def gen_queries():
    generate_queries = (
        prompt
        | llm
        | StrOutputParser()
        | (lambda x : x.split("\n"))
    )
    return generate_queries

generate_queries = gen_queries()
print(generate_queries)


# In[7]:


# retriever
from langchain_community.vectorstores import Chroma

persist_directory = './chroma_e5_db'

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,  
    persist_directory=persist_directory
)
retriever = vectorstore.as_retriever()


# In[8]:


from langchain_core.load import dumps, load

# parallel process
def get_unique_union(documents: list[list]):
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]

    unique_docs = list(set(flattened_docs))
    return [load(docs) for docs in unique_docs]

question = "I don't understand the context. Give me summary."

@traceable
def get_docs_retrieval_chain():
    retrieval_chain = generate_queries | retriever.map() | get_unique_union

    docs = retrieval_chain.invoke({"question": question})

    print(len(docs))
    print(docs)
    return retrieval_chain

retrieval_chain = get_docs_retrieval_chain()


# In[9]:


# Final RAG
from operator import itemgetter

@traceable
def final_query():
    template = """Your job is to give a concise answer. Answer the question based on the context:
    {context},
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    final_rag_chain = (
        {"context": retrieval_chain,
        "question": itemgetter("question")}
        | prompt 
        | llm
        | StrOutputParser()
    )

    response = final_rag_chain.invoke({"question": question})
    return response

response = final_query()
print(response)


# In[ ]:




