# RAG

## LangSmith

This code interacts with LLM APIs, specifically using the Gemini-2.5-Flash model, for a document retrieval task utilising the ChatPromptTemplate. A retriever function reads content from text, simulating a document store. The subsequent RAG (Retrieval-Augmented Generation) function uses this retrieved content to constrain the LLM's responses through a system prompt. The script integrates the LLM API with the LangChain framework, which is monitored using `LangSmith`.

Template used:
```
systemTemplate = f"Answer the users question using only the provided information below: {docs}"
```

## RagSingleQuery

The larger text is then divided into manageable chunks using the RecursiveCharacterTextSplitter. These chunks are subsequently embedded using the HuggingFace model 'intfloat/e5-large' and stored in a persistent Chroma vector database.

`RAG Chain`

Retrieval: The user's question is passed to the retriever, which performs a vector similarity search to fetch relevant context documents.

Formatting: A custom format_docs function concatenates the content of the retrieved documents into a single string.

Generation: The formatted context and the original question are passed to a standard RAG prompt pulled from the LangChain Hub (rlm/rag-prompt).

```
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```
The Gemini LLM then processes the context and the question to produce the final answer based only on the retrieved information.


## MultiQuery

The larger text is divided into chunks using RecursiveCharacterTextSplitter. These chunks are then embedded using the HuggingFace model 'infloat/e5-large'. The MultiQuery technique is employed to allow multiple queries, and the responses are combined to enhance performance and accuracy. The first process involves taking a single user question and goes through several stages: Prompt → LLM (Gemini) → String Output Parser → Split by newline. This process results in a list of six related questions, which includes one original question along with five generated questions.


`Parallel Retrieval Chain`: A custom function is used to take all documents returned from the six parallel retrievals, flatten the list, and return only the unique documents. This ensures a diverse context is gathered.

```
retrieval_chain = generate_queries | retriever.map() | get_unique_union
```


`RAG chain`: To generate a final answer, we define a final prompt template that asks the LLM to provide a concise answer based only on the information given.

```
final_rag_chain = (
    {"context": retrieval_chain,
    "question": itemgetter("question")}
    | prompt 
    | llm
    | StrOutputParser()
)
```
The Gemini LLM processes the retrieved documents and the original question to provide the final, synthesised response.
