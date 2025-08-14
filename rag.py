from decouple import config
import os

# notice it says Chat not simply Groq, it is a chat model and not an llm
from langchain_groq import ChatGroq

# used for loading plain text files --> community pacakge
# from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# temporary vector database
# from langchain_community.vectorstores import FAISS
from langchain_postgres import PGVector

from langchain_text_splitters import RecursiveCharacterTextSplitter
# MarkdownTextSplitter inherits from RecursiveCharacterTextSplitter --> see
from langchain_text_splitters.markdown import MarkdownTextSplitter, MarkdownHeaderTextSplitter

# ye core wali cheez hai prompt template
from langchain_core.prompts import ChatPromptTemplate

'''
Purpose: A utility in LangChain that passes data from one step to another without modification.

'''

from langchain_core.runnables import RunnablePassthrough

'''
Purpose: Parses the raw output from the LLM (like JSON or plain text) into a Python string.

Use case: Useful when you're expecting a plain text answer from the LLM and want it as a str.
'''
from langchain_core.output_parsers import StrOutputParser


#? loading the document

document_path = "./2025.md"
loader = UnstructuredMarkdownLoader(document_path)

documents=loader.load()

# initialising the text splitter with some settings
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=400,
#     # to maintain context
#     chunk_overlap=120
# )

# text_splitter = MarkdownHeaderTextSplitter(
#     headers_to_split_on=[
#         ("#", "Note Theme"),
#         ("###", "Section")
#     ]
# )

text_splitter = MarkdownTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


'''
🔢 1. chunk_size

This defines how big each chunk is (in characters).
💡 Consider:

    Model input limits: Chunk size should be small enough to leave room for prompts and responses in your model's token limit. 250 chars ≈ 50 tokens.

    Text type:

        Technical code/docs: 200-400 characters

        Dense academic content: 300-500

        Light blogs/articles: 500-800

👍 Rule of thumb:

    Use chunk_size = 200-500 for QA/chatbots

    If your downstream model uses short context, stay under 300

🔁 2. chunk_overlap

This defines how much overlap you keep between chunks (to preserve continuity).
💡 Consider:

    High overlap helps prevent missing context at chunk boundaries

    Too much overlap increases storage & retrieval cost

👍 Rule of thumb:

    20-30% of chunk size is common:

        chunk_size = 250 → chunk_overlap = 50-75

'''

# this is an array of document objects of the related document
# print(documents)
# input type should be string not document

# !!!! changing text splitters may introduce more vectors in the db thereby reducing effiency of RAG
splits = text_splitter.split_documents(documents)

#! debugging step
print("----Starting the splits----")
for i, chunk in enumerate(splits):
    print(f"\n=== Chunk {i} ===\n")
    print(chunk.page_content)

# print(splits)

os.environ["GROQ_API_KEY"] = config("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = config("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


# making the vector DB and convert the splits to vectors using the embedding model from above and store in the database
# vectorstore = FAISS.from_documents(
#     documents=splits, 
#     embedding=embeddings
# )

# Using pg vector

CONNECTION_STRING = config("NEON_POSTGRES_DB").replace("postgresql://", "postgresql+psycopg://")
COLLECTION_NAME="obsdian_rag"

# https://python.langchain.com/docs/integrations/vectorstores/pgvector/
# https://python.langchain.com/api_reference/postgres/vectorstores/langchain_postgres.vectorstores.PGVector.html
# https://python.langchain.com/api_reference/postgres/vectorstores/langchain_postgres.vectorstores.PGVector.html#langchain_postgres.vectorstores.PGVector.from_documents

# 1 chunk == 1 vector

vectorstore = PGVector.from_documents(
    # documents is not supported in PG Vector --> use PGVector.from_documents and when using this use embedding not
    documents=splits,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
)

# see documentation + use perplexity --> this is only while using PGVector not PGVector.from_documents
# vectorstore.add_documents(documents=splits)

retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

docs = retriever.invoke("What books do I plan to read?")
# kind of like the auto iterator in c++
print("----Top 3 retrieved chunks/vectors---")
for i, doc in enumerate(docs):
    print(f"\n=== Retrieved Doc {i} ===\n{doc.page_content}")





# search kwargs return the top 3 most relevant chunks/vectors


'''
What It Does:

It converts your vectorstore into a retriever object that can be used to fetch the most *relevant documents* in response to a user query.
'''

# Intialising the chat model

model = ChatGroq(
    # temperature is creatvity/randomness of the chatmodel's output, low temp less creativity
    temperature=0.5,
    model_name='gemma2-9b-it'
)

# System prompt(defined by the developer)

# context --> vector DB se aaega
# question --> user's query

rag_prompt = ChatPromptTemplate.from_template("""You are a helpful AI assistant. Answer the question based only on the following context:

Context:
{context}

Question: {question}

Helpful Answer:""")


# Langchain Expression Language (LCEL) to build a rag chain

# Create RAG Chain using LCEL
def create_rag_chain():
    return (
        # Take the input question and pass it to the retriever
        # RunnablePassthrough() is a no-op step in LangChain — it simply forwards the input as-is to the next step in your chain.
        {"context": retriever, "question": RunnablePassthrough()}
        # Format the context and question in the prompt
        | rag_prompt
        # Pass to the chatmodel
        | model
        # Convert to string output
        | StrOutputParser()
    )

# Initialize the RAG Chain
rag_chain = create_rag_chain()


def perform_rag(query):
    return rag_chain.invoke(query)

query1 = "What actions i am planning to take for fitness?"
query2 = "What books do i plan to read "

result = perform_rag(query1)
result2 = perform_rag(query2)

print(result)
print(result2)

