from decouple import config
import os

from langchain_groq import ChatGroq

from langchain_community.document_loaders import ObsidianLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_postgres import PGVector

from langchain_text_splitters.markdown import MarkdownTextSplitter

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.runnables import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser


# loading the Obsidian folder containing obsidian files

# D:\Obsidian-vault\Obsidian Notes

loader = ObsidianLoader("/mnt/d/Obsidian-vault/Obsidian Notes")

documents=loader.load()

text_splitter = MarkdownTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

splits = text_splitter.split_documents(documents)

os.environ["GROQ_API_KEY"] = config("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = config("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

CONNECTION_STRING = config("NEON_POSTGRES_DB").replace("postgresql://", "postgresql+psycopg://")

COLLECTION_NAME="rag_on_obsidian_vault"

vectorstore = PGVector.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
)

retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

model = ChatGroq(
    temperature=0.5,
    model_name='gemma2-9b-it'
)

