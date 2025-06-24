from decouple import config
import os

from langchain_groq import ChatGroq

from langchain_community.document_loaders import ObsidianLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_postgres import PGVector

from langchain_text_splitters.markdown import MarkdownTextSplitter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage,HumanMessage, AIMessage

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
    temperature=1,
    model_name='llama-3.3-70b-versatile'
)


# chat_history --> store in DB after user clicks of the chat, but in new chat bring in the array and use array only ---> RAM

# probably change DB put everything to sqlite

# if user asked retrieve the document


chat_history = [
    SystemMessage(content='''
You are a helpful and intelligent assistant integrated with an Obsidian-based knowledge management system.
The user may query based on their personal Obsidian vault, which includes markdown notes organized into folders.
If no relevant Obsidian documents are found, respond with:
"No relevant Obsidian files found."
Then ask the user if they would like to continue the conversation in a general sense, without using the context of Obsidian.
''')
]

session = True

while session:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        session = False
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI", result.content)
    
print(chat_history)