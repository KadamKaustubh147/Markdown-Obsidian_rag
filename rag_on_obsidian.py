from decouple import config
import os

from langchain_groq import ChatGroq
from langchain_community.document_loaders import ObsidianLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# raw strings to treat the backlashes as they are instead of escape character
VAULT_PATH = r"C:\Users\kaust\OneDrive\Documents\Obsidian vaults\Obsidian_datascience"
print(f"Loading documents from: {VAULT_PATH}")

loader = ObsidianLoader(VAULT_PATH)
documents = loader.load()
print(f"✅ Loaded {len(documents)} documents")

if not documents:
    raise ValueError("❌ No documents found. Check the vault path or permissions.")

'''
Type schema for this text splitter function

split_documents(

    documents: Iterable[Document],

) → list[Document]

'''

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

splits = text_splitter.split_documents(documents)
print(f"✅ Created {len(splits)} text chunks (splits)")

if not splits:
    raise ValueError("❌ No text chunks were created. Check if your documents have content.")

os.environ["GROQ_API_KEY"] = config("GROQ_API_KEY")

embedding = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={"device": "cuda"},
    encode_kwargs={"batch_size": 64}
)

'''
When you pass a list of texts to embedding.embed_documents() (or .embed_query()), internally it calls the model’s .encode() method.

"batch_size": 64 means the model will process 64 sentences/documents at a time before moving to the next batch.

This batching is important because:

    It reduces the number of GPU calls (faster than doing one at a time).

    It helps fit into GPU/CPU memory — if you set this too high, you might get an out-of-memory (OOM) error.

    Too low, and performance will drop because of under-utilization.
'''


# Test embedding generation
try:
    test_vec = embedding.embed_query("test")
    print(f"✅ Embedding model loaded. Vector length: {len(test_vec)}")
except Exception as e:
    raise RuntimeError(f"❌ Failed to generate embeddings: {e}")

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "rag_on_obsidian_vault"

# Create or load local Chroma DB
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding,
    persist_directory=CHROMA_DIR
)

# Only add documents if the DB is empty
# https://docs.trychroma.com/docs/collections/manage-collections --> this collection count is from chroma no docs about it in langchain
doc_count = vectorstore._collection.count()
print(f"📦 Chroma collection '{COLLECTION_NAME}' currently has {doc_count} documents")

def add_documents_in_batches(vectorstore, documents, batch_size=4000):
    total_docs = len(documents)
    print(f"Adding {total_docs} documents in batches of {batch_size}...")
    # starts from 0 till 6000ish, batch size 4000; i is 0,4000,8000 etc.
    for i in range(0, total_docs, batch_size):
        # partitions the documents in list in factors of batch size --> batch is the paritioned list of documents
        batch = documents[i : i + batch_size]
        vectorstore.add_documents(batch)
        print(f"Added batch {i // batch_size + 1} ({len(batch)} documents)")

if doc_count == 0:
    print("📥 Adding documents to Chroma in batches...")
    add_documents_in_batches(vectorstore, splits, batch_size=4000)
    # vectorstore.persist()
    print(" Documents added & persisted.")

# remaining docs add karna hai 
else:
    print(" Using existing Chroma collection.")

retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

model = ChatGroq(
    temperature=1,
    model_name='llama-3.3-70b-versatile'
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
You are a helpful and intelligent assistant integrated with an Obsidian-based knowledge management system.
The user may query based on their personal Obsidian vault, which includes markdown notes organized into folders.
If no relevant Obsidian documents are found, respond with:
"No relevant Obsidian files found."
Then ask the user if they would like to continue the conversation in a general sense, without using the context of Obsidian.
"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
    ("system", "Context from Obsidian:\n{context}")
])

def retrieve_context(question: str):
    docs = retriever.invoke(question)
    if not docs:
        return "No relevant Obsidian files found."
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"],
        "context": lambda x: retrieve_context(x["question"])
    }
    | prompt
    | model
)

# -------------------------
# Chat loop
# -------------------------
chat_history = []

session = True
while session:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        session = False
        break

    result = chain.invoke({
        "question": user_input,
        "chat_history": chat_history
    })

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=result.content))

    print("AI:", result.content)

print(chat_history)
