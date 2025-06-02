import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import tempfile
import hashlib
from dotenv import load_dotenv
import os

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    api_key=groq_api_key,
    temperature=0.7,
    model_name="llama3-70b-8192"
)

st.title("PDF QA Assistant")

def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()  # read once and reuse
    file_hash = get_file_hash(file_bytes)

    # Check if this is a new file
    if st.session_state.get("last_file_hash") != file_hash:
        with st.spinner("Processing new PDF..."):
            # Write to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name

            # Load and process
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = [doc.page_content for doc in docs]
            chunks = splitter.create_documents(texts)

            # Create embeddings and retriever
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
            vector_store = FAISS.from_documents(chunks, embeddings)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

            # Store in session
            st.session_state.retriever = retriever
            st.session_state.last_file_hash = file_hash
            st.session_state.chain = None  # reset chain

            st.success("PDF processed and vector store created.")

# Create chain if retriever exists and chain not set
if st.session_state.get("retriever") and st.session_state.get("chain") is None:
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY using the context below.
        If the context is insufficient, say you cannot help.

        Context:
        {context}

        Question: {question}
        """,
        input_variables=['context', 'question']
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    parser = StrOutputParser()

    parallel_chain = RunnableParallel({
        'context': st.session_state.retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    chain = parallel_chain | prompt | model | parser
    st.session_state.chain = chain

# Ask a question
if st.session_state.get("chain"):
    user_question = st.text_input("Enter your question")

    if st.button("Get Answer") and user_question.strip() != "":
        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke(user_question)
        st.markdown("### Answer:")
        st.write(response)
