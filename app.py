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
import os
from dotenv import load_dotenv

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    api_key=groq_api_key,
    temperature=0.7,
    model_name="llama3-70b-8192"
)

st.title("PDF QA Assistant")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None and "retriever" not in st.session_state:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = [doc.page_content for doc in docs]
        chunks = splitter.create_documents(texts)
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        st.session_state.retriever = retriever
        st.session_state.docs = docs
        st.success("PDF processed and vector index created.")

if "chain" not in st.session_state and "retriever" in st.session_state:
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

if "chain" in st.session_state:
    user_question = st.text_input("Enter your question", key="question")

    if st.button("Get Answer") and user_question.strip() != "":
        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke(user_question)
        st.markdown("### Answer:")
        st.write(response)
