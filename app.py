import streamlit as st
import os
from dotenv import load_dotenv

from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory as RunnableMessageHistory

from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Streamlit UI ---
st.set_page_config(page_title="Chat with PDF", layout="centered")
st.title("📄 Chat with PDF (Conversational RAG)")
st.caption("Upload a PDF and ask questions about it. Memory-aware, context-driven answers.")

with st.sidebar:
    st.header("🔐 Configuration")
    api_key = st.text_input("Enter your Groq API key", type="password")
    session_id = st.text_input("Session ID", value="default_session")
    uploaded_file = st.file_uploader("📤 Upload a PDF", type="pdf", accept_multiple_files=False)

if api_key and uploaded_file:
    llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    with st.spinner("📚 Processing your document..."):
        temp_pdf = "./temp.pdf"
        with open(temp_pdf, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        texts = [str(doc.page_content) for doc in splits if doc.page_content and str(doc.page_content).strip()]
        metadatas = [doc.metadata for doc in splits if doc.page_content and str(doc.page_content).strip()]

        vectorstore = FAISS.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas
)


        retriever = vectorstore.as_retriever()

        # Prompt templates
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question, which might reference context in the chat history, formulate a standalone question. Do not answer it."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(create_history_aware_retriever(llm, retriever, contextualize_q_prompt), question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Chat interface
        st.divider()
        st.subheader("💬 Chat Interface")

        user_input = st.chat_input("Ask a question about the PDF content...")
        session_history = get_session_history(session_id)

        if user_input:
            with st.spinner("🧠 Thinking..."):
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                st.session_state.last_user_input = user_input
                st.session_state.last_response = response["answer"]

        # Show chat messages
        for msg in session_history.messages:
            with st.chat_message("user" if msg.type == "human" else "assistant"):
                st.markdown(msg.content)

        # Show latest response if not already in history
        if st.session_state.get("last_response") and user_input:
            with st.chat_message("assistant"):
                st.markdown(st.session_state.last_response)

        # Cleanup temp file
        os.remove(temp_pdf)

else:
    st.info("🔑 Please enter your Groq API key and upload a PDF to begin.")
