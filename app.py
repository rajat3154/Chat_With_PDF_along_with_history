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

# Streamlit UI
st.title("Conversational RAG with PDF Uploads and Chat History")
st.write("Upload a PDF file to start chatting with its content.")

api_key = st.text_input("Enter your Groq API key:", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")

    session_id = st.text_input("Session ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False)

    if uploaded_file:
        with st.spinner("Processing document..."):
            # Save the uploaded PDF to disk
            temp_pdf = "./temp.pdf"
            with open(temp_pdf, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Load and split the document
            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            # Create vectorstore
            texts = [str(doc.page_content) for doc in splits if doc.page_content and str(doc.page_content).strip()]
            metadatas = [doc.metadata for doc in splits if doc.page_content and str(doc.page_content).strip()]


            vectorstore = FAISS.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas
)




            retriever = vectorstore.as_retriever()

            # Prompt to reformulate user query with chat history
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question, "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do not answer the question, "
                "just reformulate it if needed and otherwise return it as it is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            # Make retriever chat-history-aware
            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

            # Prompt to answer question using retrieved context
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

            # Create final RAG chain
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            # Manage session-level chat history
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

        # Ask user question
        user_input = st.text_input("Ask a question about the PDF content:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            st.write("**Assistant:**", response["answer"])

            with st.expander("Chat History"):
                for msg in session_history.messages:
                    st.markdown(f"**{msg.type.title()}:** {msg.content}")

        # Clean up temp file
        os.remove(temp_pdf)
else:
    st.warning("Please enter your Groq API key to use the application.")

