import os
import warnings
import logging
import uuid
import time
import streamlit as st

# Phase 2 libraries
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# Disable warnings and info logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title('Ask PDF')

# Setup a session state variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Show old messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Vectorstore function from file path
@st.cache_resource
def get_vectorstore_from_file(file_path):
    try:
        loaders = [PyPDFLoader(file_path)]
        index = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        ).from_loaders(loaders)
        return index.vectorstore
    except Exception as e:
        st.error(f"Failed to load and process PDF: {str(e)}")
        return None

# Prompt input
prompt = st.chat_input("Enter your question about the PDF")

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    if uploaded_file is None:
        st.warning("Please upload a PDF file before asking a question.")
        st.stop()

    # Save the uploaded file safely
    temp_file_path = f"temp_{uuid.uuid4().hex}.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing the document..."):
        vectorstore = get_vectorstore_from_file(temp_file_path)

    # Delete temp file if needed
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    if not vectorstore:
        st.error("Vectorstore creation failed. Try another PDF.")
        st.stop()

    try:
        # Ensure API key is present
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY not found. Please set it in your environment.")
            st.stop()

        model = "llama3-8b-8192"
        groq_chat = ChatGroq(groq_api_key=api_key, model_name=model)

        # Create retrieval chain
        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type='stuff',
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True
        )

        # Show assistant response like typing
        with st.chat_message('assistant'):
            msg_placeholder = st.empty()
            full_response = ""
            with st.spinner("Generating answer..."):
                result = chain({"query": prompt})
                response = result.get("result", "No response generated.")

                # Typing animation effect
                for chunk in response:
                    full_response += chunk
                    msg_placeholder.markdown(full_response + "▌")  # Blinking cursor effect
                    time.sleep(0.01)  # Adjust typing speed here
                msg_placeholder.markdown(full_response)  # Final response without cursor

        # Append the response to session state
        st.session_state.messages.append({'role': 'assistant', 'content': full_response})

    except Exception as e:
        st.error(f"Unexpected error during processing: {str(e)}")

