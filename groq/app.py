import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
import time
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Load the Groq API key
groq_api_key = os.environ.get('GROQ_API_KEY')
if not groq_api_key:
    st.error("GROQ_API_KEY environment variable not set.")
    st.stop()

try:
    if "vector" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
except Exception as e:
    logger.error(f"Error during initialization: {e}")
    st.error(f"Error during initialization: {e}")
    st.stop()

st.title("Krishna - ChatGroq Demo")

try:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-It")
except Exception as e:
    logger.error(f"Failed to initialize ChatGroq: {e}")
    st.error(f"Failed to initialize ChatGroq: {e}")
    st.stop()

prompt_template = """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""

try:
    prompt = ChatPromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
except Exception as e:
    logger.error(f"Failed to create retrieval chain: {e}")
    st.error(f"Failed to create retrieval chain: {e}")
    st.stop()

prompt_input = st.text_input("Input your prompt here")

if prompt_input:
    try:
        start = time.process_time()
        response = retrieval_chain.invoke({"input": prompt_input})
        logger.info(f"Response time: {time.process_time() - start}")
        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    except Exception as e:
        logger.error(f"Error during prompt processing: {e}")
        st.error(f"Error during prompt processing: {e}")
