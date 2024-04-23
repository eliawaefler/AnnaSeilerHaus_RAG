import time
from datetime import datetime
import openai
import tiktoken
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from html_templates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import os
import numpy as np
import requests
import json


def save_vectorstore(vectorstore):
    """Merges new data into the Vectara corpus."""
    url = f"{BASE_URL}/corpus/{VECTARA_CUSTOMER_ID}/corpus/{VECTARA_CORPUS_ID}/documents"
    response = requests.post(url, headers=headers, data=json.dumps(vectorstore))
    return response.json()


def get_vectorstore():
    """Retrieves vector store from Vectara corpus."""
    url = f"{BASE_URL}/corpus/{VECTARA_CUSTOMER_ID}/corpus/{VECTARA_CORPUS_ID}/documents"
    response = requests.get(url, headers=headers)
    return response.json()


def add_documents(files):
    """Adds documents to the Vectara corpus after vectorizing them."""
    url = f"{BASE_URL}/corpus/{VECTARA_CUSTOMER_ID}/corpus/{VECTARA_CORPUS_ID}/documents"
    for file in files:
        # Example: Assuming each file is text and needs simple handling
        # Vectorization logic depends on the specific requirements or APIs available
        document = {
            "content": file.getvalue().decode(),
            "metadata": {"filename": file.name}
        }
        response = requests.post(url, headers=headers, data=json.dumps(document))
        print(response.text)  # Handling the response based on your logic


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_faiss_vectorstore(text_chunks):
    if st.session_state.openai:
        my_embeddings = OpenAIEmbeddings()
    else:
        my_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=my_embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    if st.session_state.openai:
        llm = ChatOpenAI()
    else:
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        # Display user message
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            print(message)
            # Display AI response
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Anna Seiler Haus KI-Assistent", page_icon=":hospital:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "openai" not in st.session_state:
        st.session_state.openai = True
    if "login" not in st.session_state:
        st.session_state.login = False

    st.header("Anna Seiler Haus KI-Assistent ASH :hospital:")
    st.session_state.login = (st.text_input("ASK_ASH_PASSWORD: ", type="password") == ASK_ASH_PASSWORD)

    if st.session_state.login:
        # ASK_ASH_PASSWORD = False
        # OPENAI_API_KEY = False
        global BASE_URL
        BASE_URL = "https://api.vectara.io/v1"
        global OPENAI_API_KEY
        OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
        global OPENAI_ORG_ID
        OPENAI_ORG_ID = os.environ["OPENAI_ORG_ID"]
        global PINECONE_API_KEY
        PINECONE_API_KEY = os.environ["PINECONE_API_KEY_LCBIM"]
        global HUGGINGFACEHUB_API_TOKEN
        HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
        global VECTARA_API_KEY
        VECTARA_API_KEY = os.environ["VECTARA_API_KEY"]
        global VECTARA_CUSTOMER_ID
        VECTARA_CUSTOMER_ID = os.environ["VECTARA_CUSTOMER_ID"]
        global headers
        headers = {"Authorization": f"Bearer {VECTARA_API_KEY}", "Content-Type": "application/json"}
        st.write("welcome")
    else:
        st.write("not logged in.")
    user_question = st.text_input("Ask a question about your documents:")

    # st.session_state.openai = st.toggle(label="use openai?")
    # if st.session_state.openai:
    #     st.session_state.openai_key = st.text_input("openai api key", type="password")
    #     OPENAI_API_KEY = st.session_state.openai_key

    if user_question:
        handle_userinput(user_question)
    if st.session_state.login:
        get_vectorstore()
        with st.sidebar:
            st.subheader("Your documents")
            pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
            if st.button("Process"):
                with st.spinner("Processing"):
                    # raw_text = get_pdf_text(pdf_docs)
                    # text_chunks = get_text_chunks(raw_text)
                    vec = get_vectorstore()
                    st.session_state.vectorstore = vec
                    st.session_state.conversation = get_conversation_chain(vec)
                st.success("added to vectorstore")

            # Save and Load Embeddings
            if st.button("Save Embeddings"):
                if "vectorstore" in st.session_state:
                    save_vectorstore(st.session_state.vectorstore)
                    st.sidebar.success("saved")
                else:
                    st.sidebar.warning("No embeddings to save. Please process documents first.")


if __name__ == '__main__':
    # Constants from the environment
    BASE_URL = False
    OPENAI_API_KEY = False
    OPENAI_ORG_ID = False
    PINECONE_API_KEY = False
    HUGGINGFACEHUB_API_TOKEN = False
    VECTARA_CORPUS_ID = "3"
    VECTARA_API_KEY = False
    VECTARA_CUSTOMER_ID = False
    headers = False
    ASK_ASH_PASSWORD = os.environ["ASK_ASH_PASSWORD"]
    main()
