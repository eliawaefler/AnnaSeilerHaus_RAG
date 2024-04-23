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


def get_vectorstore(text_chunks):
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
            # Display source document information if available in the message
            if hasattr(message, 'source') and message.source:
                st.write(f"Source Document: {message.source}", unsafe_allow_html=True)


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

    st.header("Anna Seiler Haus KI-Assistent ASH :hospital:")
    if st.text_input("ASK_ASH_PASSWORD: ") == ASK_ASH_PASSWORD:
        OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
        # ASK_ASH_PASSWORD = False
        OPENAI_API_KEY = False
        OPENAI_ORG_ID = os.environ["OPENAI_ORG_ID"]
        PINECONE_API_KEY = os.environ["PINECONE_API_KEY_LCBIM"]
        HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
        VECTARA_CORPUS_ID = "3"
        VECTARA_API_KEY = os.environ["VECTARA_API_KEY"]
        VECTARA_CUSTOMER_ID = os.environ["VECTARA_CUSTOMER_ID"]

    user_question = st.text_input("Ask a question about your documents:")

    # st.session_state.openai = st.toggle(label="use openai?")
    # if st.session_state.openai:
    #     st.session_state.openai_key = st.text_input("openai api key", type="password")
    #     OPENAI_API_KEY = st.session_state.openai_key

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vec = get_vectorstore(text_chunks)
                st.session_state.vectorstore = vec
                st.session_state.conversation = get_conversation_chain(vec)

        # Save and Load Embeddings
        if st.button("Save Embeddings"):
            if "vectorstore" in st.session_state:
                st.session_state.vectorstore.save_local(str(datetime.now().strftime("%Y%m%d%H%M%S")) + "faiss_index")
                st.sidebar.success("saved")
            else:
                st.sidebar.warning("No embeddings to save. Please process documents first.")

        if st.button("Load Embeddings"):
            if "vectorstore" in st.session_state:
                new_db = FAISS.load_local()
                if new_db is not None:  # Check if this is working
                    combined_db = merge_faiss_indices(new_db, st.session_state.vectorstore)
                    st.session_state.vectorstore = combined_db
                    st.session_state.conversation = get_conversation_chain(combined_db)
                else:
                    st.sidebar.warning("Couldn't load embeddings")
            else:
                new_db = FAISS.load_local("faiss_index")
                if new_db is not None:  # Check if this is working
                    st.session_state.vectorstore = new_db
                    st.session_state.conversation = get_conversation_chain(new_db)


if __name__ == '__main__':
    ASK_ASH_PASSWORD = os.environ["ASK_ASH_PASSWORD"]
    main()
