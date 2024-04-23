import time
from datetime import datetime
import openai
import tiktoken
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from html_templates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import os
import numpy as np
import requests
import json

if True:
    BASE_URL = False
    OPENAI_API_KEY = False
    OPENAI_ORG_ID = False
    PINECONE_API_KEY = False
    HUGGINGFACEHUB_API_TOKEN = False
    VECTARA_CORPUS_ID = "3"
    VECTARA_API_KEY = False
    VECTARA_CUSTOMER_ID = False
    headers = False
    ASK_ASH_PASSWORD = False


def set_global_variables():
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


# not working yet. (where get info from?)
def vectara_index(document, doc_id, doc_title, metadataJson):
    url = "https://api.vectara.io/v1/index"
    payload = json.dumps({
        "customerId": str(VECTARA_CUSTOMER_ID),
        "corpusId": VECTARA_CORPUS_ID,
        "document": {
            "documentId": "string",
            "title": "string",
            "description": "string",
            "metadataJson": "string",
            "section": [
                {
                    "id": 0,
                    "title": "string",
                    "text": "string",
                    "metadataJson": "string",
                    "section": [
                        None
                    ]
                }
            ]
        }
    })
    my_headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'x-api-key': VECTARA_API_KEY
    }
    response = requests.request("POST", url, headers=my_headers, data=payload)
    print(response.text)


def vectara_add_document(payload, metadata_json=False):
    url = "https://api.vectara.io/v1/upload"
    my_headers = {
        'Content-Type': 'multipart/form-data',
        'Accept': 'application/json',
        'x-api-key': VECTARA_API_KEY
    }
    if metadata_json:
        payload = {"doc_metadata": metadata_json, "file": payload}
    response = requests.request("POST", url, headers=my_headers, data=payload)
    print(response.text)


def parse_api_response(response):
    result_list = []
    for item in response['responseSet']:
        # Extract relevant data from each part
        status_info = [{'code': s.get('code', None), 'detail': s.get('statusDetail', None)} for s in item['status']]
        summary_info = [{'text': s.get('text', None), 'lang': s.get('lang', None)} for s in item['summary']]
        # Create a simplified dictionary for each response item
        result_dict = {
            'status': status_info,
            'futureId': item.get('futureId', None),
            'summary': summary_info
        }
        result_list.append(result_dict)
    return result_list


def vectara_query(query):
    url = "https://api.vectara.io/v1/query"

    payload = json.dumps({
        "query": [
            {
                "query": query,
                "start": 0,
                "numResults": 1,
                "contextConfig": {
                    "sentences_before": 3,
                    "sentences_after": 3,
                    "start_tag": "<b>",
                    "end_tag": "</b>"
                },
                "corpusKey": [
                    {
                        "corpus_id": VECTARA_CORPUS_ID
                    }
                ],
                "summary": [
                    {
                        "max_summarized_results": 1,
                        "response_lang": "de"
                    }
                ]
            }
        ]
    })
    my_headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'x-api-key': VECTARA_API_KEY
    }
    response = requests.request("POST", url, headers=my_headers, data=payload)
    return response.json()["responseSet"][0]["response"][0]


def handle_userinput(user_query):
    retrieved_info = vectara_query(user_query)
    st.write(str(retrieved_info["text"]))


def main():
    st.set_page_config(page_title="Anna Seiler Haus KI-Assistent", page_icon=":hospital:")
    st.write(css, unsafe_allow_html=True)
    if True:
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
        set_global_variables()
        # st.session_state.vectorstore = get_vectorstore()
        # st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
        st.write("welcome")
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)

        with st.sidebar:
            st.subheader("Add document(s)")
            pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
            if st.button("Process"):
                with st.spinner("Processing"):
                    for doc in pdf_docs:
                        vectara_add_document(doc)
                st.success("added to vectorstore")

    else:
        st.write("not logged in.")

    # st.session_state.openai = st.toggle(label="use openai?")
    # if st.session_state.openai:
    #     st.session_state.openai_key = st.text_input("openai api key", type="password")
    #     OPENAI_API_KEY = st.session_state.openai_key


def test():
    VECTARA_API_KEY = os.environ["VECTARA_API_KEY"]
    VECTARA_CUSTOMER_ID = os.environ["VECTARA_CUSTOMER_ID"]
    a = vectara_query("wer hat die lüftung eingebaut?")
    print(a)


if __name__ == '__main__':
    ASK_ASH_PASSWORD = os.environ["ASK_ASH_PASSWORD"]

    main()
