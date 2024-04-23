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


def set_user():
    st.session_state.user_pw = True
    st.session_state.page = "home"
    st.empty()
    display_home()


def display_sing_up():
    st.title("sing up")
    st.session_state.username = st.text_input("email")
    st.session_state.username = st.text_input("username")
    st.session_state.user_pw = st.text_input("password", type="password", on_change=set_user)
    if st.button("back"):
        st.session_state.page = "home"
        st.empty()
        display_home()


def display_sing_in():
    st.empty()
    st.title("sign in")
    st.title(f"Welcome {st.session_state.username}")
    if st.session_state.user_pw:
        st.session_state.page = "user_page"
    st.session_state.username = st.text_input("username")
    st.session_state.user_pw = st.text_input("password", type="password", on_change=set_user)
    if st.button("forgot password"):
        st.empty()
        st.session_state.page = "forgot_pw"
        display_forgot_pw()
    if st.button("back"):
        st.session_state.page = "home"
        st.empty()
        display_home()


def display_forgot_pw():
    st.title("forgot pw")
    st.text_input("email or username")
    if st.button("send link"):
        st.empty()
        st.session_state.page = "home"
        display_home()
    if st.button("back"):
        st.session_state.page = "sign_in"
        st.empty()
        display_sing_in()


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


def display_home():
    st.empty()
    st.header("Anna Seiler Haus KI-Assistent ASH :hospital:")

    if st.session_state.username:
        st.subheader("welcome " + str(st.session_state.username))

    col1, col2, col3 = st.columns([5, 1, 1])

    with col1:
        user_question = st.text_input("Ask a question about your documents:")
        st.session_state.openai = st.toggle(label="use openai?")
        if st.session_state.openai:
            st.session_state.openai_key = st.text_input("openai api key", type="password")
            OPENAI_API_KEY = st.session_state.openai_key
        if user_question:
            handle_userinput(user_question)

    if not st.session_state.user_pw:
        with col3:
            st.write("")
            st.write("")
            if st.button("login"):
                st.session_state.page = "sign_in"
                st.empty()
                display_sing_in()

    else:
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
                    st.session_state.vectorstore.save_local("faiss_index")
                    st.sidebar.success("saved")
                else:
                    st.sidebar.warning("No embeddings to save. Please process documents first.")

            if st.button("Load Embeddings"):
                new_db = None
                if "vectorstore" in st.session_state:
                    new_db = FAISS.load_local("faiss_index", )
                if new_db is not None:  # Check if this is working
                    st.session_state.vectorstore = new_db
                    st.session_state.conversation = get_conversation_chain(new_db)
                else:
                    st.sidebar.warning("Couldn't load embeddings")


def main():
    st.set_page_config(page_title="Anna Seiler Haus KI-Assistent", page_icon=":hospital:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "openai" not in st.session_state:
        st.session_state.openai = False
    if "openai_key" not in st.session_state:
        st.session_state.openai_key = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "user_pw" not in st.session_state:
        st.session_state.user_pw = False
    if "page" not in st.session_state:
        st.session_state.page = "home"
    display_home()


if __name__ == '__main__':

    # OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    OPENAI_API_KEY = False
    OPENAI_ORG_ID = os.environ["OPENAI_ORG_ID"]
    PINECONE_API_KEY = os.environ["PINECONE_API_KEY_LCBIM"]
    HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
    main()
