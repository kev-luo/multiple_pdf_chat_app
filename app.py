import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from html_templates import css, bot_template, user_template


def get_final_doc(pdf_doc):
    loader = OnlinePDFLoader(pdf_doc)
    pages = loader.load_and_split()
    return pages


def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    text_chunks = text_splitter.split_documents(raw_text)
    return text_chunks


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
    return vector_store


def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), memory=memory
    )
    return conversation_chain


# def handle_user_input(user_question):
#     response = st.session_state.conversation({"question": user_question})
#     st.write(response)
#     pass


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with Multiple PDFs")
    user_question = st.text_input(label="Ask a question about your documents")

    # if user_question:
    #     handle_user_input(user_question)

    # st.write(user_template.replace("{{MSG}}", "yo what's good"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "Hello human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your Docs")
        # pdf_docs = st.file_uploader(label="Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        pdf_doc = "http://www.bussigel.com/systemsforplay/wp-content/uploads/2014/05/Hackers-and-Painters.pdf"
        # if user clicks button then execute
        if st.button(label="Process"):
            # spinner appears while processing logic in the with statement
            with st.spinner(text="Processing"):
                # get pdf text
                final_doc = get_final_doc(pdf_doc)
                # get text chunks
                text_chunks = get_text_chunks(final_doc)
                # create vector store with embeddings
                # vector_store = get_vector_store(text_chunks)
                # create conversation chain
                # st.session_state.conversation = get_conversation_chain(vector_store)

                st.write(text_chunks)


if __name__ == "__main__":
    main()
