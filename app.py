import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def get_pdf_docs(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    text_chunks = text_splitter.split_text(text=raw_text)
    return text_chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def main():
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
    st.header("Chat with Multiple PDFs")
    st.text_input(label="Ask a question about your documents")

    with st.sidebar:
        st.subheader("Your Docs")
        pdf_docs = st.file_uploader(label="Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        # if user clicks button then execute
        if st.button(label="Process"):
            # spinner appears while processing logic in the with statement
            with st.spinner(text="Processing"):
                # get pdf text
                pdfs_raw_text = get_pdf_docs(pdf_docs=pdf_docs)
                # get text chunks
                text_chunks = get_text_chunks(raw_text=pdfs_raw_text)
                # create vector store with embeddings
                get_vector_store(text_chunks)
                pass


if __name__ == "__main__":
    main()
