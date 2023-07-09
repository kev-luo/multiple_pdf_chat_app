import streamlit as st
from PyPDF2 import PdfReader


def get_pdf_docs(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text


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
                # create vector store with embeddings
                pass


if __name__ == "__main__":
    main()
