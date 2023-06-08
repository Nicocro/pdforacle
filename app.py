from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os


def main():
    load_dotenv()
    openaikey = os.getenv('OPENAI_API_KEY')

    st.set_page_config(page_title="pdf Oracle")
    st.header("Chat with PDF Oracle")

    #uploading PDF File
    pdf = st.file_uploader("Upload your PDF Document", type="pdf")

    #extract text in the PDF file
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        #split the text extracted from PDF into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200, 
            length_function=len
        )

        chunks = text_splitter.split_text(text)

        #create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openaikey)
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        st.write(knowledge_base)



if __name__ == '__main__':
    main()