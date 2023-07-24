from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import numpy as np

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your contract?")
    
    # uplaod a PDF
    pdf = st.file_uploader("Upload your contract here", type="PDF")
    
    # extract text
    if pdf is not None:
        # embeddings = load_knowledge_base(pdf.name)
        
        pdf_reader = PdfReader(pdf)
        text = ""
          
        i = 0      
        while i < 4:
            text += pdf_reader.pages[i].extract_text()
            i += 1            
            
        # st.write(text)
        
        # chunk text into result
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # st.write(chunks)
        
        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # show user input
        user_question = st.text_input("Ask a question about your contract:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            # st.write(docs)   
            
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question = user_question)
            
            st.write(response)

def create_knowledge_base(file_name, text):
    text_splitter = CharacterTextSplitter(
            separator="\n",      
            chunk_size=1000,
            chunk_overlap=200,            
            length_function=len
        )
    chunks = text_splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    np.save(file_name + '.npy', knowledge_base)      

def load_knowledge_base(file_name):
    return np.load(file_name + '.npy')

if __name__ == '__main__':
    main()