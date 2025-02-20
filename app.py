import streamlit as st
import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Document Search Chatbot")
st.header(" 🔍 Document Search Chatbot 👋👋  I am your chatbot 💬 , Ask me a question ❓ 🤖")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Extract text from PDF
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])

    # Debugging: Check if text is extracted
    if not text.strip():
        st.error("No text found in the PDF. Please upload a different file.")
        st.stop()

    # Split text into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(text)

    # Debugging: Check if texts list is populated
    if not texts:
        st.error("Text chunking failed. No data to process.")
        st.stop()

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Debugging: Check if embeddings are working
    if not embeddings:
        st.error("Embeddings could not be initialized.")
        st.stop()

    # Create FAISS vectorstore
    docsearch = FAISS.from_texts(texts, embeddings)

    # Initialize Groq model
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.0)

    # Create Retrieval QA Chain
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

    user_query = st.text_input("Ask me a question:", "")

    if st.button("Search"):
        if user_query:
            response = qa.run(user_query)
            st.subheader("Answer:")
            st.write(response)
        else:
            st.write("Please ask a question.")
