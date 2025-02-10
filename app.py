import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up Streamlit UI
st.set_page_config(page_title="Document Search Chatbot")
st.header(" 🔍 Document Search Chatbot 👋👋  I am your chatbot 💬 , Ask me a question ❓ 🤖")

# Load the document
loader = TextLoader(r"C:\Users\MAN DHANANI\OneDrive\Desktop\LLM Based Project\Document Search Chatbot Textfile\data.txt")
docs = loader.load()

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docsearch = FAISS.from_documents(docs, embeddings)

# Initialize Groq model
llm = ChatGroq(model="llama3-8b-8192", temperature=0.0)

# Create Retrieval QA Chain
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

# User input for query
user_query = st.text_input("Ask me a question:", "")

# Button to search
if st.button("Search"):
    if user_query:
        response = qa.run(user_query)
        st.subheader("Answer:")
        st.write(response)
    else:
        st.write("Please ask a question.")
