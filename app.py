import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import fitz

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Google API Key is missing! Set the GOOGLE_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=api_key)

# Load PDF text
def load_pdf_text(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as pdf:
            for page_num in range(pdf.page_count):
                text += pdf.load_page(page_num).get_text()
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
    return text

# Load and process PDF content
pdf_file_path = "gla_data.pdf"  
if not os.path.exists(pdf_file_path):
    st.error(f"PDF file '{pdf_file_path}' not found!")
    st.stop()

gla_data_text = load_pdf_text(pdf_file_path)

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

# Create FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Ensure FAISS index exists
if not os.path.exists("faiss_index"):
    st.info("Generating FAISS vector store...")
    text_chunks = get_text_chunks(gla_data_text)
    get_vector_store(text_chunks)
    st.success("Vector store created successfully!")

# Conversational chain setup
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, say "answer is not available in the context".
    Context:
    {context}
    Question:
    {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Handle user input and search FAISS index
def user_input(user_question):
    try:
        if not isinstance(user_question, str) or not user_question.strip():
            st.error("Invalid input! Please enter a valid question.")
            return

        # Append query to search history
        st.session_state.search_history.append(user_question)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        if not os.path.exists("faiss_index"):
            st.error("FAISS index not found! Generating a new index.")
            text_chunks = get_text_chunks(gla_data_text)
            get_vector_store(text_chunks)

        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        docs = new_db.similarity_search(user_question)
        if not docs:
            st.warning("No relevant information found.")
            return

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        formatted_reply = response.get("output_text", "No response generated.").strip()

        st.markdown("**Reply:**")
        st.markdown(formatted_reply)

    except Exception as e:
        st.error(f"Error processing query: {e}")

# Main function
def main():
    st.set_page_config(page_title="GLA ChatBot", layout="wide")
    st.header("GLA ChatBot Help & Support 🧑‍💻")

    # Initialize search history
    if "search_history" not in st.session_state:
        st.session_state.search_history = []

    # User input
    user_question = st.text_input("Ask a Question from the GLA Data")

    if user_question:
        user_input(user_question)

    # Sidebar with search history
    with st.sidebar:
        st.title("Search History")
        if st.session_state.search_history:
            for idx, query in enumerate(st.session_state.search_history, 1):
                st.write(f"{idx}. {query}")
        else:
            st.write("No searches yet.")

if __name__ == "__main__":
    main()
