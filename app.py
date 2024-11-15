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
import fitz  # PyMuPDF

# Load environment variables (Google API key)
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load PDF text
def load_pdf_text(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf.load_page(page_num)
            text += page.get_text()
    return text

# Load PDF content
pdf_file_path = "gla_data.pdf"  # Path to your PDF file
gla_data_text = load_pdf_text(pdf_file_path)

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Create vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, just say,
    "answer is not available in the context", and don't provide a wrong answer.\n\n
    Context:\n {context}?\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Handle user input
def user_input(user_question, search_history):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    reply = response["output_text"]
    formatted_reply = "\n".join(line.strip() for line in reply.splitlines() if line.strip())

    # Add question to search history
    search_history.append(user_question)

    # Display the response
    st.markdown("**Reply:**")
    st.markdown(formatted_reply)

# Main function
def main():
    # Configure Streamlit page
    st.set_page_config("Chat PDF")
    st.header("GLA ChatBot Help & Support 💁")

    # Initialize search history (persisted across user sessions)
    if "search_history" not in st.session_state:
        st.session_state.search_history = []

    # User input section
    user_question = st.text_input("Ask a Question from the GLA Data")

    if user_question:
        user_input(user_question, st.session_state.search_history)

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
