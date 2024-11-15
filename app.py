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


# Load environment variables (Google API key)
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

import fitz  # PyMuPDF

def load_pdf_text(pdf_path):
    # Open the PDF and extract text from each page
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf.load_page(page_num)
            text += page.get_text()
    return text

# Specify the PDF file path
pdf_file_path = "gla_data.pdf"  # Ensure the file is in the same directory or provide the correct path

# Load and print the PDF text
gla_data_text = load_pdf_text(pdf_file_path)
print(gla_data_text)




def get_text_chunks(text):
    # Split the static text into chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # Embed the text chunks into a vector store using Google's embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    # Define the prompt for answering questions based on the context
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    
    # Use Google's ChatGoogleGenerativeAI for the LLM model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    # Reload the vector store and find relevant documents based on user input
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # Use the conversational chain to generate a response
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    # Output the response formatted in bullet points
    reply = response["output_text"]
    
    # Format the response into bullet points
    bullet_points = reply.splitlines()
    formatted_reply = "\n".join(f"- {point.strip()}" for point in bullet_points if point.strip())

    # Output the response using Markdown for bullet points
    # st.markdown("**Reply:**")
    st.markdown(formatted_reply)

def main():
    # Configure Streamlit page
    st.set_page_config("Chat PDF")
    st.header("GLA ChatBot Help & Support 💁")

    # User input section
    user_question = st.text_input("Ask a Question from the GLA Data")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        if st.button("Process Data"):
            with st.spinner("Processing..."):
                # Instead of processing PDF, process the static GLA data
                text_chunks = get_text_chunks(gla_data)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
