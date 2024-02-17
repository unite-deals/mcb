import streamlit as st
import pdfplumber
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pyperclip
import pandas as pd

hide_github_link_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visiblity: hidden;}
    header {visibility: hidden;}
        .viewerBadge_container__1QSob {
            display: none !important;
        }
    </style>
"""
st.markdown(hide_github_link_style, unsafe_allow_html=True)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

custom_css = """
<style>
body {
    font-family: 'Arial', sans-serif;
    background-color: #f5f5f5;
    margin: 0;
    padding: 0;
}

.container {
    max-width: 800px;
    margin: 0 auto;
}

.header {
    background-color: #3498db;
    color: #ffffff;
    padding: 1rem;
    text-align: center;
    border-radius: 5px;
}

.sidebar {
    background-color: #2c3e50;
    color: #ffffff;
    padding: 1rem;
    border-radius: 5px;
}

.sidebar h3 {
    margin-bottom: 1.5rem;
}

.content {
    background-color: #ffffff;
    padding: 1rem;
    margin-top: 1rem;
    border-radius: 5px;
}

.button {
    background-color: #3498db;
    color: #ffffff;
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.button:hover {
    background-color: #2980b9;
}

</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Function to read PDF and extract text
def read_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to save email list to CSV
def save_to_csv(email_list, csv_filename="email_list.csv"):
    df = pd.DataFrame(email_list, columns=["Email"])
    df.to_csv(csv_filename, index=False)


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100000, chunk_overlap=0)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    expert and capable to read multiple languages .Answer the question as detailed as possible from the provided context if possible add some key points in list format , make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context & add your suggestion is differet subtitle also", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# Main Streamlit app
def main():
    # Header
    st.header("Chat with PDF using Gemini💁")
    st.markdown("Welcome to the interactive PDF chat application.")

    # Sidebar
    st.sidebar.title("Menu:")
    pdf_docs = st.sidebar.file_uploader("Upload your PDF File and Click on the Submit & Process Button", type=["pdf"])

    # Email input box
    email_input = st.sidebar.text_input("Enter your email:")

    if st.sidebar.button("Submit Email"):
        st.info(f"Email submitted: {email_input}")

        # Assuming you want to save the email to a list for further processing
        email_list = []  # Replace this with your actual list
        email_list.append(email_input)

        # Save email list to CSV
        save_to_csv(email_list)

    if st.sidebar.button("Submit & Process"):
        if pdf_docs is not None:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
        else:
            st.warning("Please upload a PDF file.")
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    # Main content
    st.text_area("Type your question here:", key="user_input")

    if st.button("Submit Question"):
        user_question = st.session_state.user_input

        if user_question:
            user_input(user_question, [])

            # Display AI reply in the chat layout
            st.text("AI Bot:")
            st.text(response["output_text"])

    # Button to copy to clipboard
    if st.button("Copy to Clipboard", key="copy_button"):
        pyperclip.copy(response["output_text"])
        st.success("Copied to Clipboard!")

if __name__ == "__main__":
    main()
