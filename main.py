import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

template = """
You are an AI assisting with resume analysis.
Given the following extracted parts of a resume and a job description, provide a score for the resume based on the job description and give actionable suggestions for improving the resume to better match the job requirements.

context: \n{context}\n

question: \n{question}\n
Answer:"""

def get_conversational_chain():
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(input_variables=["question", "context"], template=template)
    chains = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chains

def load_faiss_index(pickle_file):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_index = FAISS.load_local(pickle_file, embeddings=embeddings, allow_dangerous_deserialization=True)
    return faiss_index

def analyze_resume(job_description, resume_text):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = load_faiss_index("faiss_index")
    docs = vector_store.similarity_search(resume_text)

    chain = get_conversational_chain()
    response = chain({
        "input_documents": docs,
        "question": f"Job Description: {job_description}\n\nResume: {resume_text}\n\nProvide a CV score, suggestions for improvement, and skills enhancement recommendations."
    })

    return response["output_text"]

st.set_page_config(
    page_title="Resume Analyzer",
    page_icon=':books:',
    layout="wide",
    initial_sidebar_state="auto"
)

with st.sidebar:
    st.title("Resume Analyzer")
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    job_description = st.text_area("Enter Job Description")
    resume_files = st.file_uploader("Upload Resume (PDF)", type=["pdf"], accept_multiple_files=True)
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)

    if st.button("Analyze"):
        with st.spinner("Analyzing Resume..."):
            resume_text = get_pdf_text(resume_files)
            text_chunks = get_text_chunks(resume_text)
            get_vector_store(text_chunks)
            st.success("VectorDB Uploading Successful!!")

def main():
    st.markdown("<hr>", unsafe_allow_html=True)
    st.header("🖇️ Resume and JD Analyzer 🗞️")
    st.markdown("<hr>", unsafe_allow_html=True)

    if job_description and resume_files:
        resume_text = get_pdf_text(resume_files)
        analysis_result = analyze_resume(job_description, resume_text)
        st.write("Analysis Result:")
        st.write(analysis_result)
    else:
        st.write("Please enter a job description and upload a resume PDF file for analysis.")

if __name__ == "__main__":
    main()
