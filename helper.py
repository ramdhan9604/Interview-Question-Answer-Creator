from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import prompt
import re

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


def file_processing(file_path):
    """Extracts and processes text from a PDF."""
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = "".join([page.page_content for page in data])

    splitter_ques_gen = TokenTextSplitter(chunk_size=10000, chunk_overlap=200)
    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)
    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    splitter_ans_gen = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
    document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)

    return document_ques_gen, document_answer_gen


def llm_pipeline(file_path):
    """Generates questions and answers from the given PDF file."""


    
    document_ques_gen, document_answer_gen = file_processing(file_path)

     
    llm_ques_gen_pipeline = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)

    PROMPT_QUESTIONS = PromptTemplate(template=prompt.prompt_template, input_variables=["text"])
    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=prompt.refine_template,
    )

    ques_gen_chain = load_summarize_chain(
        llm=llm_ques_gen_pipeline,
        chain_type="refine",
        verbose=False,
        question_prompt=PROMPT_QUESTIONS,
        refine_prompt=REFINE_PROMPT_QUESTIONS,
    )

    ques = ques_gen_chain.run(document_ques_gen)

    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    
    llm_answer_gen = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)

    filtered_ques_list = re.findall(r'\d+\.\s(.*?\?)', ques)

    answer_generation_chain = RetrievalQA.from_chain_type(
        llm=llm_answer_gen, 
        chain_type="stuff", 
        retriever=vector_store.as_retriever()
    )

    return answer_generation_chain, filtered_ques_list