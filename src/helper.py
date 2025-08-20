from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
import re
from dotenv import load_dotenv
from src.prompt import *

# OpenAI authentication
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def file_processing(file_path):
    """
    Process the PDF file and create document chunks for question generation and answer retrieval.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        tuple: (document_question_gen, document_answer_gen)
    """
    try:
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        # Load PDF
        loader = PyPDFLoader(file_path)
        data = loader.load()
        
        if not data:
            raise ValueError("No content found in the PDF file")
        
        # Combine all pages for question generation
        question_gen = ""
        for page in data:
            if page.page_content.strip():  # Only add non-empty pages
                question_gen += page.page_content + "\n"
        
        if not question_gen.strip():
            raise ValueError("No readable content found in the PDF file")
        
        # Split text for question generation (larger chunks)
        splitter_question_gen = TokenTextSplitter(
            model_name="gpt-3.5-turbo",
            chunk_size=10000,
            chunk_overlap=200
        )
        
        chunk_question_gen = splitter_question_gen.split_text(question_gen)
        document_question_gen = [Document(page_content=t) for t in chunk_question_gen]
        
        # Split documents for answer generation (smaller chunks for better retrieval)
        splitter_ans_gen = TokenTextSplitter(
            model_name="gpt-3.5-turbo",
            chunk_size=1000,
            chunk_overlap=100
        )
        
        document_answer_gen = splitter_ans_gen.split_documents(document_question_gen)
        
        print(f"Created {len(document_question_gen)} chunks for question generation")
        print(f"Created {len(document_answer_gen)} chunks for answer generation")
        
        return document_question_gen, document_answer_gen
        
    except Exception as e:
        print(f"Error in file_processing: {str(e)}")
        raise e

def clean_and_filter_questions(questions_text):
    """
    Clean and filter the generated questions.
    
    Args:
        questions_text (str): Raw questions text from LLM
        
    Returns:
        list: Filtered list of valid questions
    """
    if not questions_text:
        return []
    
    # Split by newlines and clean each line
    questions_list = [q.strip() for q in questions_text.split("\n") if q.strip()]
    
    # Filter for valid questions
    filtered_questions = []
    for question in questions_list:
        # Remove numbering (1., 2., etc.)
        clean_question = re.sub(r'^\d+\.?\s*', '', question).strip()
        
        # Check if it's a valid question
        if (clean_question and 
            (clean_question.endswith('?') or clean_question.endswith('.')) and
            len(clean_question) > 10 and  # Minimum length check
            any(keyword in clean_question.lower() for keyword in 
                ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'describe', 'explain', 'discuss'])):
            filtered_questions.append(clean_question)
    
    return filtered_questions[:20]  # Limit to 20 questions to avoid overwhelming

def llm_pipeline(file_path):
    """
    Main pipeline for generating questions and setting up answer generation chain.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        tuple: (answer_generation_chain, filtered_questions_list)
    """
    try:
        print(f"Processing file: {file_path}")
        
        # Process the file
        document_question_gen, document_answer_gen = file_processing(file_path)
        
        # Initialize LLM for question generation
        llm_ques_gen_pipeline = ChatOpenAI(
            model='gpt-3.5-turbo',
            temperature=0.3,
            max_tokens=2000
        )
        
        # Set up prompts for question generation
        PROMPT_QUESTION = PromptTemplate(
            template=prompt_template, 
            input_variables=['text']
        )
        
        REFINE_PROMPT_QUESTIONS = PromptTemplate(
            input_variables=['existing_answer', 'text'],
            template=refine_template
        )
        
        # Create question generation chain
        ques_gen_chain = load_summarize_chain(
            llm=llm_ques_gen_pipeline,
            chain_type="refine",
            verbose=True,
            question_prompt=PROMPT_QUESTION,
            refine_prompt=REFINE_PROMPT_QUESTIONS
        )
        
        print("Generating questions...")
        questions = ques_gen_chain.run(document_question_gen)
        
        # Clean and filter questions
        filtered_questions_list = clean_and_filter_questions(questions)
        
        if not filtered_questions_list:
            raise ValueError("No valid questions were generated from the document")
        
        print(f"Generated {len(filtered_questions_list)} questions")
        
        # Create embeddings and vector store for answer generation
        print("Creating vector store for answer generation...")
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(document_answer_gen, embeddings)
        
        # Initialize LLM for answer generation
        llm_answer_gen = ChatOpenAI(
            temperature=0.1, 
            model="gpt-3.5-turbo",
            max_tokens=1000
        )
        
        # Create answer generation chain
        answer_generation_chain = RetrievalQA.from_chain_type(
            llm=llm_answer_gen,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=False
        )
        
        print("Pipeline setup complete!")
        return answer_generation_chain, filtered_questions_list
        
    except Exception as e:
        print(f"Error in llm_pipeline: {str(e)}")
        raise e

def validate_environment():
    """
    Validate that all required environment variables and dependencies are available.
    """
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    print("Environment validation successful!")
