import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from apscheduler.schedulers.background import BackgroundScheduler
import time


# Set Google API key
os.environ['GOOGLE_API_KEY'] = "AIzaSyDe7ZhEt2qkDSwo3UB2IY9tY5713IdfSss"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Define LLM and embeddings
llm = ChatGoogleGenerativeAI(model="gemini-pro")
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

# Vector database file path
vectordb_file_path = "faiss_index"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_vector_db(csv_file_path):
    """
    Create a vector database from a CSV file.
    """
    loader = CSVLoader(file_path=csv_file_path, source_column="prompt")
    data = loader.load()

    if not data:
        logger.error("No data loaded from the CSV file.")
        raise ValueError("No data loaded from the CSV file")

    # Create FAISS vector database
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)
    logger.info("Vector database created successfully.")

def update_vector_db(csv_file_path):
    """
    Update the vector database with new data.
    """
    if not os.path.exists(vectordb_file_path):
        logger.info("Vector database does not exist, creating a new one.")
        create_vector_db(csv_file_path)
    else:
        loader = CSVLoader(file_path=csv_file_path, source_column="prompt")
        new_data = loader.load()

        if new_data:
            vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)
            vectordb.add_documents(new_data)
            vectordb.save_local(vectordb_file_path)
            logger.info(f"Vector database updated with {len(new_data)} new entries.")
        else:
            logger.info("No new data to update.")

def get_qa_chain():
    """
    Load the vector database and return a QA chain.
    """
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    return chain

def update_data_periodically():
    """
    Periodically check for new data and update the vector database.
    """
    csv_file_path = "C:/DataScience_Internship/T2_Chatbot/dataset/Banking_Dataset.csv"
    try:
        update_vector_db(csv_file_path)
    except Exception as e:
        logger.error(f"Error updating vector database: {e}")

# Initialize the scheduler to run the update function periodically
scheduler = BackgroundScheduler()
scheduler.add_job(update_data_periodically, 'interval', minutes=5)  # Update period
scheduler.start()

if __name__ == "__main__":
    # Initial setup of the vector database (first time only)
    csv_file_path = "C:/DataScience_Internship/T2_Chatbot/dataset/Banking_Dataset.csv"
    create_vector_db(csv_file_path)

    # Start the chatbot
    chain = get_qa_chain()
    logger.info("Chatbot is now running. Type your queries below.")
    
    try:
        while True:
            user_query = input("Ask a question: ")
            response = chain({"query": user_query})
            print(response)
    except KeyboardInterrupt:
        logger.info("Shutting down the chatbot.")
        scheduler.shutdown()
