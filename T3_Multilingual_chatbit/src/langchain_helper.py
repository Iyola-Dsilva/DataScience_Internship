import os
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from apscheduler.schedulers.background import BackgroundScheduler
from langdetect import detect,DetectorFactory
from googletrans import Translator
import time

# Load environment variables
load_dotenv()

# Set Google API key
os.environ['GOOGLE_API_KEY'] = "your_google_api_key"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Define LLM and embeddings
llm = ChatGoogleGenerativeAI(model="gemini-pro")
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

# Vector database file path
vectordb_file_path = "faiss_index"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize translation tools
translator = Translator()

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

DetectorFactory.seed = 0
translator = Translator()

# Allowed languages
ALLOWED_LANGUAGES = {
    "en": "English",
    "kn": "Kannada",
    "hi": "Hindi",
    "fr": "French"
}

def detect_and_translate_to_english(text):
    """
    Limits detection to English, Kannada, Hindi, and French.
    """
    try:
        # Direct handling for common greetings
        greetings = {
            "hi": "en", "hello": "en", "hey": "en",
            "namaste": "hi", "bonjour": "fr", "hola": "en",  # English fallback for 'hola'
            "ನಮಸ್ಕಾರ": "kn", "ಹಲೋ": "kn"  # Kannada greetings
        }
        if text.lower() in greetings:
            detected_lang = greetings[text.lower()]
            return text, detected_lang

        # Detect the language
        detected_lang = detect(text)

        # Restrict to allowed languages
        if detected_lang not in ALLOWED_LANGUAGES:
            detected_lang = "en"  # Default to English for unsupported languages

        # Translate to English if needed
        if detected_lang != "en":
            translated = translator.translate(text, src=detected_lang, dest="en").text
            return translated, detected_lang

        return text, "en"  # Already in English
    except Exception as e:
        # Default to English if detection fails
        return text, "en"
    
def translate_from_english(text, target_lang):
    """
    Translates English text to the target language if it's one of the allowed languages.
    """
    if target_lang == "en" or target_lang not in ALLOWED_LANGUAGES:
        return text  # No translation needed for English or unsupported languages
    try:
        return translator.translate(text, src="en", dest=target_lang).text
    except Exception as e:
        return text  # Fallback to original text if translation fails


def update_data_periodically():
    """
    Periodically check for new data and update the vector database.
    """
    csv_file_path = "dataset/Banking_Dataset.csv"
    try:
        update_vector_db(csv_file_path)
    except Exception as e:
        logger.error(f"Error updating vector database: {e}")

# Initialize the scheduler to run the update function periodically
scheduler = BackgroundScheduler()
scheduler.add_job(update_data_periodically, 'interval', minutes=2)
scheduler.start()

if __name__ == "__main__":
    # Initial setup of the vector database (first time only)
    csv_file_path = "dataset/Banking_Dataset.csv"
    create_vector_db(csv_file_path)

    # Start the chatbot
    chain = get_qa_chain()
    logger.info("Chatbot is now running. Type your queries below.")

    try:
        while True:
            user_query = input("Ask a question in any language: ")
            # Detect and translate input
            translated_query, detected_lang = detect_and_translate_to_english(user_query)

            # Query the chatbot
            response = chain({"query": translated_query})

            # Translate and adapt response
            final_response = translate_from_english(response['result'], detected_lang)
            
            print(f"Response ({detected_lang}): {final_response}")
    except KeyboardInterrupt:
        logger.info("Shutting down the chatbot.")
        scheduler.shutdown()
