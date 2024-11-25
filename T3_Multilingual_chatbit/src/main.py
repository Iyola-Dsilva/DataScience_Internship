import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db, detect_and_translate_to_english, translate_from_english
# session state for chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.set_page_config(page_title="Banking Chatbot", page_icon="🤖", layout="centered")
st.title("BANKING CHATBOT 🤖")
st.write("Ask your banking-related questions in any language!")

# Button to create a knowledge base
if st.button("Create Knowledge Base"):
    create_vector_db("dataset/Banking_Dataset.csv")
    st.success("Knowledge Base created successfully!")

# Input
question = st.text_input("Ask a Question:", placeholder="Type your question here...")

if question:
    chain = get_qa_chain()

    # Detect language and translate to English
    translated_query, detected_lang = detect_and_translate_to_english(question)

    response = chain({"query": translated_query})
    answer_in_english = response["result"]

    # Translate response back to the detected language
    final_answer = translate_from_english(answer_in_english, detected_lang)


    
    st.session_state["chat_history"].append({
        "language": detected_lang,
        "question": question,
        "answer": final_answer
    })

    # Display the current answer
    st.header("Answer")
    st.write(final_answer)

st.write("---")
st.header("Chat History")

if st.session_state["chat_history"]:
    for i, chat in enumerate(st.session_state["chat_history"], start=1):
        st.write(f"**{i}. Q ({chat['language']}):** {chat['question']}")
        st.write(f"**A:** {chat['answer']}")
        st.write("")
else:
    st.write("No chat history yet.")
