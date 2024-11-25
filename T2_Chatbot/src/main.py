import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.title("BANKING CHATBOT 🤖")

# Button to create a knowledge base
btn = st.button("Create Knowledge Base")
if btn:
    create_vector_db("dataset/Banking_Dataset.csv")
    st.success("Knowledge Base created successfully!")

# Input for user question
question = st.text_input("Ask a Question:")

if question:
    # Get the QA chain and response
    chain = get_qa_chain()
    response = chain.invoke(question)
    answer = response["result"]

    # Store question and answer in chat history
    st.session_state["chat_history"].append({"question": question, "answer": answer})

    # Display current answer
    st.header("Answer")
    st.write(answer)

# Display chat history
st.write("---")
st.header("Chat History")
for chat in st.session_state["chat_history"]:
    st.write(f"**Q:** {chat['question']}")
    st.write(f"**A:** {chat['answer']}")
    st.write("")  # Add a blank line for spacing
