import tkinter as tk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import numpy as np

nltk.download('punkt')
nltk.download('punkt_tab')

# 4 Model functions 
def preprocess_text(text):
    sentences = sent_tokenize(text)
    return sentences

def compute_tfidf(sentences):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return tfidf_matrix

def score_sentences(tfidf_matrix):
    scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
    return scores

def generate_summary(text):
    sentences = preprocess_text(text)
    tfidf_matrix = compute_tfidf(sentences)
    scores = score_sentences(tfidf_matrix)
    average_score = np.mean(scores)

    summary_sentences = [sentence for sentence, score in zip(sentences, scores) if score >= average_score]
    summary = ' '.join(summary_sentences)

    return summary

# Tkinter GUI
def on_generate_summary():
    input_text = text_input.get("1.0", "end-1c")  # Get text from input field
    if input_text.strip():
        summary = generate_summary(input_text)
        text_output.delete("1.0", "end")  # Clear the output field
        text_output.insert("1.0", summary)  # Display the summary
    else:
        messagebox.showwarning("Input Error", "Please enter some text for summarization.")

# main window
root = tk.Tk()
root.title("Text Summarizer")

# input text field
text_input_label = tk.Label(root, text="Enter Text for Summarization:")
text_input_label.pack(pady=10)

text_input = ScrolledText(root, width=60, height=10)
text_input.pack(pady=10)

# summary button
summary_button = tk.Button(root, text="Generate Summary", command=on_generate_summary)
summary_button.pack(pady=10)

# output text field
text_output_label = tk.Label(root, text="Summary:")
text_output_label.pack(pady=10)

text_output = ScrolledText(root, width=60, height=10)
text_output.pack(pady=10)

root.mainloop()
