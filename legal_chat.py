import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Create a Streamlit title
st.title("Legal Question Answering App")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
model = AutoModelForQuestionAnswering.from_pretrained("law-ai/InLegalBERT")

# Add text input for the context
context = st.text_area("Enter the legal context:", "In a recent legal case, the plaintiff accused the defendant of breach of contract.")

# Add text input for the question
question = st.text_input("Enter your legal question:", "What was the accusation in the case?")

# Tokenize and process the input
input_text = f"Question: {question} Context: {context}"
inputs = tokenizer(input_text, return_tensors="pt")

# Perform question-answering
start_scores, end_scores = model(**inputs)
start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores)
answer_span = input_text[inputs.input_ids[0][start_index]:inputs.input_ids[0][end_index] + 1]
answer = tokenizer.decode(answer_span)

# Display the answer
st.subheader("Answer:")
st.write(answer)
