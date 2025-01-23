import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the DeBERTa model and tokenizer
model_name = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Assuming 3 labels: positive, negative, neutral
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Label map
label_map = {0: "neutral", 1: "positive", 2: "negative"}

# Function to analyze sentiment for a single line
def analyze_sentiment(line):
    inputs = tokenizer(line, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits).item()
    return label_map[predicted_label]

# Function to process the statement
def process_statement(statement):
    lines = statement.split('.')
    sentiments = [analyze_sentiment(line.strip()) for line in lines if line.strip()]  # Avoid empty lines

    positive_count = sentiments.count('positive')
    negative_count = sentiments.count('negative')
    neutral_count = sentiments.count('neutral')
    total = len(sentiments)

    positive_percentage = (positive_count / total) * 100 if total > 0 else 0
    negative_percentage = (negative_count / total) * 100 if total > 0 else 0
    neutral_percentage = (neutral_count / total) * 100 if total > 0 else 0

    overall_sentiment = max(sentiments, key=sentiments.count) if sentiments else "neutral"

    return {
        "Positive Percentage": positive_percentage,
        "Negative Percentage": negative_percentage,
        "Neutral Percentage": neutral_percentage,
        "Overall Sentiment": overall_sentiment
    }

# Streamlit app
st.title("Sentiment Analysis with DeBERTa")
st.write("Enter a statement to analyze its sentiment.")

# Input form
with st.form("sentiment_form"):
    statement = st.text_area("Enter your statement here:")
    submitted = st.form_submit_button("Analyze Sentiment")

# Display results
if submitted and statement:
    result = process_statement(statement)
    st.subheader("Sentiment Analysis Results:")
    st.write(f"**Positive Percentage:** {result['Positive Percentage']:.2f}%")
    st.write(f"**Negative Percentage:** {result['Negative Percentage']:.2f}%")
    st.write(f"**Neutral Percentage:** {result['Neutral Percentage']:.2f}%")
    st.write(f"**Overall Sentiment:** {result['Overall Sentiment']}")
