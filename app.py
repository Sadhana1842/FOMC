from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load pre-trained fine-tuned model and tokenizer from Hugging Face's public model hub
model_name = "Respair/deberta-v3-large-finetuned-style"  # Using Respair's fine-tuned DeBERTa model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define sentiment analysis pipeline (no need for manual device handling)
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Function to process the statement
def process_statement(statement):
    lines = statement.split('.')
    sentiments = []
    
    for line in lines:
        line = line.strip()
        if line:  # Avoid empty lines
            result = sentiment_analyzer(line)
            sentiments.append(result[0]['label'].lower())
    
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
import streamlit as st

st.title("Sentiment Analysis App")
st.write("Enter a statement below to analyze its sentiment:")

# Input Text Area
user_input = st.text_area("Your statement:", placeholder="Type here...")

# Button to analyze sentiment
if st.button("Analyze Sentiment"):
    if user_input.strip():
        result = process_statement(user_input)
        st.subheader("Sentiment Analysis Results:")
        st.write(f"**Positive Percentage:** {result['Positive Percentage']:.2f}%")
        st.write(f"**Negative Percentage:** {result['Negative Percentage']:.2f}%")
        st.write(f"**Neutral Percentage:** {result['Neutral Percentage']:.2f}%")
        st.write(f"**Overall Sentiment:** {result['Overall Sentiment'].capitalize()}")
    else:
        st.warning("Please enter a statement to analyze.")
