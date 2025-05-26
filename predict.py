import joblib
from preprocessing import clear_text

# Load the trained model and vectorizer
model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

def predict_sentiment(text):
    clean_text = clear_text(text)
    vector = vectorizer.transform([clean_text])
    pred = model.predict(vector)[0]
    return "Positive" if pred == 1 else "Negative"

# Test
if __name__ == "__main__":
    text_usuario = input("Enter a review: ")
    resultado = predict_sentiment(text_usuario)
    print("Prediction:", resultado)
