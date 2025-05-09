from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from utils import mask_pii
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the saved model and tokenizer
model_path = "./email_classifier_bert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load label encoder
df = pd.read_csv("emails.csv")
le = LabelEncoder()
le.fit(df['type'])

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def classify_email(email):
    # Mask PII in the email
    masked_email, _ = mask_pii(email)
    
    # Tokenize the email
    inputs = tokenizer(
        masked_email,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors="pt"
    )
    
    # Move inputs to the appropriate device
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
    
    # Decode the predicted class
    predicted_label = le.inverse_transform([predicted_class])[0]
    return predicted_label

if __name__ == "__main__":
    sample_email = input("Enter a sample email: ")
    # Classify the sample email
    result = classify_email(sample_email)
    print(f"The email is classified as: {result}")
