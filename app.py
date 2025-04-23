from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import mask_pii
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging

# Initialize the FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the BERT-based model and tokenizer
try:
    model_path = "./email_classifier_bert"  # Path to the saved model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set the model to evaluation mode
    logging.info("BERT model and tokenizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or tokenizer: {str(e)}")
    raise RuntimeError("Failed to load model or tokenizer. Ensure the files exist and are valid.")

# Define the request schema
class EmailBatchRequest(BaseModel):
    email_bodies: list[str]

@app.post("/classify_batch")
async def classify_emails(request: EmailBatchRequest):
    try:
        # Validate input
        if not request.email_bodies:
            raise HTTPException(status_code=400, detail="Email bodies cannot be empty.")

        # Initialize response list
        response = []

        # Process each email
        for email in request.email_bodies:
            print(f"Processing email: {email}")  # Debug log
            masked_email, entities = mask_pii(email)
            print(f"Masked Email: {masked_email}")  # Debug log
            print(f"Entities: {entities}")  # Debug log

            # Tokenize the masked email
            inputs = tokenizer(masked_email, return_tensors="pt", truncation=True, padding=True, max_length=256)
            print(f"Tokenized Inputs: {inputs}")  # Debug log

            # Move inputs to the same device as the model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            inputs = {key: val.to(device) for key, val in inputs.items()}

            # Perform predictions
            with torch.no_grad():
                outputs = model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=-1).cpu().item()

            # Define a mapping from category IDs to human-readable labels
            category_mapping = {0: "Change", 1: "Incident", 2: "Problem", 3: "Request"}
            category = category_mapping[prediction]

            # Format the response for this email
            email_response = {
                "input_email_body": email,
                "list_of_masked_entities": entities,
                "masked_email": masked_email,
                "category_of_the_email": category
            }
            response.append(email_response)

        return response

    except Exception as e:
        print(f"Error: {str(e)}")  # Debug log
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")