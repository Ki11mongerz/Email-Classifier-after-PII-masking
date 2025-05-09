from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset
import torch

# 1. Load the saved model and tokenizer
model_path = "./email_classifier_bert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 2. Load the evaluation dataset
df = pd.read_csv("emails.csv")

# Mask PII in the email column (if needed)
from utils import mask_pii
df["masked_email"] = df["email"].apply(lambda x: mask_pii(x)[0])

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label'] = le.fit_transform(df['type'])

# Split the data (ensure the same split as during training)
_, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert evaluation data to Hugging Face Dataset format
eval_dataset = Dataset.from_pandas(eval_df)

# 3. Tokenize the evaluation dataset
def preprocess_function(examples):
    return tokenizer(
        examples["masked_email"],
        truncation=True,
        padding='max_length',
        max_length=256
    )

tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Convert tokenized dataset to PyTorch DataLoader
from torch.utils.data import DataLoader

# Hugging Face datasets need to be converted to PyTorch tensors
def collate_fn(batch):
    return {
        "input_ids": torch.tensor([item["input_ids"] for item in batch]),
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch]),
        "labels": torch.tensor([item["label"] for item in batch]),
    }

eval_dataloader = DataLoader(tokenized_eval_dataset, batch_size=8, collate_fn=collate_fn)

# 4. Run predictions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_predictions(model, dataloader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
            outputs = model(**inputs)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())
    return predictions, true_labels

predictions, true_labels = get_predictions(model, eval_dataloader)

# 5. Generate performance report
print("Accuracy:", accuracy_score(true_labels, predictions))
print("\nClassification Report:\n", classification_report(true_labels, predictions, target_names=le.classes_))