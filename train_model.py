from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import DatasetDict, Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils import mask_pii  # Import the mask_pii function

# 1. Load and prepare data
df = pd.read_csv("emails.csv")

# 2. Mask PII in the email column
df["masked_email"] = df["email"].apply(lambda x: mask_pii(x)[0])  # Mask PII and keep the masked text

# 3. Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['type'])

# 4. Split data
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# 5. Convert to Hugging Face dataset format
dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'eval': Dataset.from_pandas(eval_df)
})

# 6. Tokenization
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    return tokenizer(
        examples["masked_email"],  # Use the masked email column
        truncation=True,
        padding='max_length',
        max_length=256
    )

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 7. Model Setup
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(le.classes_),
    id2label={i: label for i, label in enumerate(le.classes_)},
    label2id={label: i for i, label in enumerate(le.classes_)}
)

# 8. Training Configuration
# 8. Training Configuration
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",        # Save the model at the end of each epoch
    learning_rate=2e-5,
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="eval_loss",  # Use evaluation loss to determine the best model
    greater_is_better=False       # Lower evaluation loss is better
)

# 9. Trainer with Early Stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 10. Train and Save
trainer.train()
trainer.save_model("./email_classifier_bert")
tokenizer.save_pretrained("./email_classifier_bert")