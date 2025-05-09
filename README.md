# Email-Classifier-after-PII-masking

This project provides tools for masking Personally Identifiable Information (PII) in emails and classifying emails into categories (e.g., spam or not spam) using a fine-tuned BERT model.

Features
Email Masking: Detects and masks sensitive information such as phone numbers, email addresses, and dates.
Email Classification: Classifies emails into predefined categories using a fine-tuned BERT model.
Prerequisites
Python 3.10 (other versions are not supported)
pip (Python package manager)
Setup Instructions
1. Clone the Repository
Clone the repository to your local machine:

2. Install Python 3.10
Download and install Python 3.10 from the official Python website. Ensure you add Python to your system's PATH during installation.

3. Create a Virtual Environment

Activate the virtual environment:
4. Install Dependencies
Install the required dependencies using the requirements.txt file:

5. Download SpaCy Model
Download the en_core_web_sm model for SpaCy:

6. Download the Trained Model
The trained BERT model is too large to include in the repository. Download it from the following link:
link : https://drive.google.com/drive/folders/1ipb9d2AjVoKc_-FIIiTud6_8tGbHt_R5?usp=sharing
Download Trained Model

After downloading, extract the model and place it in the project directory under the folder email_classifier_bert.

7. Ensure Data Availability
Ensure the emails.csv file is present in the project directory. This file is used to load label encodings for classification.

Usage Instructions
1. Test Email Masking
To test the email masking functionality, run the test_masking.py 

This script will mask PII (e.g., phone numbers, email addresses) in a sample email and display the masked text along with detected entities.
2. Test Email Classification

Run the check_sample_email.py script:

Enter a sample email when prompted (e.g., "Enter a sample email: ").

View the classification result. The script will display the predicted category of the email.
Thank you.
