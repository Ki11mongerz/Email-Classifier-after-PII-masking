from utils import mask_pii

# Sample email text
sample_text = """
I am Apoorva ANSH. My phone is +91-9876543210 and my email is ap@op.ao and I was born in 02/09/2900.
"""

masked_text, entities = mask_pii(sample_text)

print("Masked Text:")
print(masked_text)
print("\nDetected Entities:")
for entity in entities:
    print(entity)