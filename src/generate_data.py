import pandas as pd
import random

# Sample data to replicate
sample_data = [
    ("Free money now!", "spam"),
    ("Meeting tomorrow at 3 PM", "ham"),
    ("Win a free iPhone!", "spam"),
    ("Important update on your account", "ham"),
    ("Congratulations! You've won a prize", "spam"),
    ("Let's catch up this weekend", "ham"),
    ("Exclusive offer just for you", "spam"),
    ("Your invoice is attached", "ham"),
]

# Function to generate dataset
def generate_dataset(num_samples):
    data = []
    
    for _ in range(num_samples):
        subject, label = random.choice(sample_data)
        # Add some randomness by changing the subject slightly
        if random.random() > 0.5:
            subject = subject.upper()
        else:
            subject = subject.lower()

        data.append([subject, label])
    
    return data

# Generate 40,000 samples
num_samples = 40000
generated_data = generate_dataset(num_samples)

# Create a DataFrame
df = pd.DataFrame(generated_data, columns=["subject", "label"])

# Save to CSV
df.to_csv('./Data/spam_dataset.csv', index=False)

print(f"Generated {num_samples} samples and saved to 'data/spam_dataset.csv'")