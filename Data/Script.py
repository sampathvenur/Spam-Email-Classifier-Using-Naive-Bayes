import pandas as pd

# File paths for the two CSV files
file1 = "Updated_Spam_emails.csv"  # Replace with the path to the first CSV file
file2 = "spam_dataset.csv"  # Replace with the path to the second CSV file

# Load both CSV files into DataFrames
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Merge the DataFrames by appending rows
merged_df = pd.concat([df1, df2], ignore_index=True)

# Save the merged DataFrame to a new CSV file
output_file = "merged_Spam_emails_file.csv"  # Replace with the desired output file name
merged_df.to_csv(output_file, index=False)

print(f"CSV files merged successfully and saved as {output_file}")