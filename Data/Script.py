import pandas as pd

file1 = "Updated_Spam_emails.csv" 
file2 = "spam_dataset.csv" 

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

merged_df = pd.concat([df1, df2], ignore_index=True)

output_file = "merged_Spam_emails_file.csv" 
merged_df.to_csv(output_file, index=False)

print(f"CSV files merged successfully and saved as {output_file}")