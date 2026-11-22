
import pandas as pd

df = pd.read_csv("dataset/emails.csv")

print("\nFirst 5 Emails:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nEmail Type Distribution:")
print(df["Email Type"].value_counts())

print("\nDataset inspection completed!")
