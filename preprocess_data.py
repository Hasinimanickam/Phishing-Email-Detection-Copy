
import pandas as pd

# Load dataset
df = pd.read_csv("dataset/emails.csv")

print("Original Dataset:", df.shape)

# Remove unnecessary column
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Remove missing values
df = df.dropna(subset=["Email Text", "Email Type"])

# Remove duplicate emails
df = df.drop_duplicates(subset=["Email Text"])

# Clean email text
df["Email Text"] = df["Email Text"].astype(str).str.strip()

# Remove empty emails
df = df[df["Email Text"] != ""]

# Reset index
df = df.reset_index(drop=True)

# Save cleaned dataset
df.to_csv("dataset/cleaned_emails.csv", index=False)

print("\nCleaned Dataset:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nEmail Distribution:")
print(df["Email Type"].value_counts())

print("\nCleaned dataset saved successfully!")
