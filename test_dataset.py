
import pandas as pd

# Read the dataset
data = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Display the first 5 rows
print(data.head())

# Display total number of messages
print("\nTotal messages:", len(data))

# Count spam and genuine messages
print("\nMessage categories:")
print(data["label"].value_counts())

print("\nDataset information:")
print(data.info())

print("\nMissing values:")
print(data.isnull().sum())