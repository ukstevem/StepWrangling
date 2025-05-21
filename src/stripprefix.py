import pandas as pd

# Load the CSV
df = pd.read_csv('dataset_labels_withprefix.csv')

# Fix the filenames by stripping the prefix before the first hyphen
df['image_path'] = df['image_path'].apply(lambda x: x.split('-', 1)[-1])

# (Optional) Save back to CSV if needed
df.to_csv('dataset_labels_cleaned.csv', index=False)
