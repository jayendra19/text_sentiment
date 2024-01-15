import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification
# Convert labels to PyTorch tensors
import torch
# Create PyTorch datasets
from torch.utils.data import DataLoader, TensorDataset
import logging
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder



file=r"C:\Users\jayen\Text-sentiment-analysis-general-purpose\artifacts\Forsentiment.csv"
n=100000
df=pd.read_csv(file,nrows=n)
print(df)

df['label']=df['label'].replace('normal','neutral')

df=df.dropna(subset=['text'])

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])


# Load pre-trained RoBERTa model and tokenizer
model_name = "roberta-base"  # You can choose a different variant, e.g., "roberta-small"
tokenizer = RobertaTokenizer.from_pretrained(model_name,do_lower_case=True,max_length=250)#u can choose max_length from tokkenization 
model = RobertaForSequenceClassification.from_pretrained(model_name,num_labels=len(set(df['label'])))

train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'].to_list(), df['label'].to_list(), test_size=0.2, random_state=42)

# Convert the lists to lists of strings CUZ ITS SHOULD BE STRING FORM 
train_texts = [str(text) for text in train_texts]
test_texts = [str(text) for text in test_texts]


# Tokenize the input texts and convert them to PyTorch tensors
train_encodings = tokenizer(train_texts, truncation=True, padding=True,return_tensors='pt')
test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt')


train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'],torch.tensor(test_labels))

print("Train dataset:")
print("Number of samples:", len(train_dataset))
print("Input_ids shape:", train_dataset[0][0].shape)
print("Attention_mask shape:", train_dataset[0][1].shape)
print("Labels shape:", train_dataset[0][2].shape)

print("\nTest dataset:")
print("Number of samples:", len(test_dataset))
print("Input_ids shape:", test_dataset[0][0].shape)
print("Attention_mask shape:", test_dataset[0][1].shape)
print("Labels shape:", test_dataset[0][2].shape)

# Creating DataLoader for training and testing sets
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)


# Training loop (fine-tuning)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
epochs = 3


for epoch in range(epochs):
    model.train()
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", position=0, leave=True):
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Decode the predicted labels
decoded_preds = label_encoder.inverse_transform(all_preds)
decoded_labels = label_encoder.inverse_transform(all_labels)

# Calculate accuracy
accuracy = accuracy_score(decoded_labels, decoded_preds)
print(f'Accuracy: {accuracy * 100:.2f}%')







