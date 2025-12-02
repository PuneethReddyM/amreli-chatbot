import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.stem import PorterStemmer
import random
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize stemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Reduced overfitting

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l3(out)
        return out

# Enhanced data augmentation
def augment_patterns(patterns):
    augmented = patterns.copy()
    
    synonyms = {
        'schemes': ['programs', 'initiatives', 'benefits', 'yojana'],
        'list': ['show', 'display', 'get', 'find'],
        'education': ['school', 'student', 'study', 'learning'],
        'agriculture': ['farming', 'crops', 'kisan', 'farmer'],
        'healthcare': ['health', 'medical', 'hospital', 'treatment'],
        'housing': ['home', 'house', 'residence', 'accommodation'],
        'women': ['female', 'ladies', 'girls'],
        'employment': ['job', 'work', 'employment', 'career'],
        'business': ['enterprise', 'startup', 'company', 'venture'],
        'senior': ['elderly', 'old', 'aged', 'pensioner'],
        'digital': ['online', 'internet', 'technology', 'e-governance'],
        'financial': ['banking', 'money', 'finance', 'economic'],
        'rural': ['village', 'countryside', 'gramin'],
        'energy': ['power', 'electricity', 'solar', 'renewable']
    }
    
    for pattern in patterns:
        words = pattern.split()
        for i, word in enumerate(words):
            if word.lower() in synonyms:
                for synonym in synonyms[word.lower()]:
                    new_pattern = words.copy()
                    new_pattern[i] = synonym
                    augmented.append(' '.join(new_pattern))
    
    return list(set(augmented))  # Remove duplicates

# Load and enhance intents
print("📚 Loading and enhancing training data...")
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Enhance patterns with data augmentation
for intent in intents['intents']:
    original_patterns = intent['patterns']
    enhanced_patterns = augment_patterns(original_patterns)
    intent['patterns'] = enhanced_patterns
    print(f"Enhanced {intent['tag']}: {len(original_patterns)} -> {len(enhanced_patterns)} patterns")

all_words = []
tags = []
xy = []

ignore_words = ['?', '!', '.', ',', "'"]

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(f"\n📊 Enhanced Dataset Statistics:")
print(f"Total patterns: {len(xy)}")
print(f"Unique tags: {len(tags)}")
print(f"Unique words: {len(all_words)}")
print(f"Tags: {tags}")

# Create training data
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Enhanced hyperparameters
num_epochs = 2000
batch_size = 16
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 16  # Increased hidden size
output_size = len(tags)

print(f"\n🏗️ Model Architecture:")
print(f"Input size: {input_size}")
print(f"Hidden size: {hidden_size}")
print(f"Output size: {output_size}")

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # L2 regularization

print("\n🚀 Starting training...")
losses = []

for epoch in range(num_epochs):
    epoch_loss = 0
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    losses.append(epoch_loss / len(train_loader))
    
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {losses[-1]:.4f}')

print(f'Final loss: {losses[-1]:.4f}')

# Save the enhanced model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags,
    "intents": intents
}

FILE = "data_improved.pth"
torch.save(data, FILE)
print(f'\n✅ Training complete! Model saved to {FILE}')

# Test the model
def predict_class(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).unsqueeze(0)
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    return tag, prob.item()

# Test with sample inputs
test_sentences = [
    "show me education schemes",
    "what agriculture programs are available",
    "health benefits",
    "house schemes",
    "women empowerment initiatives",
    "job programs",
    "business support",
    "senior citizen benefits",
    "digital services",
    "banking schemes",
    "village development",
    "solar energy programs"
]

print("\n🧪 Testing model with sample inputs:")
for test_sentence in test_sentences:
    tag, prob = predict_class(test_sentence)
    print(f"'{test_sentence}' -> {tag} (confidence: {prob:.3f})")