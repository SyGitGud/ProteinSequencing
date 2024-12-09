import numpy as np
import os
import joblib
import ast
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def calc_accuracy():
    return accuracy_train, accuracy

# Paths for saving model and mappings
running_file = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(running_file, '../..')  # Move up two levels
data_file_path = os.path.join(running_file, "data", "mab_training_sequence.txt")
model_path = os.path.join(running_file, "knn_model.pkl")
char_to_int_path = os.path.join(running_file, "CHAR_TO_INT.txt")
int_to_char_path = os.path.join(running_file, "INT_TO_CHAR.txt")
curr_dir = os.getcwd()
print(curr_dir)
def get_sequences(file_name):
    sequences = []
    lines = []
    with open(file_name, "r") as input_file:
        lines = list(filter(None, input_file.read().split("\n")))

    parts = []
    for line in lines:
        if line.startswith(">"):
            if parts:
                sequences.append("".join(parts))
            parts = []
        else:
            parts.append(line)
    if parts:
        sequences.append("".join(parts))
    return sequences
# Function to convert sequences to numeric representation
def sequence_to_numeric(sequence):
    return [CHAR_TO_INT[char] for char in sequence]

# Function to convert numeric representation back to sequence
def numeric_to_sequence(numeric_seq):
    return "".join([INT_TO_CHAR[num] for num in numeric_seq])

# Function to process training data
def process_data(sequences):
    input_output_pairs = []
    for seq in sequences:
        for start in range(len(seq) - 11):
            end = start + 11
            seq_in = seq[start:end]
            temp = seq_in[0:10] + "-"
            seq_out = seq_in[10]
            input_output_pairs.append((temp, seq_out))
            temp = "-" + seq_in[1:11]
            seq_out = seq_in[0]
            input_output_pairs.append((temp, seq_out))
    return input_output_pairs

# Load training sequences
print("Loading data...")
training_sequences = list(set(get_sequences(data_file_path)))
sequences_to_train_on = len(training_sequences)

all_chars = set("".join(training_sequences) + "-")
NUM_CLASSES = len(all_chars)
CHAR_TO_INT = {c: i for i, c in enumerate(all_chars, start=1)}
INT_TO_CHAR = {v: k for k, v in CHAR_TO_INT.items()}

# Save CHAR_TO_INT and INT_TO_CHAR mappings to files
with open(char_to_int_path, "w") as file:
    file.write(str(CHAR_TO_INT))

with open(int_to_char_path, "w") as file:
    file.write(str(INT_TO_CHAR))

# Prepare training data
X_train_data = []
y_train_data = []
training_seq_dict = process_data(training_sequences)
for keys in training_seq_dict:
    X_train_data += [sequence_to_numeric(keys[0])]
    y_train_data += [keys[1]]

# Pad sequences to the same length
max_seq_length = max(len(seq) for seq in X_train_data)
padded_training_sequences_numeric = [seq + [0] * (max_seq_length - len(seq)) for seq in X_train_data]

X_train = np.array(padded_training_sequences_numeric)
y_train = np.array(y_train_data)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Train the KNN model
def KNN_model(X_train, y_train):  
    # Train the KNN classifier
    for k in [3, 5, 6]:
        knn_classifier = KNeighborsClassifier(n_neighbors=k)

        # Use tqdm to visualize training progress
        with tqdm(total=len(X_train), desc="TrainingSequences") as pbar:
            knn_classifier.fit(np.array(X_train), y_train)
            pbar.update(len(X_train))

        # Save the KNN model
        joblib.dump(knn_classifier, model_path)
        
    return knn_classifier
    #data split

# Train and save the model
knn_model = KNN_model(X_train, y_train)

# Evaluate the model
predicted_labels = knn_model.predict(X_train)
accuracy_train = accuracy_score(y_train, predicted_labels)
print("Training Accuracy:", accuracy_train)

y_pred = knn_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Testing Accuracy:", accuracy)

