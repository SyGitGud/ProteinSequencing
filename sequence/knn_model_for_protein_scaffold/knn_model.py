import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import joblib
from sklearn.decomposition import TruncatedSVD
import django
import os

running_file = os.path.dirname(os.path.abspath(__file__))

project_dir = os.path.join(running_file, '../..')  # Move up two levels
print(project_dir)
os.chdir(project_dir)
print(project_dir)
data_file_path = os.path.join(running_file, "data", "mab_training_sequence.txt")
model_path = os.path.join(running_file, "knn_model.pkl")
char_to_int_path = os.path.join(running_file, "CHAR_TO_INT.txt")
int_to_char_path = os.path.join(running_file, "INT_TO_CHAR.txt")


print("Current working directory:", os.getcwd())
# Function to convert sequences to numeric representation
def sequence_to_numeric(sequence):
    return [CHAR_TO_INT[char] for char in sequence]

# Function to convert numeric representation back to sequence
def numeric_to_sequence(numeric_seq):
    return "".join([INT_TO_CHAR[num] for num in numeric_seq])

def remove_duplicate(word_list):
    unique_words = set()
    result = []

    for word in word_list:
        if word not in unique_words:
            unique_words.add(word)
            result.append(word)

    return result

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


def process_data(sequences):
    input_output_pairs = []
    for seq in sequences:
        for start in range(len(seq)-11):
            end = start + 11
            seq_in = seq[start:end]
            temp = seq_in[0:10]+ "-"
            seq_out = seq_in[10]
            input_output_pairs.append((temp, seq_out))
            temp = "-" + seq_in[1:11]
            seq_out = seq_in[0]
            input_output_pairs.append((temp, seq_out))
    return input_output_pairs

print("loading data")
# Load training sequences

print("Data file path:", data_file_path)
training_sequences = list(set(get_sequences(data_file_path)))
sequences_to_train_on = len(training_sequences)

all_chars = set("".join(training_sequences) + "-")
NUM_CLASSES = len(all_chars)
CHAR_TO_INT = {c: i for i, c in enumerate(all_chars, start=1)}
INT_TO_CHAR = {v: k for k, v in CHAR_TO_INT.items()}


with open(char_to_int_path, "w") as input_file:
    input_file.write(str(CHAR_TO_INT))



with open(int_to_char_path, "w") as input_file:
    input_file.write(str(INT_TO_CHAR))


X_train_data = []
y_train_data = []
training_seq_dict= process_data(training_sequences)
for keys in training_seq_dict:
    X_train_data = X_train_data + [sequence_to_numeric(keys[0])]
    y_train_data = y_train_data + [keys[1]]


def KNN_model(X_train, y_train):  
    # Train the KNN classifier
    k = 6
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    # Use tqdm to visualize training progress
    with tqdm(total=len(X_train), desc="TrainingSequences") as pbar:
        knn_classifier.fit(np.array(X_train), y_train)
        pbar.update(len(X_train))

    # Save the KNN model
    joblib.dump(knn_classifier, model_path)
    
    return knn_classifier
    #data split
max_seq_length = max(len(seq) for seq in X_train_data)

# Pad training sequences to the same length
padded_training_sequences_numeric = [seq + [0] * (max_seq_length - len(seq)) for seq in X_train_data]

X_train = np.array(padded_training_sequences_numeric)
y_train = np.array(y_train_data)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

knn_model = KNN_model(X_train,y_train)

predicted_labels = knn_model.predict(X_train)
accuracy_train = accuracy_score(y_train, predicted_labels)
print("training Accuracy :", accuracy_train)
y_pred = knn_model.predict(X_val)
accuracy =  accuracy_score(y_val, y_pred)
print("testing Accuracy :",accuracy)

#load trained model
import joblib
import numpy as np
import ast

# Load the saved KNN model
knn_classifier = joblib.load(model_path)



with open(int_to_char_path, "r") as int_to_char_file:
    INT_TO_CHAR = ast.literal_eval(int_to_char_file.read())  # Load dictionary from file



def sequence_to_numeric(sequence):
    with open(char_to_int_path, "r") as char_to_int_file:
        CHAR_TO_INT = ast.literal_eval(char_to_int_file.read())  # Load dictionary from file
    numeric_representation = []
    for char in sequence:
        if char not in CHAR_TO_INT:
            print(f"Warning: Character '{char}' not found in CHAR_TO_INT.")
            numeric_representation.append(0)  # Placeholder or raise an error as needed
        else:
            numeric_representation.append(CHAR_TO_INT[char])
    print("Numeric representation of sequence:", numeric_representation)
    return numeric_representation

#gap filling

#mabcampath scaffold


def predict_seq(de_novo_sequence):
    print("Initial de_novo_sequence:", de_novo_sequence)
    de_novo_sequence_numerics = sequence_to_numeric(de_novo_sequence)
    print("Numeric representation for prediction:", de_novo_sequence_numerics)

    X_de_novos = np.array([de_novo_sequence_numerics])
    print("Shape of input to model:", X_de_novos.shape)

    # Make predictions

    new_seq = []
    for count,  i in enumerate(range(len(de_novo_sequence) - 11 + 1)):
        kmer = de_novo_sequence[i:i + 11]
        new_seq = new_seq + [kmer]

    print("New kmers", new_seq)



    while "-" in de_novo_sequence:
        keys_with_dash = []
        print("Content", new_seq)
        if any(key.count('-') >= 1 and '--' not in key for key in new_seq):
            keys_with_dash = [key for key in new_seq if key.count('-') >= 1 and '--' not in key]
            print("keys_with_dash (multiple non-consecutive '-'): ", keys_with_dash)

        if any(key.count('--') >= 1 and '---' not in key for key in new_seq):
            keys_with_dash = [key for key in new_seq if key.count('--') >= 1 and '---' not in key]
            print("keys_with_dash (multiple non-consecutive '--'): ", keys_with_dash)

        print(keys_with_dash)

        for k in keys_with_dash:
            print("yay")
            print("sucess")
            if k in de_novo_sequence:
                    
                    # Convert de novo sequence and its reverse to numeric representation
                de_novo_sequence_numeric = sequence_to_numeric(k)

                X_de_novo = np.array([de_novo_sequence_numeric])
                # Make predictions for the de novo sequence reverse
                y_pred_de_novo = knn_model.predict(X_de_novo)
                # Convert the predicted labels back to sequences for verification
                predicted_value = y_pred_de_novo[0]
                index = de_novo_sequence.index(k)
                index1 = k.index("-")
                if 0 <= (index + index1) < len(de_novo_sequence):
                    de_novo_sequence = de_novo_sequence[:(index + index1)] + predicted_value +de_novo_sequence[(index + index1) + 1:]

                # Update new_seq after filling a gap
                new_seq.clear()
                for count, i in enumerate(range(len(de_novo_sequence) - 11 + 1)):
                    kmer = de_novo_sequence[i:i + 11]
                    new_seq = new_seq + [kmer]
                keys_with_dash = [key for key in new_seq if key.count('-') == 1]
                if len(keys_with_dash) == 0:
                    keys_with_dash = [key for key in new_seq if key.count('--') == 2]
            
        # Print the predicted sequence for the de novo sequence
    return de_novo_sequence








