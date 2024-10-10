import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import joblib
from sklearn.decomposition import TruncatedSVD

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
training_sequences = list(set(get_sequences("data/mab_training_sequence.txt")))
sequences_to_train_on = len(training_sequences)

all_chars = set("".join(training_sequences) + "-")
NUM_CLASSES = len(all_chars)
CHAR_TO_INT = {c: i for i, c in enumerate(all_chars, start=1)}
INT_TO_CHAR = {v: k for k, v in CHAR_TO_INT.items()}
with open("CHAR_TO_INT.txt", "w") as input_file:
    input_file.write(str(CHAR_TO_INT))
                         
with open("INT_TO_CHAR.txt", "w") as input_file:
    input_file.write(str(INT_TO_CHAR))


X_train_data = []
y_train_data = []
training_seq_dict= process_data(training_sequences)
for keys in training_seq_dict:
    X_train_data = X_train_data + [sequence_to_numeric(keys[0])]
    y_train_data = y_train_data + [keys[1]]


def KNN_model(X_train, y_train):  
    # Train the KNN classifier
    k = 5
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    # Use tqdm to visualize training progress
    with tqdm(total=len(X_train), desc="TrainingSequences") as pbar:
        knn_classifier.fit(np.array(X_train), y_train)
        pbar.update(len(X_train))

    # Save the KNN model
    joblib.dump(knn_classifier, 'knn_model.pkl')
    
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
knn_classifier = joblib.load('knn_model.pkl')

with open("CHAR_TO_INT.txt", "r") as char_to_int_file:
    CHAR_TO_INT = ast.literal_eval(char_to_int_file.read())  # Load dictionary from file

with open("INT_TO_CHAR.txt", "r") as int_to_char_file:
    INT_TO_CHAR = ast.literal_eval(int_to_char_file.read())  # Load dictionary from file

print("CHAR_TO_INT:", CHAR_TO_INT)
print("INT_TO_CHAR:", INT_TO_CHAR)

def sequence_to_numeric(sequence):
    return [CHAR_TO_INT[char] for char in sequence]

#gap filling

#mabcampath scaffold
de_novo_sequence = "---MTQSPSSLSASVGDRVTITCK---NIDKYLNWYQQKPGKAPKLLIYNTNNLQTGVPS\
RF---G----FTFTI-----------YCLQHISRPRTFGQGTKVEIKRTVAAPSVFIFPP\
SDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLT\
LSKADYEKHKVYACEVTHQGLSSPVTKSFN----"

new_seq = []
for count,  i in enumerate(range(len(de_novo_sequence) - 11 + 1)):
    kmer = de_novo_sequence[i:i + 11]
    new_seq = new_seq + [kmer]

while "-" in de_novo_sequence:
    keys_with_dash = [key for key in new_seq if key.count('-') == 1]
    if len(keys_with_dash) == 0:
        keys_with_dash = [key for key in new_seq if key.count('--') == 2]
    for k in keys_with_dash:
        if k[0] == "-" or k[10] == "-":
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
print("Predicted Sequence for De Novo:", de_novo_sequence)


def add_color(s, color):
    """Wrap string in ANSI color codes for terminal display."""
    colors = {
        "black": "\033[30m",   # Black text
        "green": "\033[32m",   # Green text
        "red": "\033[31m",     # Red text
        "reset": "\033[0m"     # Reset to default
    }
    return f'{colors[color]}{s}{colors["reset"]}'

def compare_sequences(target, denovo, predicted):
    """Compare target, denovo, and predicted sequences and assign colors based on conditions."""
    result = []
    
    for t, d, p in zip(target, denovo, predicted):
        if t == d == p:
            result.append(add_color(p, "reset"))  
        elif t == p and t != d:
            result.append(add_color(p, "green"))  
        else:
            result.append(add_color(p, "red"))  

    return "".join(result)


predicted = de_novo_sequence

target = "DIQMTQSPSSLSASVGDRVTITCKASQNIDKYLNWYQQKPGKAPKLLIYNTNNLQTGVPS\
RFSGSGSGTDFTFTISSLQPEDIATYYCLQHISRPRTFGQGTKVEIKRTVAAPSVFIFPP\
SDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLT\
LSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"

scaffold = "---MTQSPSSLSASVGDRVTITCK---NIDKYLNWYQQKPGKAPKLLIYNTNNLQTGVPS\
RF---G----FTFTI-----------YCLQHISRPRTFGQGTKVEIKRTVAAPSVFIFPP\
SDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLT\
LSKADYEKHKVYACEVTHQGLSSPVTKSFN----"


colored_predicted = compare_sequences(target, scaffold, predicted)
print(scaffold)
print(colored_predicted)
