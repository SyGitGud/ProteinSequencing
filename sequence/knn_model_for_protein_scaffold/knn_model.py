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

#load trained model
import joblib
import numpy as np
import ast

# Load the saved KNN model
knn_classifier = joblib.load(model_path)



with open(int_to_char_path, "r") as int_to_char_file:
    INT_TO_CHAR = ast.literal_eval(int_to_char_file.read())  # Load dictionary from file

with open(char_to_int_path, "r") as char_to_int_file:
    CHAR_TO_INT = ast.literal_eval(char_to_int_file.read())

def sequence_to_numeric(sequence):
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
            if k in de_novo_sequence:
                    
                    # Convert de novo sequence and its reverse to numeric representation
                de_novo_sequence_numeric = sequence_to_numeric(k)

                X_de_novo = np.array([de_novo_sequence_numeric])
                # Make predictions for the de novo sequence reverse
                y_pred_de_novo = knn_classifier.predict(X_de_novo)
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











