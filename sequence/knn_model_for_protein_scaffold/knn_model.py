#load trained model


# Load the saved KNN model

def loaded_model():
    import os
    import joblib
    import numpy as np
    import ast
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct file paths for CHAR_TO_INT.txt and INT_TO_CHAR.txt
    char_to_int_path = os.path.join(script_dir, "CHAR_TO_INT.txt")
    int_to_char_path = os.path.join(script_dir, "INT_TO_CHAR.txt")
    
    # Load the KNN model
    knn_classifier = joblib.load(os.path.join(script_dir, 'knn_model.pkl'))
    
    # Load CHAR_TO_INT and INT_TO_CHAR dictionaries
    with open(char_to_int_path, "r") as char_to_int_file:
        CHAR_TO_INT = ast.literal_eval(char_to_int_file.read())  # Load dictionary from file

    with open(int_to_char_path, "r") as int_to_char_file:
        INT_TO_CHAR = ast.literal_eval(int_to_char_file.read())  # Load dictionary from file

    return knn_classifier, CHAR_TO_INT, INT_TO_CHAR


def sequence_to_numeric(sequence, CHAR_TO_INT):
    return [CHAR_TO_INT[char] for char in sequence]

#gap filling

def predict_sequence(user_seq, knn_model, CHAR_TO_INT, INT_TO_CHAR):
    new_seq = []
    for count, i in enumerate(range(len(user_seq) - 11 + 1)):
        kmer = user_seq[i:i + 11]
        new_seq = new_seq + [kmer]

# Fill the gaps in the sequence
def predict_sequence(user_seq, knn_model, CHAR_TO_INT, INT_TO_CHAR):
    # Split the user sequence into 11-length subsequences (k-mers)
    new_seq = []
    for i in range(len(user_seq) - 11 + 1):
        kmer = user_seq[i:i + 11]
        new_seq.append(kmer)

    if "-" not in user_seq:
        print("No gaps found in the initial sequence. Exiting function.")
        return user_seq
    
    print("Initial sequence:", user_seq)
    print("Initial k-mers:", new_seq)
    if "-" not in user_seq:
        print("No gaps found in the initial sequence. Exiting function.")
        return user_seq

    # Fill the gaps in the sequence
    while "-" in user_seq:
        # Find k-mers with  one gap
        keys_with_dash = [key for key in new_seq if key.count('-') == 1]

        if len(keys_with_dash) == 0:
            keys_with_dash = [key for key in new_seq if key.count('-') == 2]

        if len(keys_with_dash) == 0:
            break

        for k in keys_with_dash:
            if k[0] == "-" or k[10] == "-":
                if k in user_seq:
                    # Convert the k-mer to a numeric sequence for prediction
                    de_novo_sequence_numeric = sequence_to_numeric(k, CHAR_TO_INT)
                    X_de_novo = np.array([de_novo_sequence_numeric])

                    # Predict the value for the gap using the trained model
                    y_pred_de_novo = knn_model.predict(X_de_novo)
                    predicted_value = y_pred_de_novo[0]

                    print("Predicted value for k-mer", k, ":", predicted_value)

                    index = user_seq.index(k)
                    index1 = k.index("-")

                    if 0 <= (index + index1) < len(user_seq):
                        user_seq = user_seq[:(index + index1)] + predicted_value + user_seq[(index + index1) + 1:]

                        new_seq = [user_seq[i:i + 11] for i in range(len(user_seq) - 11 + 1)]
                        print("Updated sequence after filling:", user_seq)
                        print("Updated k-mers:", new_seq)
                    else:
                        print("Index out of bounds error for gap replacement.")
            else:
                print("K-mer has no gaps, skipping.")

    return user_seq

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






