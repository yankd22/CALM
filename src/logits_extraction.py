import os
import math
import numpy as np

def calculate_entropy(logits):
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)
    entropy = -np.sum(probs * np.log(probs + 1e-9)) 
    return entropy

def credible_entropy_samples(dir_path, data_name, ratio, standard):
    results = {}
    
    file_path = os.path.join(dir_path, f"{data_name}_embedding.txt")
    
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return

    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    samples = []
    logits_list = []
    labels = []

    for i in range(0, len(lines), 2):
        sample_line = lines[i].strip()
        logits_line = lines[i+1].strip()
        
        sample_id = int(sample_line.split()[1].strip(','))
        label = int(sample_line.split(":")[1])
        logits = list(map(float, logits_line.split(":")[1].strip(' []').split(',')))
        
        samples.append(sample_id)
        logits_list.append(logits)
        labels.append(label)
    
    logits_array = np.array(logits_list)
   
    if standard:
        entropies = [calculate_entropy(logits) for logits in logits_list]
        sorted_indices = np.argsort(entropies)
        top_indices = sorted_indices[:int(len(entropies) * ratio)]
    else:
        entropies = [calculate_entropy(logits) for logits in logits_list]
        entropies = np.array(entropies)

        max_logit_labels = np.argmax(logits_array, axis=1)
        selected_indices = []

        for unique_label in np.unique(max_logit_labels):
            label_indices = np.where(max_logit_labels == unique_label)[0]

            label_entropies = entropies[label_indices]
            label_samples = label_indices

            sorted_indices = label_samples[np.argsort(label_entropies)]

            selected_size = int(len(sorted_indices) * ratio)

            selected_indices.extend(sorted_indices[:selected_size])

        top_indices = selected_indices

    for idx in top_indices:
        max_logit_label = np.argmax(logits_array[idx])
        results[samples[idx]] = max_logit_label
    
    return results


