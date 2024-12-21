import os
import math
import numpy as np

def calculate_entropy(logits):
    """计算给定logits的熵"""
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)  # softmax计算概率
    entropy = -np.sum(probs * np.log(probs + 1e-9))  # 避免log(0)
    return entropy

def credible_entropy_samples(dir_path, data_name, ratio, standard):
    results = {}
    
    file_path = os.path.join(dir_path, f"{data_name}_embedding.txt")
    
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return
    
    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    samples = []
    logits_list = []
    labels = []
    
    # 解析文件内容，提取logits和label
    for i in range(0, len(lines), 2):  # 假设每个样本由两行组成
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
        # 按熵排序并提取最大的熵值样本
        entropies = [calculate_entropy(logits) for logits in logits_list]
        sorted_indices = np.argsort(entropies)  # 从小到大排序
        top_indices = sorted_indices[:int(len(entropies) * ratio)]
    else:
        # 计算每个样本的熵
        entropies = [calculate_entropy(logits) for logits in logits_list]
        entropies = np.array(entropies)

        # 按 logits 的最大值标签进行分类
        max_logit_labels = np.argmax(logits_array, axis=1)
        selected_indices = []

        # 按标签进行采样
        for unique_label in np.unique(max_logit_labels):
            # 找到当前标签的所有样本索引
            label_indices = np.where(max_logit_labels == unique_label)[0]

            # 提取当前标签下的熵和索引
            label_entropies = entropies[label_indices]
            label_samples = label_indices

            # 按熵从小到大排序
            sorted_indices = label_samples[np.argsort(label_entropies)]

            # 计算需要选择的样本数量
            selected_size = int(len(sorted_indices) * ratio)

            # 提取熵最小的样本索引
            selected_indices.extend(sorted_indices[:selected_size])

        top_indices = selected_indices
        
    # 记录结果
    for idx in top_indices:
        max_logit_label = np.argmax(logits_array[idx])
        results[samples[idx]] = max_logit_label
    
    return results

# import os
# import numpy as np
# import matplotlib.pyplot as plt

# def calculate_entropy(logits):
#     """计算给定logits的熵"""
#     exp_logits = np.exp(logits)
#     probs = exp_logits / np.sum(exp_logits)  # softmax计算概率
#     entropy = -np.sum(probs * np.log(probs + 1e-9))  # 避免log(0)
#     return entropy

# def credible_entropy_samples(dir_path, dataset, ratio, standard):
#     accuracies = []
    
#     file_path = os.path.join(dir_path, f"{dataset}_embedding.txt")
    
#     if not os.path.exists(file_path):
#         print(f"File {file_path} does not exist.")
#         return 0
    
#     # 读取文件内容
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
    
#     samples = []
#     logits_list = []
#     labels = []
    
#     # 解析文件内容，提取logits和label
#     for i in range(0, len(lines), 2):  # 假设每个样本由两行组成
#         sample_line = lines[i].strip()
#         logits_line = lines[i+1].strip()
        
#         sample_id = int(sample_line.split()[1].strip(','))
#         label = int(sample_line.split(":")[1])
#         logits = list(map(float, logits_line.split(":")[1].strip(' []').split(',')))
        
#         samples.append(sample_id)
#         logits_list.append(logits)
#         labels.append(label)
    
#     logits_array = np.array(logits_list)
    
#     if standard:
#         # 按熵排序并提取最大的熵值样本
#         entropies = [calculate_entropy(logits) for logits in logits_list]
#         sorted_indices = np.argsort(entropies)  # 从小到大排序
#         top_indices = sorted_indices[:int(len(entropies) * ratio)]
#     else:
#         # 计算每个样本的熵
#         entropies = [calculate_entropy(logits) for logits in logits_list]
#         entropies = np.array(entropies)

#         # 按 logits 的最大值标签进行分类
#         max_logit_labels = np.argmax(logits_array, axis=1)
#         selected_indices = []

#         # 按标签进行采样
#         for unique_label in np.unique(max_logit_labels):
#             # 找到当前标签的所有样本索引
#             label_indices = np.where(max_logit_labels == unique_label)[0]

#             # 提取当前标签下的熵和索引
#             label_entropies = entropies[label_indices]
#             label_samples = label_indices

#             # 按熵从小到大排序
#             sorted_indices = label_samples[np.argsort(label_entropies)]

#             # 计算需要选择的样本数量
#             selected_size = int(len(sorted_indices) * ratio)

#             # 提取熵最小的样本索引
#             selected_indices.extend(sorted_indices[:selected_size])

#         top_indices = selected_indices
    
#     # 计算准确率
#     correct_count = sum([1 for idx in top_indices if np.argmax(logits_array[idx]) == labels[idx]])
#     accuracy = correct_count / len(top_indices)
    
#     return accuracy

# def plot_accuracies(dir_path, datasets, save_dir):
#     ratios = np.linspace(0.1, 1.0, 10)
    
#     plt.figure(figsize=(10, 6))
    
#     # 绘制 standard=True 的图像
#     for dataset in datasets:
#         standard_accuracies = []
        
#         for ratio in ratios:
#             standard_acc = credible_entropy_samples(dir_path, dataset, ratio, standard=True)
#             standard_accuracies.append(standard_acc)
        
#         plt.plot(ratios, standard_accuracies, marker='o', label=f'{dataset}')
    
#     plt.xlabel('Ratio')
#     plt.ylabel('Accuracy')
#     plt.title('Accuracy vs Ratio (Standard=True) for Different Datasets')
#     plt.grid(True)
#     plt.legend()
#     standard_img_path = os.path.join(save_dir, 'accuracy_standard_true.png')
#     plt.savefig(standard_img_path)  # 保存图像
#     plt.close()  # 关闭当前图像窗口
    
#     # 绘制 standard=False 的图像
#     plt.figure(figsize=(10, 6))
    
#     for dataset in datasets:
#         non_standard_accuracies = []
        
#         for ratio in ratios:
#             non_standard_acc = credible_entropy_samples(dir_path, dataset, ratio, standard=False)
#             non_standard_accuracies.append(non_standard_acc)
        
#         plt.plot(ratios, non_standard_accuracies, marker='o', label=f'{dataset}')
    
#     plt.xlabel('Ratio')
#     plt.ylabel('Accuracy')
#     plt.title('Accuracy vs Ratio (Standard=False) for Different Datasets')
#     plt.grid(True)
#     plt.legend()
#     non_standard_img_path = os.path.join(save_dir, 'accuracy_standard_false.png')
#     plt.savefig(non_standard_img_path)  # 保存图像
#     plt.close()  # 关闭当前图像窗口
    
#     print(f"Images saved to {save_dir}")

# result = plot_accuracies("/home/ykd/project/model_merging/AdaMerging/embedding/", ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD'], 
# "/home/ykd/project/model_merging/AdaMerging/logs/")

