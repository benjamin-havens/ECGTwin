from collections import defaultdict
from tqdm import tqdm
import torch
from datetime import datetime

test = True
multi = True
if test:
    test = '_test'
else:
    test = ''
if multi:
    multi = '_multi'
else:
    multi = ''

# src = f"./PTBXL_vae{multi}_nomic_test.pt"
src = f"./Mimic_vae{multi}_nomic{test}.pt"
dataset = torch.load(src)
grouped_dataset = defaultdict(list)

# grouping
for entry in tqdm(dataset):
    id = entry['label']['subject_id']
    grouped_dataset[id].append(entry)
print(f"number of patient: {len(grouped_dataset)}")

# delete patient with only one ecg data
remove_list = []
for subject_id in grouped_dataset.keys():
    if len(grouped_dataset[subject_id]) < 2:
        remove_list.append(subject_id)
# print(f"number of patient with only ONE ecg data: {len(remove_list)}")

for subject_id in remove_list:
    del(grouped_dataset[subject_id])
print(f"number of patient include: {len(grouped_dataset)}")

# aggregate dataset in circle
paired_mimic_vae_w_embedding = []
for value in tqdm(grouped_dataset.values()):
    for i in range(len(value)):
        for j in range(i + 1, len(value)):
            if value[i]['label']['hr'] == value[j]['label']['hr'] and value[i]['label']['text'] == value[j]['label']['text']:
                continue
            # the previous one is stored at index 0
            ref_ecg_time = datetime.strptime(value[i]['label']['ecg_time'], '%Y-%m-%d %H:%M:%S')
            tar_ecg_time = datetime.strptime(value[j]['label']['ecg_time'], '%Y-%m-%d %H:%M:%S')
            if ref_ecg_time <= tar_ecg_time:
                data = (value[i], value[j])
            else:
                data = (value[j], value[i])
            paired_mimic_vae_w_embedding.append(data)
print(f"Number of paired ecg for training: {len(paired_mimic_vae_w_embedding)}")
torch.save(paired_mimic_vae_w_embedding, f"./paired_{src}")
