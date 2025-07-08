from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn.functional as F
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from store_embedding_nomic import prompt_propcess, mean_pooling

test = True
if test:
    test = '_test'
else:
    test = ''

dataset = torch.load(f"./Mimic_vae_multi_nomic{test}.pt")
# dataset = torch.load(f"./PTBXL_vae_multi_nomic{test}.pt")
grouped_dataset = defaultdict(list)

device = "cuda:7"
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, safe_serialization=True)
model.to(device)
model.eval()

print("get whole report embedding")
for data in tqdm(dataset):
    text = data['label']['text']
    prompt_text = prompt_propcess(text)

    encoded_input = tokenizer(prompt_text, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

    with torch.no_grad():
        model_output = model(**encoded_input)

    embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    embedding = F.layer_norm(embedding, normalized_shape=(embedding.shape[1],))
    embedding = embedding[:, :768]
    embedding = F.normalize(embedding, p=2, dim=1)

    data['label']['text_embed_whole'] = embedding[0].tolist()

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
torch.save(paired_mimic_vae_w_embedding, f"./paired_Mimic_vae_mix_nomic{test}.pt")
# torch.save(paired_mimic_vae_w_embedding, f"./paired_PTBXL_vae_mix_nomic{test}.pt")
