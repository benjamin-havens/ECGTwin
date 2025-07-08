import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

HR_MEAN = 77.95
HR_STD = 20.37
AGE_MEAN = 64.25
AGE_STD = 17.13

def normalize_patient_info(key, value):
    if key == "age":
        value = (value - AGE_MEAN) / AGE_STD
    if key == "hr":
        value = (value - HR_MEAN) / HR_STD
    return value

def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


num = ['1st', '2nd', '3rd']

def prompt_propcess(text: str): 
    text = text.lower()
    prompt_text = ''
    c = 0
    s = ''
    for ch in text:
        if ch == '|':
            if c == 0:
                prompt_text += 'Most importantly, the 1st diagnosis is {' + s + '}.'
            else:
                prompt_text += 'As a supplementary condition, the ' + (num[c] if c <= 2 else str(c + 1) + 'th') + ' diagnosis is {' + s + '}.'
            c += 1
            s = ''
        else:
            s += ch
    if s != '':
        if c == 0:
            prompt_text += 'Most importantly, the 1st diagnosis is {' + s + '}.'
        else:
            prompt_text += 'As a supplementary condition, the ' + (num[c] if c <= 2 else str(c + 1) + 'th') + ' diagnosis is {' + s + '}.'
        c += 1
        s = ''

    return prompt_text 

def get_text_embedding(text: str, tokenizer, embedding_model, mix=False): 
    # mix=True: interpreting whole text as one embedding
    # mix=False: return split text embeddings
    if mix:
        text_input = prompt_propcess(text) 
    else:
        text_input = text.split('|')

    encoded_input = tokenizer(text_input, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {key: value.to(embedding_model.device) for key, value in encoded_input.items()}

    with torch.no_grad():
        model_output = embedding_model(**encoded_input)

    embedding = _mean_pooling(model_output, encoded_input['attention_mask'])
    embedding = F.layer_norm(embedding, normalized_shape=(embedding.shape[1],))
    embedding = embedding[:, :768]
    embedding = F.normalize(embedding, p=2, dim=1)

    return embedding

def process_pat_info(normalize=True, add_token=False, **kwargs):
    """
    input shape: Tensor([B])
    ouput shape: Tensor([B, L])
    """
    pat_info = []
    for key, value in zip(kwargs.keys(), kwargs.values()):
        if normalize:
            value = normalize_patient_info(key, value)
        value = value.unsqueeze(-1)
        value = value.to(dtype=torch.float32)
        pat_info.append(value)

    # Append one token to indicate end of pat info
    if add_token:
        pat_info.append(torch.ones_like(value))

    pat_info = torch.concat(pat_info, dim=1)
    return pat_info

def sex_transform(sex: str):
    return 0 if sex == 'F' else 1

class PairedECGDataset(Dataset):

    def __init__(self, path):
        self.data = torch.load(path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        ecg_0 = self.data[index][0]
        ecg_1 = self.data[index][1]
        
        return ecg_0, ecg_1


class ListDataset(Dataset):
    def __init__(self, path:str):
        self.data= torch.load(path)

    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self, idx):
        latent_dict = self.data[idx] 
        return latent_dict['data'], latent_dict['label']


def _pad_text_embed(embed_list):
    lengths = [e.shape[0] for e in embed_list]
    max_len = max(lengths)
    embed_dim = embed_list[0].shape[1]

    padded_embeds = []
    masks = []

    for embed in embed_list:
        pad_len = max_len - embed.shape[0]
        padded = torch.cat([embed, torch.zeros(pad_len, embed_dim)], dim=0)
        mask = torch.cat([torch.ones(embed.shape[0]), torch.zeros(pad_len)])
        padded_embeds.append(padded)
        masks.append(mask)

    return torch.stack(padded_embeds), torch.stack(masks)

def paired_ecg_collate_fn(batch, pad=True):
    ecg0_list, ecg1_list = zip(*batch)  # unzip
    ecg0 = {}
    ecg1 = {}

    ecg0['data'] = torch.stack([ecg['data'] for ecg in ecg0_list])
    ecg1['data'] = torch.stack([ecg['data'] for ecg in ecg1_list])

    ecg0['label'] = {}
    ecg1['label'] = {}
    for key, value in ecg0_list[0]['label'].items():
        if key in ['text_embed', 'text_embed_mask']:
            continue

        key_type = type(value)
        if key_type in (float, int, np.float64):
            ecg0['label'][key] = torch.tensor([ecg['label'][key] for ecg in ecg0_list])
            ecg1['label'][key] = torch.tensor([ecg['label'][key] for ecg in ecg1_list])
    
        elif key == 'sex':
            ecg0['label'][key] = torch.tensor([sex_transform(ecg['label'][key]) for ecg in ecg0_list])
            ecg1['label'][key] = torch.tensor([sex_transform(ecg['label'][key]) for ecg in ecg1_list])

        elif key == 'text_embed_whole':
            ecg0['label'][key] = torch.tensor([ecg['label'][key] for ecg in ecg0_list])
            ecg1['label'][key] = torch.tensor([ecg['label'][key] for ecg in ecg1_list])
        
        else:
            ecg0['label'][key] = [ecg['label'][key] for ecg in ecg0_list]
            ecg1['label'][key] = [ecg['label'][key] for ecg in ecg1_list]
        

    ecg0_embed_list = [ecg['label']['text_embed'] for ecg in ecg0_list]
    ecg1_embed_list = [ecg['label']['text_embed'] for ecg in ecg1_list]

    if pad:
        ecg0_embed, ecg0_mask = _pad_text_embed(ecg0_embed_list)
        ecg1_embed, ecg1_mask = _pad_text_embed(ecg1_embed_list)

        ecg0['label'].update({
            'text_embed': ecg0_embed,
            'text_embed_mask': ecg0_mask
        })
        ecg1['label'].update({
            'text_embed': ecg1_embed,
            'text_embed_mask': ecg1_mask
        })
    else:
        ecg0['label'].update({
            'text_embed': ecg0_embed_list,
        })
        ecg1['label'].update({
            'text_embed': ecg1_embed_list,
        })

    return (ecg0, ecg1)

def ecg_collate_fn(batch, pad=True):
    ecg_data = torch.stack([entry[0] for entry in batch])

    ecg_label = {}
    for key, value in batch[0][1].items():
        if key in ['text_embed', 'text_embed_mask']:
            continue

        key_type = type(value)
        if key_type in (float, int, np.float64):
            ecg_label[key] = torch.tensor([entry[1][key] for entry in batch])
    
        elif key == 'sex':
            ecg_label[key] = torch.tensor([sex_transform(entry[1][key]) for entry in batch])

        elif key == 'text_embed_whole':
            ecg_label[key] = torch.tensor([entry[1][key] for entry in batch])
        
        else:
            ecg_label[key] = [entry[1][key] for entry in batch]
        

    ecg_embed_list = [entry[1]['text_embed'] for entry in batch]

    if pad:
        ecg_embed, ecg_mask = _pad_text_embed(ecg_embed_list)

        ecg_label.update({
            'text_embed': ecg_embed,
            'text_embed_mask': ecg_mask
        })
    else:
        ecg_label.update({
            'text_embed': ecg_embed_list,
        })

    return (ecg_data, ecg_label)