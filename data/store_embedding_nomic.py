import torch 
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import tqdm 
import re

num = ['1st', '2nd', '3rd']

def prompt_propcess(text: str): 
    text = text.lower()
    prompt_text = ''
    c = 0
    s = ''
    for ch in text:
        if ch == '|':
            # prompt_text += 'The ' + (num[c] if c <= 2 else str(c) + 'th') + ' diagnosis is {' + s + '}. '
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

    # print(prompt_text)
    return prompt_text 

PRESERVE_UPPER = {
    'ECG', 'NDT', 'NST', 'DIG', 'LNGQT', 'NORM', 'IMI', 'ASMI', 'LVH', 'LAFB', 'ISC', 'IRBBB', '1AVB', 'IVCD', 
    'ISCAL', 'CRBBB', 'CLBBB', 'ILMI', 'LAO/LAE', 'AMI', 'ALMI', 'ISCIN', 'HYP', 'CD', 'INJAS', 'VPC', 'LMI', 
    'ISCIL', 'ISCI', 'LPFB', 'LAFB/LPFB', 'ISCAS', 'ISCA', 'INJAL', 'ISCLA', 'RVH', 'ANEUR', 'RAO/RAE', 'EL', 
    'WPW', 'ILBBB', 'IPLMI', 'ISCAN', 'IPMI', 'SEHYP', 'INJIN', 'INJLA', 'PMI', '3AVB', 'INJIL', '2AVB', 'ABQRS', 
    'PVC', 'STD', 'VCLVH', 'QWAVE', 'LOWT', 'NT', 'PAC', 'LPR', 'INVT', 'LVOLT', 'HVOLT', 'TAB', 'PRC(S)', 'SR', 
    'AFIB', 'STACH', 'SARRH', 'SBRAD', 'PACE', 'SVARR', 'BIGU', 'AFLT', 'SVTAC', 'PSVT', 'TRIGU', '2AVB1', '2AVB2', 
    'ABI', 'ALS', 'APB', 'AQW', 'ARS', 'AVB', 'CCR', 'CR', 'ERV', 'FQRS', 'IDC', 'IVB', 'JEB', 'JPT', 'LBBB', 'LBBBB',
    'LFBBB', 'LVQRSAL', 'LVQRSCL', 'LVQRSLL', 'MI', 'MIBW', 'MIFW', 'MILW', 'MISW', 'PRIE', 'PWC', 'QTIE', 'RAH', 
    'RBBB', 'STDD', 'STE', 'STTC', 'STTU', 'TWC', 'TWO', 'UW', 'VB', 'VEB', 'VFW', 'VPB', 'VPE', 'VET', 'WAVN', 'SB',
    'ST', 'AF', 'SA', 'SVT', 'AT', 'AVNRT', 'AVRT', 'SAAWR'
}

def format_report(report: str) -> str:
    if not report.strip():
        return ""

    report = report.strip().rstrip('.')
    tokens = re.findall(r"(\w+|\s+|[^\w\s])", report)

    processed = []
    for token in tokens:
        if token.isalpha():  
            if token.upper() in PRESERVE_UPPER:
                processed.append(token.upper())  
            else:
                processed.append(token.lower())  
        else:  
            processed.append(token)

    formatted = ''.join(processed)
    formatted = ' '.join(formatted.split())  

    formatted = formatted.replace('|', ', ')
    formatted = formatted + '.'
    return formatted


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def store_whole_embedding_to_dataset(src: str, dst: str, tokenizer, model, device): 
    assert src != dst 
    src_dataset = torch.load(src)  

    new_dataset_with_embedding = []
    for value in tqdm.tqdm(src_dataset): 
        text = value['label']['text'] 

        prompt_text = prompt_propcess(text) 
        # prompt_text = format_report(text)
        # print(prompt_text)
        encoded_input = tokenizer(prompt_text, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

        with torch.no_grad():
            model_output = model(**encoded_input)

        embedding = mean_pooling(model_output, encoded_input['attention_mask'])
        embedding = F.layer_norm(embedding, normalized_shape=(embedding.shape[1],))
        embedding = embedding[:, :768]
        embedding = F.normalize(embedding, p=2, dim=1)

        value['label']['text_embed'] = embedding[0].tolist()
        new_dataset_with_embedding.append(value)

    torch.save(new_dataset_with_embedding, dst) 


def store_split_embedding_to_dataset(src: str, dst: str, tokenizer, model, device): 
    assert src != dst 
    src_dataset = torch.load(src)  

    new_dataset_with_embedding = []
    for value in tqdm.tqdm(src_dataset): 
        all_report = value['label']['text'] 
        all_report = all_report.split('|')
        encoded_input = tokenizer(all_report, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

        with torch.no_grad():
            model_output = model(**encoded_input)

        embedding = mean_pooling(model_output, encoded_input['attention_mask'])
        embedding = F.layer_norm(embedding, normalized_shape=(embedding.shape[1],))
        embedding = embedding[:, :768]
        embedding = F.normalize(embedding, p=2, dim=1)

        value['label']['text_embed'] = embedding.to("cpu")
        new_dataset_with_embedding.append(value)

    torch.save(new_dataset_with_embedding, dst) 

if __name__ == '__main__': 
    src = './Mimic_vae.pt'
    dst = './Mimic_vae_nomic.pt'
    # src = './PTBXL_vae_test.pt'
    # dst = './PTBXL_vae_nomic_test.pt'
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, safe_serialization=True)
    model.to(device)
    model.eval()
    if 'multi' in dst:
        store_split_embedding_to_dataset(src, dst, tokenizer, model, device)
    else:
        store_whole_embedding_to_dataset(src, dst, tokenizer, model, device)