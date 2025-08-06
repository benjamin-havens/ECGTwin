# Dataset Construction Tutorial

This tutorial tells the method to get following necessary training datasets:
```python
['paired_Mimic_vae_multi_nomic.pt''paired_Mimic_vae_multi_nomic_test.pt', 'paired_Mimic_vae_mix_nomic.pt', 'paired_Mimic_vae_mix_nomic_test.pt']
```

They can also be directly downloaded from [huggingface repo](https://huggingface.co/datasets/Laiyf/ECGTwin_Data/tree/main) if you do not want to buld them locally.

## STEP 0: Build from Original MIMIV-IV-ECG (Optional)

Download MIMIC-IV-ECG dataset from [PhysioNet](https://physionet.org/content/mimic-iv-ecg/1.0/) and prepare `patient.csv` table of MIMIC-IV-Clinical from [PhysioNet](https://physionet.org/content/mimiciv/3.1/).

Run following script from repo root directory to get latent dataset `mimic_vae.pt` and `mimic_vae_test.pt`:
```sh
python -m data.vae_encoding
```

This step may take a long time.

## STEP 1: Get Text Embeddings

Prepare latent dataset `mimic_vae.pt` and `mimic_vae_test.pt` by downloading from [huggingface](https://huggingface.co/datasets/Laiyf/ECGTwin_Data/tree/main) or via STEP 0

Change the `src` and `dst` variable in `store_embedding_nomic.py` and run the script to get `Mimic_vae_nomic.pt`, `Mimic_vae_multi_nomic.pt` from `Mimic_vae.pt` and get `Mimic_vae_nomic_test.pt`, `Mimic_vae_multi_nomic_test.pt` from `Mimic_vae_test.pt`:
```sh
python store_embedding_nomic.py
```

## STEP 2: Get Paried Dataset

Change the `test` variable in `dataset_construction.py` and run the script to get `paired_Mimic_vae_multi_nomic.pt` and `paired_Mimic_vae_multi_nomic_test.pt`:
```sh
python dataset_construction.py
```

Similarly change the `test` variable in `mix_dataset_construction.py` and run the script to get `paired_Mimic_vae_mix_nomic.pt` and `paired_Mimic_vae_mix_nomic_test.pt`
```sh
python mix_dataset_construction.py
```

Note that the `_mix` dataset is for the training of `_adaLN` models.

## STEP 3: Get PTBXL Version Dataset

Download PTBXL latent dataset `PTBXL_vae_test.pt` from [huggingface repo](https://huggingface.co/datasets/Laiyf/ECGTwin_Data/tree/main) and repeat STEP 1 to 2 with dataset name changing into PTBXL.