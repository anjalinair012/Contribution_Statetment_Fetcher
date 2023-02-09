# Contribution_Statetment_Fetcher
Repository for fetching contribution statements from research papers and generating summaries. Implemented BERT and TFIDF
- Tools: Python, Tensorflow, Transformers (Huggingface), Flask


The following can be attained on request
Datasets are found at : https://drive.google.com/drive/folders/1566xOabHqYtdpeDUqHl8DBryt7P-1gFz?usp=sharing
Model checkpoints and configuration files at : https://drive.google.com/drive/folders/1vFrqJUDPmnCYzBDCeoQsADh2dut39jYK?usp=sharing

## Contents of this repository

* requirements.txt --- virtual environemnt dependencies
* prep_data.py --- preprocessing and creating the dataset
* Model_traning.py --- Script to fine tune a bert model in classification of lines into contributing and non contributing. Can be opened on colab
* prediction_pipeline.py --- model pipeline when in testing
* app.py -- Flask deployment
* model_chcekpoints - automatically loaded model weights from our training
* static, templates - html and css files for Flask
* test-data-master, training-data-master, parsed_docs_testing - data files

## Instructions for use

Create virtual environemnt:
pip install -r requirements.txt


* run python prep_data.py to prepare your dataset
* Model_training.py uses the prepaperd datasets for training distillBert. Model was trained on colab. Model checkpoints and config can be fetched from the drive
* To run the summarizer : python app.py 
