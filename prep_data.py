import os
import glob
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import pdb
import re

def create_dataset():
    data = pd.DataFrame(columns=["text", "label", "file_name", "line_num", "score"])
    print("-------------------creating training data--------------")
    for taskDir in subdirList:
        if not os.path.isdir(os.path.join(root_dir, taskDir)):
            continue
        fileDirList = os.listdir(os.path.join(root_dir, taskDir))

        # Use a for loop to iterate through the files in the directory
        for fileDir in fileDirList:
            dir_path = os.path.join(root_dir, taskDir, fileDir)

            # Get all lines from the Stanza-out file
            stanza_f = open(glob.glob(os.path.join(dir_path, "*-Stanza-out.txt"))[0], "r")
            stanza_sentences = stanza_f.readlines()

            # Get indexes of all contribution sentences
            sentence_f = open(os.path.join(dir_path, "sentences.txt"), "r")
            positive_sentences = [int(num) for num in sentence_f.read().split()]

            # Get indexes of all non-contribution sentences
            negative_f = open(os.path.join(dir_path, "negative_sentences.txt"), "r")
            negative_sentences = [int(num) for num in negative_f.read().split()]

            for line_num, line in enumerate(stanza_sentences):
                if line_num + 1 in positive_sentences:
                    data = data.append({"text": line.strip().lower(), "label": 1, "file_name": str(dir_path), "line_num": line_num+1, "score": 0}, ignore_index=True)
                elif line_num + 1 in negative_sentences:
                    data = data.append({"text": line.strip().lower(), "label": 0, "file_name": str(dir_path), "line_num": line_num+1, "score": 0}, ignore_index=True)
            negative_f.close()
            sentence_f.close()
            stanza_f.close()
    print("-------------data being preprocessed----------")
    data = preprocess(data)
    print('---------find dataset in data.csv-------------')
    data.to_csv('{}/data.csv'.format(root_dir))


def preprocess(data):
    to_drop = []
    for index in range(len(data)):
        text = data["text"][index]
        if len(text.split(" ")) <= 5:
            to_drop.append(index)
        else:
            text = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}', '', text)  # remove all text in brackets
            text = re.sub(r'[^\w\s]', '', text)  # remove all special characters
            data["text"][index] = re.sub(r'http\S+', '', text).strip()  # remove links
            #data["text"][index] = re.sub(r'\d+', 'num', text) # replace numbers with "num"
    data = data.drop(to_drop, axis=0)
    # Upsample contributing sentences to increase the dataset size
    # selected_rows = data.loc[data["label"] == 1]
    # data = data.append(selected_rows)
    return data


def get_negative_pairs():
    # Creates negative_sentences.txt with all non-contributing sentences from *-Stanza-out.txt
    print("--------------Creating file for non-contributing statements------------------")
    for taskDir in subdirList:
        if not os.path.isdir(os.path.join(root_dir, taskDir)):
            continue
        fileDirList = os.listdir(os.path.join(root_dir,taskDir))

        # Use a for loop to iterate through the files in the directory
        for fileDir in fileDirList:
            dir_path = os.path.join(root_dir, taskDir, fileDir)
            stanza_f = open(glob.glob(os.path.join(dir_path, "*-Stanza-out.txt"))[0], "r")
            sentence_f = open(os.path.join(dir_path, "sentences.txt"), "r")
            positive_sentences = [int(num) for num in sentence_f.read().split()]
            negative_f = open(os.path.join(dir_path, "negative_sentences.txt"), "w")
            for line_num, line in enumerate(stanza_f):

                # consider line to be a negative/non-contribution line is not in sentences.txt
                # if line has less than 5 words, it's probably a heading
                if line_num+1 not in positive_sentences and len(line.split()) > 5:
                    negative_f.write(str(line_num+1) + "\n")

            # close all files
            stanza_f.close()
            sentence_f.close()
            negative_f.close()


def count_sentences(sentence_file):
    positive_pair_count = 0
    for taskDir in subdirList:
        if not os.path.isdir(os.path.join(root_dir, taskDir)):
            continue
        fileDirList = os.listdir(os.path.join(root_dir,taskDir))
        # Use a for loop to iterate through the files in the directory
        for fileDir in fileDirList:
            # Open only sentences.txt file to count number of sentences
            f = open(os.path.join(root_dir, taskDir, fileDir, sentence_file), "r")
            # Take count of all unique line numbers in file
            positive_pair_count += len(set(f.readlines()))
            f.close()
    print("Total number of sentences in {}: {}".format(sentence_file, positive_pair_count))


def balance_sampler_random():
    # Store a part of the negative_sentences in another file "filtNegative_sentences.txt"
    data = load_data()
    negative_sentences = data[data["label"] == 0].index
    indexes = random.sample(list(negative_sentences), len(negative_sentences) - len(data[data["label"] == 1]))
    pdb.set_trace()
    data.drop(indexes, axis = 0, inplace=True)
    pdb.set_trace()
    data.reset_index().to_csv(os.path.join(root_dir,"filtered_data_rand.csv"), columns = ["text","label","file_name","line_num"])
    print("done")

def load_data():
    return pd.read_csv("{}/data.csv".format(root_dir), header=0, index_col = [0])


def tfidf_score():
    data = load_data()
    data = data.dropna(subset=["text"]).reset_index()
    # Apply the tf-idf algorithm to the DataFrame
    filenames = data["file_name"].unique()
    for filename in filenames:
            # Create a TfidfVectorizer object
            vectorizer = TfidfVectorizer()
            file_data = data[data["file_name"] == filename]
            tfidf_matrix = vectorizer.fit_transform(file_data["text"])
            data.loc[data["file_name"] == filename, "score"] = np.array(tfidf_matrix.sum(axis=1)).flatten()
    for index in range(len(data)):
        try:
            data.loc[index,"score"] = data["score"][index]/len(data["text"][index].split())
        except:
            pdb.set_trace()
    data.to_csv('{}/data.csv'.format(root_dir))
    return


def analyse_data(sentence_scores):
    data = load_data()
    positive_rows = data.loc[data['label'] == 1].index
    positive_scores = sentence_scores[positive_rows]
    negative_rows = data.loc[data['label'] == 0].index
    negative_scores = sentence_scores[negative_rows]
    pos_quantile = np.quantile(positive_scores,[0,0.25,0.75])
    neg_quantile = np.quantile(negative_scores, [0, 0.25, 0.75])


def balance_sampler_tfidf():
    # Select non-contributing samples based on their tf-idf scores
    data = load_data().reset_index()
    # Fetch all folder-names
    file_names = data["file_name"].unique()
    file_index=0
    filtered_data = pd.DataFrame()
    while file_index<len(file_names):
        subset_neg = data.loc[(data["file_name"] == file_names[file_index]) & (data["label"] == 0)]
        subset_pos = data.loc[(data["file_name"] == file_names[file_index]) & (data["label"] == 1)]
        # Sort non-contributing sentences based on their score and select top len(subset_pos) from top
        subset_neg = subset_neg.sort_values(by="score", ascending=False)[:] #min(len(subset_pos), len(subset_neg))
        filtered_data = pd.concat([filtered_data,subset_pos], axis = 0)
        filtered_data = pd.concat([filtered_data,subset_neg], axis = 0)
        file_index+=1
    filtered_data = filtered_data.drop(["level_0"], axis = 1)
    filtered_data = filtered_data.reset_index()
    print(filtered_data.head())
    filtered_data.to_csv(os.path.join(root_dir,"filtered_data.csv"), columns = ["text","label","file_name","line_num"])


root_dir = 'training-data-master'
subdirList = os.listdir(root_dir)
subdirList.remove("README.md")

# get_negative_pairs()
create_dataset()
balance_sampler_random()
tfidf_score()
balance_sampler_tfidf()
analyse_data()