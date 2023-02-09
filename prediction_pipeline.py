import tensorflow
import pandas as pd
from transformers import AutoModel, AutoTokenizer, TFAutoModelForSequenceClassification
import numpy as np
from tensorflow.keras.optimizers import Adam
import pdb 
import re
from sklearn.feature_extraction.text import TfidfVectorizer
#from summarizer import Summarizer


class NLP_Pipeline:

	def __init__(self):
		self.model = None
		self.tokenizer = None
		self.data = pd.DataFrame(columns=["true_text", "text", "line_num", "label", "score"])

	# Parse pdf to text
	def pdf_parser(self, filename, context):
		if not context:
			context = open(filename,"r").read()
		context = re.split(r'[\.\r\n]+', context)  # split into lines
		for line_num, line in enumerate(context):
			self.data = self.data.append({"true_text" : line.strip(), "text": line.strip().lower(), "line_num": line_num + 1, "label": 1, "score": 0}, ignore_index=True)
		self.data = self.data.dropna(subset=["text"])
		self.data.reset_index(inplace=True)
		return

	# clean data and prep input
	def data_preprocess(self):
		to_drop = []
		for index in range(len(self.data)):
			text = self.data["text"][index]
			if len(text.split(" ")) <= 5:
				to_drop.append(index)
			else:
				text = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}', '', text)  # remove all text in brackets
				text = re.sub(r'[^\w\s]', '', text)  # remove all special characters
				self.data["text"][index] = re.sub(r'http\S+', '', text).strip()  # remove links
		self.data = self.data.drop(to_drop, axis=0).reset_index()
		return

	def tfidf_cleaner(self, summary_len=10):
		# Create a TfidfVectorizer object
		vectorizer = TfidfVectorizer()
		score = vectorizer.fit_transform(self.data["text"]).toarray().sum(axis=1)  #get tfidf scores for all lines
		for s in range(len(score)):
			self.data["score"][s] = score[s]
		pos_dataset = self.data[self.data["label"]==1].copy()
		pos_dataset["score"] = pd.to_numeric(pos_dataset["score"])
		average = pos_dataset['score'].mean()
		return pos_dataset.nlargest(summary_len, 'score')  # return the top summary_len contributing lines

	# load model
	def model_loader(self):
		self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
		self.model = TFAutoModelForSequenceClassification.from_pretrained('./model_checkpoint/')

	# get contributing statements:
	def get_sentences(self):
		max_len = 256
		tokenized_data = self.tokenizer.batch_encode_plus(list(self.data["text"]),padding="max_length", truncation=True, max_length = max_len, return_tensors="tf")
		logits = self.model.predict(dict(tokenized_data))  # softmax returns logits
		self.data["label"] = np.argmax(logits["logits"], axis=1).flatten()
		return

	# summarizer
	def summarizer(self, summary_len=10):
		subset_pos = self.tfidf_cleaner(summary_len=summary_len)
		summary=subset_pos["true_text"].apply(lambda x: x).str.cat(sep='.')  #concatenate all postive sentences
		return summary


	def run(self, filename="training-data-master/natural_language_inference/6/1812.10464v2-Grobid-out.txt", context=None):
		# Function puts together the pipeline

		if context:
			self.pdf_parser(None, context)
		else:
			self.pdf_parser(filename, None)
		self.data_preprocess()
		self.model_loader()
		self.get_sentences()
		summary = self.summarizer()
		return summary


if __name__ == "__main__":
	obj = NLP_Pipeline()
	obj.run()