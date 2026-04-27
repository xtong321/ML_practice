"""
Identify tweet toxic

binary text classification problem
look at the datset - # of sampels, value_count [harmful vs not-harmful] samples
preprocess pipeline
training setup
evaluation on test data
"""

import pandas as pd
import numpy as np

data_df = pd.read_csv("tweets_flagged_v2.csv")
data_df = data_df.drop(columns=["Unnamed: 0"])

# to show head
data_df.head(5) 
data_df["harmful"].value_counts()

for _ in range(10):
    random_ind = np.random.randint(0, len(data_df))
    random_data = data_df.iloc[random_ind]
    print(random_data["tweet"], random_data["harmful"])

# preprocess
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-based-cased")

X = data_df["tweet"].values
y = data_df["harmful"].values

sequences = [sequence for sequence in X]
model_inputs = tokenizer(sequences, padding=True, return_tensors='tf')

#show model_inputs
#model_inputs
import tensorflow as tf
dataset = tf.data.Dataset.from_tensor_slices((model_inputs['input_idx'],y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8)

#training pipeline, 70% training, 20% validation, 10% testing
train = dataset.take(int(len(dataset)*0.7))
val = dataset.skip(int(len(dataset)*0.7)).take(int(len(dataset)*0.2))
test = dataset.take(int(len(dataset)*0.9)).take(int(len(dataset)*0.1))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense

model = Sequential(name="text-classifier")
model.add(Embedding(len(tokenizer.get_vocab)), 32)
model.add(Bidirectional(LSTM(32, activation='tanh')))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy", optimizer='Adam')

#check
model.summary()

with tf.device("/device:GPU:0"):
    history = model.fit(train, epochs=1, batch_size=16, validation_data=val)

# monitor the training loss: 1) loss drop too quick, 2) validation loss; 3) or test loss

#evalution
from tensorflow.keras.metrics import Precision, Recall, Accuracy
pre = Precision()
rec = Recall()
acc = Accuracy()

for batch in test.as_numpy_iterator():
    x_true, y_true = batch
    y_hat = model.predict(x_true)
    pre.update_state(y_true, y_hat)
    rec.update_state(y_true, y_hat)
    acc.update_state(y_true, y_hat)

print("precision", pre)
print("recall", rec)
print("accuracy", acc)