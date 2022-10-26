# def preprocess(text):
#     new_text = []
#     for t in text.split(" "):
#         t = '@user' if t.startswith('@') and len(t) > 1 else t
#         t = 'http' if t.startswith('http') else t
#         new_text.append(t)
#     return " ".join(new_text)

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load tweet
df_tweet = pd.read_csv('df_completec.csv')
df = df_tweet[['text','text1']]
df['max_em'] = 0
#emotion = joy, optimism, anger, sadness
tasks = ['emotion','hate','offensive','irony']
for task in tasks:
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # download label mapping
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    emo_df = pd.DataFrame(0, index=df.index, columns=labels)

    for i in np.arange(df.shape[0]):
        text = df.text1[i]
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores) #ordina dal più basso al più alto
        emo_df.iloc[i] = scores
        if task == 'emotion':
            max = ranking[3] #prendo l'ultima posizione che corrisponde all indice max  
            df['max_em'].iloc[i] = labels[max]
        #if task == 'hate':

    df = pd.concat([df, emo_df], axis=1)

df.to_csv('df_task.csv', index = False)

count_max_em = df.groupby('max_em').max_em.count().reset_index(name="count")
count_max_em = count_max_em.set_index('max_em')

#PLOT
#leggi il df finale creato sopra
df = pd.read_csv('df_task.csv')

#pie chart - emotions 
plot = count_max_em.plot.pie(y='count', figsize=(5, 5))
plt.show()

#bar plot - other sentiment
df2 = df[['not-hate','hate','not-offensive','offensive','non_irony','irony']]
df2_sum = df2.sum().to_frame(name="sum")
sns.barplot(data=df2_sum, x=df2_sum.index, y='sum')
plt.show()