#!/usr/bin/env python
# coding: utf-8

# ## In Memory implementation

# In[14]:


import pandas as pd
import numpy as np


# In[15]:


import requests 

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)


# In[16]:


documents[1]


# In[17]:


df = pd.DataFrame(documents, columns=['course','section','question','text'])


# In[18]:


df.head()


# In[27]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[28]:


cv = CountVectorizer(max_df=5)


# In[29]:


cv.fit(df.text)


# In[30]:


cv.get_feature_names_out().shape


# In[31]:


doc_ex = [
    "Course starts on 15th Jan 2024",
    "Prerequisites listed on GitHub",
    "Submit homeworks after start date",
    "Registration not required for participation",
    "Setup Google Cloud and Python before course"
]


# In[32]:


CV = TfidfVectorizer(stop_words='english')
CV.fit(doc_ex)


# In[33]:


CV.get_feature_names_out()


# In[34]:


X = CV.transform(doc_ex)


# In[35]:


pd.DataFrame(X.todense(),columns=CV.get_feature_names_out()).T


# In[36]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

CV = TfidfVectorizer(stop_words='english', min_df=5)
CV.fit(df.text)

X = CV.transform(df.text)

names = CV.get_feature_names_out()

df_doc = pd.DataFrame(X.toarray(),columns=names).T
df_doc.round(2)


# In[37]:


X


# In[38]:


query = "Do I need to know python to sign up for the January course?"

q = CV.transform([query])
q.toarray()


# In[39]:


from sklearn.metrics.pairwise import cosine_similarity


# In[40]:


#X.dot(q.T).todense()..->gives same result as cosine_similarity(X,q)----> cosine_similarity is just a normalized dot product


# In[41]:


#falatten to change from 2 dim to just one dim numpy array
score = cosine_similarity(X,q).flatten()


# In[42]:


#this gives the indices of the documents sorted, in increasing order
np.argsort(score)[-5:]


# In[43]:


df.iloc[27].text


# In[44]:


fields = ['section', 'question','text']


# In[45]:


matrices = {}
vecterizor = {}

for f in fields:
    cv = TfidfVectorizer(stop_words='english', min_df=5)
    X = cv.fit_transform(df[f])
    matrices[f] = X
    vecterizor[f] = cv


# In[46]:


matrices


# In[47]:


vecterizor


# In[48]:


df


# In[49]:


n = len(df)


# In[50]:


score = np.zeros(n)
boosts = {
    'question': 3,
    'test': 0.5
}

query = "Do I need to know python to sign up for the January course?"

for f in fields:
    q = vecterizor[f].transform([query])
    x = matrices[f]

    boost = boosts.get(f, 1.0)
    f_Score = cosine_similarity(x,q).flatten()

    score =  score + boost * f_Score


# In[51]:


filters =  {
    'course' : 'data-engineering-zoomcamp'
}


# In[52]:


for field, value in filters.items():
    mask = (df[field] == value).astype(int).values
    score = score * mask


# In[53]:


idx = np.argsort(-score)[:5]


# In[54]:


df.iloc[idx]


# In[1]:


class TextSearch:

    def __init__(self, text_fields):
        self.text_fields = text_fields
        self.matrices = {}
        self.vectorizers = {}

    def fit(self, records, vectorizer_params={}):
        self.df = pd.DataFrame(records)

        for f in self.text_fields:
            cv = TfidfVectorizer(**vectorizer_params)
            X = cv.fit_transform(self.df[f])
            self.matrices[f] = X
            self.vectorizers[f] = cv

    def search(self, query, n_results=10, boost={}, filters={}):
        score = np.zeros(len(self.df))

        for f in self.text_fields:
            b = boost.get(f, 1.0)
            q = self.vectorizers[f].transform([query])
            s = cosine_similarity(self.matrices[f], q).flatten()
            score = score + b * s

        for field, value in filters.items():
            mask = (self.df[field] == value).values
            score = score * mask

        idx = np.argsort(-score)[:n_results]
        results = self.df.iloc[idx]
        return results.to_dict(orient='records')


# In[ ]:


index = TextSearch(
    text_fields= ['section', 'question','text']
)
index.fit(documents)

index.search(
    query = "Do I need to know python to sign up for the January course?",
    n_results= 5,
    boost= {'question': 3.0},
    filters = {'course' : 'data-engineering-zoomcamp'}
)


# #### Embeddings  --- bag of words to --SVD-->embeddings
# - all the above was with bag of words, no order and exact match no synomous matches, and embeddings are dense vectors

# In[57]:


from sklearn.decomposition import TruncatedSVD

x = matrices['text']
cv = vecterizor['text']

svd = TruncatedSVD(n_components=16)
x_emb = svd.fit_transform(x)
x_emb.shape


# In[58]:


x_emb[0]


# In[79]:


query = "Do I need to know python to sign up for the January course?"
query_1 = "where to learn about Docker?"

Q = cv.transform([query_1])
Q_emb = svd.transform(Q)


# In[80]:


Q_emb[0]


# In[81]:


np.dot(Q_emb[0], x_emb[0])


# In[82]:


score = cosine_similarity(x_emb, Q_emb).flatten()
idx = np.argsort(-score)[:5]
list(df.loc[idx].text)
df.loc[idx]


# In[64]:


from sklearn.decomposition import NMF

nmf = NMF(n_components=16)
x_emb = nmf.fit_transform(x)
x_emb[0]


# In[77]:


query = "where to learn about Docker?"
Q = cv.transform([query])
Q_emb = nmf.transform(Q)
Q_emb[0]


# In[83]:


score = cosine_similarity(x_emb, Q_emb).flatten()
idx = np.argsort(-score)[:10]
list(df.loc[idx].text)
df.loc[idx]


# In[85]:


import torch

from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()


# In[86]:


texts = [
    "Yes, we will keep all the materials after the course finishes.",
    "You can follow the course at your own pace after it finishes"
]
encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')


# In[87]:


encoded_input


# In[90]:


with torch.no_grad():
    outputs = model(**encoded_input)
    hidden_states = outputs.last_hidden_state

hidden_states.shape


# In[89]:


sentence_embeddings = hidden_states.mean(dim=1)
sentence_embeddings.shape


# In[91]:


sentence_embeddings


# In[ ]:




