import pandas as pd
import numpy as np

#import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

#import matplotlib.pyplot as plt
import streamlit as st

#from nltk.tokenize import sent_tokenize, word_tokenize
#from nltk.corpus import stopwords,wordnet
#from nltk.stem import WordNetLemmatizer



@st.cache(allow_output_mutation=True)
def load_data():
	df = pd.read_csv("https://github.com/tunthunchawin/pizza/blob/main/pizza_data.csv")
	return df

st.set_page_config(layout='wide')

df = load_data()

df.columns = df.columns.str.lower()

df['ingredients']= df['ingredients'].str.replace(' and','')

df['ingredients_new'] = df.ingredients.str.split(',')


df2=df['ingredients'].str.get_dummies(',')
df3=pd.get_dummies(df.type)

df4=pd.concat([df,df2,df3],axis=1)

indices = pd.Series(df.index, index=df['fullname']).drop_duplicates()

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['ingredients'])


cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

st.title("***ARE YOU BORED CHOOSING PIZZA?***")

st.write("This wep-app is created to help you find the least similarty of ingredients on pizza in order to make you more enjoyable with your meals."" "
	"To acquire the result, we apply some of the mathametic formula which is so-called Cosine similarity to the model.")

st.latex(r'''similarity = \sum_{i=1}^{n} A_{i}B_{i}(\frac{1}{\sum_{i=1}^{n}\sqrt{A_{i}^2}{\sum_{i=1}^{n}\sqrt{B_{i}^2}}})
''')

st.write("Anyway, don't worry about it. Just go scroll down!!!")

st.title('**MENU**')
st.write('***PIZZA***')

st.sidebar.image("pizza logo.jpg", use_column_width=True,caption="SLICES PIZZARIA")