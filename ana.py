#!/usr/bin/env python
# coding: utf-8


import streamlit as st
import pandas as pd
import pickle
import sklearn
import numpy as np
from sklearn.cluster import KMeans




st.write("""
# RS : ML_Book
## Book Recommendeed Prediction App

This app predicts the **Book** type
""")




def load_data():
	df=pd.DataFrame({'title':['A','B','C','D','E'],
			'author':['F','G','H','I','J']})
			
	return df
df = load_data()


#st.sidebar.header('User Input Parameters')

def user_input_features():
    title = st.selectbox('input:',df['title'].unique())
    author = st.selectbox('author:',df['author'].unique())
        
    
    data = {'title': title,'author': author}
    features = pd.DataFrame(data,index=[0])

    return features

df1 = user_input_features()

input_feature=[int(x) for x in df1]
final_feat=[np.array(input_feature)]

st.subheader('User Input parameters')
st.write(df)

popular=pickle.load(open('/Users/pc/LAB ML/popular.pkl', 'rb'))
prediction = popular.predict(final_feat)
prediction_proba = popular.predict_proba(df1)

st.subheader('Class labels and their corresponding index number')
st.write(pd.DataFrame(popular.classes_))

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)


# In[33]:





# In[ ]:





# In[ ]:





# In[ ]:




