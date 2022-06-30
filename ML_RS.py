#!/usr/bin/env python
# coding: utf-8

# <h1><center>MSDE4 / Nabil EL Karafi</center></h1>
# <h1><center>Machine Learning</center></h1>
# <h1><center>RS</center></h1>

# # 1- Présentation de la problématique/sujet en explicitant le rôle du ML dans l’étude du thème

# ### Problématique
# * Recommandation de livres à des utilisateurs
# Les systèmes de recommandation sont largement utilisés aujourd'hui pour recommander des produits aux utilisateurs en fonction de leurs intérêts. Un système de recommandation est l'un des systèmes les plus puissants pour augmenter les profits en retenant plus d'utilisateurs dans une très grande compétition. Dans cet article, je vais vous expliquer comment créer un système de recommandation de livres avec Machine Learning à l'aide du langage de programmation Python.
# 
# Les sites Web de lecture et de vente de livres en ligne comme Kindle et Goodreads se font concurrence sur de nombreux facteurs. L'un de ces facteurs importants est leur système de recommandation de livres. Un système de recommandation de livres est conçu pour recommander des livres intéressants à l'acheteur.
# 
# Le but d'un système de recommandation de livres est de prédire l'intérêt de l'acheteur et de lui recommander des livres en conséquence. Un système de recommandation de livres peut prendre en compte de nombreux paramètres tels que le contenu et la qualité des livres en filtrant les avis des utilisateurs. Dans la section ci-dessous, je vais vous présenter un projet d'apprentissage automatique sur le système de recommandation de livres utilisant Python.
# 
# Dans ce projet, nous allons apprendre à créer un système pour vous recommander de nouveaux livres à lire. Nous commencerons par télécharger les données (lien). Nous allons apprendre quelques astuces pour traiter ces données sur votre machine locale. Ensuite, nous créerons un moteur de recherche pour rechercher dans l'ensemble de données les livres que vous avez lus. Enfin, nous utiliserons la liste des livres que vous avez lus pour trouver des recommandations d'autres personnes.
# 
# En cours de route, nous utiliserons pandas, scikit-learn et numpy..
# 
# À la fin, nous aurons une liste personnalisée de recommandations de livres et un projet que nous pourrons intégrer à notre portfolio. en fin, nous verrons comment améliorer vos recommandations grâce au filtrage collaboratif.
# 
# plan
# Présentation
# Données et aperçu du projet
# Explorer les données en python
# Analyse des métadonnées de notre livre
# Traiter les métadonnées d'un livre avec des pandas
# Construire un moteur de recherche de livres
# Créer une liste de livres aimés
# Explorer les données d'évaluation des livres
# Trouver des utilisateurs qui aiment les mêmes livres que nous
# Trouver les livres que ces utilisateurs ont aimés
# Créer des recommandations de livres initiales
# Améliorer nos recommandations de livres
# Récapitulation et prochaines étapes
# 
# 
# ### Data Description :
# Dataset: books.csv - rating.csv - users.csv
# 
# The data contains the following columns:
# 
# * 
# * 
# * 
# * 

# ### Import Libraries

# In[6]:


#https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/download


# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


from sklearn.cluster import KMeans
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# ### Import & check the Data

# In[13]:


#import csv file into dataframe (df)
#url='https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/download/Books.csv'
#df = pd.read_html(Books.csv)

#reponse=requests.get(url)


# In[14]:


#unzip  nabil

from zipfile import ZipFile

with ZipFile('archive.zip', 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()


# In[15]:


#df = pd.read_csv('books.csv',on_bad_lines='skip')
books = pd.read_csv('books.csv', error_bad_lines=False, engine ='python')
users = pd.read_csv('users.csv', error_bad_lines=False, engine ='python')
ratings = pd.read_csv('ratings.csv', error_bad_lines=False, engine ='python')


# In[16]:


books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher','Image-URL-S','Image-URL-M','Image-URL-L']]
books.rename(columns = {'Book-Title':'title', 'Book-Author':'author', 'Year-Of-Publication':'year', 'Publisher':'publisher','Image-URL-S':'Image_URL_S','Image-URL-M':'Image_URL_M','Image-URL-L':'Image_URL_L'}, inplace=True)
users.rename(columns = {'User-ID':'user_id', 'Location':'location', 'Age':'age'}, inplace=True)
ratings.rename(columns = {'User-ID':'user_id', 'Book-Rating':'rating'}, inplace=True)


# In[17]:


books['Image_URL_M'][1]


# In[18]:


users.dtypes


# In[19]:


books


# In[20]:


users


# In[21]:


ratings


# In[22]:


#show thehead of df
books.head()


# In[23]:


#show the size /shape of df
books.shape


# In[24]:


users.shape


# In[25]:


ratings.shape


# ## Exploration de données

# #### L'ensemble de données qui contient des informations sur les livres, qui a écrit ces livres et d'autres informations pertinentes. Maintenant que nous savons à quoi ressemblent nos données, allons-y et trouvons toutes les valeurs nulles présentes dans nos données :

# In[26]:


#show the structure/info of df
books.info()


# In[27]:


#show the columns' names of files
print(books.columns)
print(users.columns)
print(ratings.columns)


# In[28]:


users.isnull().sum()


# In[29]:


ratings.isnull().sum()


# In[30]:


books.duplicated().sum()


# In[31]:


users.duplicated().sum()


# In[32]:


ratings.duplicated().sum()


# In[33]:


#ratings_count=


# # Popularity Based Recommender System

# Les résultats ci-dessus nous montrent les 10 meilleurs livres de nos données. Nous avons vu que le score maximum dans nos données était de 5,0, mais nous ne voyons aucun livre dans le résultat ci-dessus avec un score de 5,0. En effet, nous avons filtré ces livres en fonction du nombre de notes. Nous nous sommes assurés que tous les livres que nous avons dans les résultats ci-dessus ont une note décente. Il peut y avoir des livres dans les données qui peuvent avoir seulement 1 ou 2 notes peuvent être notés 5.0. Nous voulons éviter de tels livres, c'est pourquoi nous avons utilisé ce type de filtrage.
#books.rename(columns = {'Book-Title':'title', 'Book-Author':'author', 'Year-Of-Publication':'year', 'Publisher':'publisher','Image-URL-S':'Image_URL_S','Image-URL-M':'Image_URL_M','Image-URL-L':'Image_URL_L'}, inplace=True)
#users.rename(columns = {'User-ID':'user_id', 'Location':'location', 'Age':'age'}, inplace=True)
#ratings.rename(columns = {'User-ID':'user_id', 'Book-Rating':'rating'}, inplace=True)
# In[34]:


ratings_with_name = ratings.merge(books,on='ISBN')


# In[35]:


num_rating_df = ratings_with_name.groupby('title').count()['rating'].reset_index()
num_rating_df.rename(columns={'rating':'num_ratings'},inplace=True)
num_rating_df


# In[36]:


avg_rating_df = ratings_with_name.groupby('title').mean()['rating'].reset_index()
avg_rating_df.rename(columns={'rating':'avg_rating'},inplace=True)
avg_rating_df


# In[37]:


popular_df = num_rating_df.merge(avg_rating_df,on='title')
popular_df


# In[38]:


popular_df = popular_df[popular_df['num_ratings']>=250].sort_values('avg_rating',ascending=False).head(50)


# In[39]:


popular_df = popular_df.merge(books,on='title').drop_duplicates('title')[['title','author','Image_URL_M','publisher','num_ratings','avg_rating']]


# In[40]:


#popular_df['Image-URL-M'][0]
popular_df


# In[41]:


popular_df.describe()


# par contre, nous allons jeter un coup d'œil à certains des meilleurs auteurs de nos données. Nous les classerons en fonction du nombre de livres qu'ils ont écrits tant que ces livres sont présents dans les données :

# In[42]:


most_books = books.groupby('author')['title'].count().reset_index().sort_values('title', ascending=False).head(10).set_index('author')
plt.figure(figsize=(15,10))
ax = sns.barplot(most_books['title'], most_books.index, palette='inferno')
ax.set_title("Top 10 authors with most books")
ax.set_xlabel("Total number of books")
totals = []
for i in ax.patches:
    totals.append(i.get_width())
total = sum(totals)
for i in ax.patches:
    ax.text(i.get_width()+.2, i.get_y()+.2,str(round(i.get_width())), fontsize=15,color='black')
plt.show()


# Dans le graphe ci-dessus, Aghata Christie et William Shakespeare ont le plus de livres dans les données. Les deux auteurs ont 1199 livres dans notre ensemble de données, suivis de Stephen King et Ann M. Martin.

# In[43]:


top_ten = popular_df[popular_df['num_ratings'] >= 260]
top_ten.sort_values('avg_rating', ascending=False)
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(10, 10))
data = top_ten.sort_values(by='avg_rating', ascending=False).head(10)
sns.barplot(x='avg_rating', y='title', data=data, palette='inferno')


# Ensuite, nous verrons quels livres ont été les plus commentés. Nous avons la colonne d'évaluation moyenne dans nos données et également le nombre de fois qu'un livre particulier a été évalué. Nous allons essayer d'utiliser cette colonne pour trouver les livres les plus commentés présents dans nos données

# In[44]:


most_rated = popular_df.sort_values('num_ratings', ascending = False).head(10).set_index('title')
plt.figure(figsize=(15,10))
ax = sns.barplot(most_rated['num_ratings'], most_rated.index, palette = 'inferno')
totals = []
for i in ax.patches:
    totals.append(i.get_width())
total = sum(totals)
for i in ax.patches:
    ax.text(i.get_width()+.2, i.get_y()+.2,str(round(i.get_width())), fontsize=15,color='black')
plt.show()


# Nous pouvons voir que Twilight a été noté plus de fois que n'importe quel autre livre ! De plus, ces notes se chiffrent toutes en millions ! Cela signifie donc que Twilight a été revu plus de 4 millions de fois, suivi de The Hobbit or There and Back Again et The Catcher in the Rye qui a été revu plus de 2 millions de fois.

# Essayons de trouver une relation entre notre score moyen et le nombre de scores. Nous faisons cela pour voir comment nous pouvons utiliser ces colonnes dans notre recommandation. Nous vérifierons également la répartition des notes moyennes avec le nombre de pages d'un livre, la langue utilisée dans le livre et le nombre de critiques de texte :

# In[45]:


popular_df.avg_rating = popular_df.avg_rating.astype(float)
fig, ax = plt.subplots(figsize=[15,10])
sns.distplot(popular_df['avg_rating'],ax=ax)
ax.set_title('Average rating distribution for all books',fontsize=20)
ax.set_xlabel('Average rating',fontsize=13)


# In[46]:


ax = sns.relplot(data=popular_df, x="avg_rating", y="num_ratings", color = 'red', sizes=(100, 200), height=7, marker='o')
plt.title("Relation between Rating counts and Average Ratings",fontsize = 15)
ax.set_axis_labels("Average Rating", "Ratings Count")


# In[47]:


plt.figure(figsize=(15,10))
ax = sns.relplot(x="avg_rating", y="publisher", data = popular_df, color = 'red',sizes=(100, 200), height=7, marker='o')
ax.set_axis_labels("Average Rating", "Number of Publisher")


# In[48]:


# Data preparation


# Après avoir comparé la note moyenne avec les différentes colonnes, nous pouvons continuer à utiliser la langue et le nombre de notes pour notre système de recommandation. Pourtant, les autres colonnes n'avaient pas beaucoup de sens et leur utilisation pourrait ne pas nous aider dans une grande mesure, nous pouvions donc les omettre.

# Je vais faire une copie de nos données d'origine juste pour être sûr afin que nous soyons en sécurité au cas où nous gâcher quoi que ce soit :

# In[49]:


popular_df2=popular_df.copy()


# # Collaborative Filtering Based Recommender System (RS)

# In[50]:


x = ratings_with_name.groupby('user_id').count()['rating'] > 200
padhe_likhe_users = x[x].index


# In[51]:


filtered_rating = ratings_with_name[ratings_with_name['user_id'].isin(padhe_likhe_users)]


# In[52]:


y = filtered_rating.groupby('title').count()['rating']>=50
famous_books = y[y].index


# In[53]:


final_ratings = filtered_rating[filtered_rating['title'].isin(famous_books)]


# In[54]:


pt = final_ratings.pivot_table(index='title',columns='user_id',values='rating')


# In[55]:


pt.fillna(0,inplace=True)


# In[56]:


pt


# # Building Book Recommendation System
# # Collaborative Filtering Based Recommender System (RS)

# In[57]:


from sklearn.metrics.pairwise import cosine_similarity


# In[58]:


similarity_scores = cosine_similarity(pt)


# In[59]:


similarity_scores.shape


# In[60]:


def recommend(book_name):
    # index fetch
    index = np.where(pt.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:5]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('title')['title'].values))
        item.extend(list(temp_df.drop_duplicates('title')['author'].values))
        item.extend(list(temp_df.drop_duplicates('title')['Image_URL_M'].values))
        
        data.append(item)
    
    return data


# In[61]:


#recommend('1984')
#recommend('Animal Farm')
#recommend('Message in a Bottle')
recommend('The Notebook')
#recommend('The Da Vinci Code')


# In[62]:


pt.index[545]


# In[63]:


import pickle
pickle.dump(popular_df,open('popular.pkl','wb'))


# In[64]:


popular_df['title'].values


# In[65]:


pickle.dump(popular_df,open('book_dict.pkl','wb'))


# In[66]:


books.drop_duplicates('title')


# In[67]:


pickle.dump(pt,open('pt.pkl','wb'))
pickle.dump(books,open('books.pkl','wb'))
pickle.dump(similarity_scores,open('similarity_scores.pkl','wb'))


# ### END
