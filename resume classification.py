#import needed libraries
import pandas as pd
import numpy as np


# import the preprocessing and text libraries from scikitlearn
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics

# string manipulation libraries
import re
import string
import nltk
from nltk.corpus import stopwords

# vizuslization libraries
import matplotlib.pyplot as plt
import seaborn as sns

from kneed import KneeLocator

# Load the data
data = pd.read_csv('jobss.csv')

# initial shape of data
shape_before = data.shape
shape_before

# Drop the empty column, all columns indicating location and salary columns
data.drop(['Unnamed: 1', 'Location', 'Longitude', 'Latitude', 'Job Experience Required', 'sal'], axis = 1, inplace = True )

# Drop records when there is no job title, industry, role or functional area provided
data.dropna(subset=['Job Title', 'Key Skills', 'Role Category', 'Industry', 'Role', 'Functional Area'], inplace = True)

# new shape pf data
data.shape

# Drop records with invalid data
data= data[data.Role != 'Other']
data= data[data.Industry != 'Other']
data= data[data['Role Category'] != 'Other']
data= data[data['Functional Area'] != 'Other']
data= data[data['Key Skills'] != 'vide']

# new shape pf data
data.shape

# text preprocessing function which:
# - removes special characters ( | , / & . () and - )
# - removes numbers
# - transforms in lowercase
# - removes excessive whitespaces
def preprocess_text(text: str) -> str:
    """Args:
        text (str): the input text you want to clean
    Returns:
        str: the cleaned text
    """
    
    # remove special characters and numbers
    text = re.sub("[|,/&.()]+", " ", text)
    text = re.sub("[-]+", " ", text) # included here as previous line processes as a range input
  
    # return text in lower case and stripped of whitespaces
    text = text.lower().strip()
    
    return text

# Apply the function to individual columns of the data
data['Job Title'] = data['Job Title'].apply(lambda x: preprocess_text(x))
data['Key Skills'] = data['Key Skills'].apply(lambda x: preprocess_text(x))
data['Role Category'] = data['Role Category'].apply(lambda x: preprocess_text(x))
data['Functional Area'] = data['Functional Area'].apply(lambda x: preprocess_text(x))
data['Industry'] = data['Industry'].apply(lambda x: preprocess_text(x))
data['Role'] = data['Role'].apply(lambda x: preprocess_text(x))

#merge all the columns in a new dataframe object to complete data analysis
data2 = pd.DataFrame()
data2['Job Indicators'] = data['Job Title'] + " " + data['Key Skills'] + " " + data['Role Category'] + " " + data['Functional Area'] + " " + data['Industry'] + " " + data['Role']

# initialize the vectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
# fit_transform applies TF-IDF to clean texts - we save the array of vectors in X
X = vectorizer.fit_transform(data2['Job Indicators'])

#initialize empty array to hold SSE values
SSE_values = []

for i in range(1, 20):
        kmeans = KMeans(n_clusters = i, random_state = 42)
        kmeans.fit(X)
        SSE_values.append(kmeans.inertia_)

# Apply KneeLocator method to identify optimal number of clusters
# this optimal number is increased later due to cluster purity
kneedle = KneeLocator(range(1, 20), SSE_values, S=1.0, curve='convex', direction='decreasing')
print('Elbow: ', kneedle.elbow)
kneedle.plot_knee()

# initialize kmeans with 7 centroids based on identified best SSE
kmeans_opt = KMeans(n_clusters=7, random_state=42)
# fit the model
kmeans_opt.fit(X)
# store cluster labels in a variable
clusters = kmeans_opt.labels_

# initialize PCA indicating to keep 85% of variance in the original data
pca = PCA(n_components=0.95, svd_solver ='full', random_state=42)
pca_vecs = pca.fit_transform(X.toarray())

# pass our X to the pca and store the reduced vectors into pca_vecs
# save our two dimensions into x0 and x1
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]

data2['cluster'] = clusters
data2['x0'] = x0
data2['x1'] = x1

# Function to identify top key words in each cluster so as to assign best cluster names
def get_top_keywords(n_terms):
    """This function returns the keywords for each centroid of the KMeans"""
    data2 = pd.DataFrame(X.todense()).groupby(clusters).mean() # groups the TF-IDF vector by cluster
    terms = vectorizer.get_feature_names() # access tf-idf terms
    for i,r in data2.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]])) # for each row of the dataframe, find the n terms that have the highest tf idf score
        
# we retrieve the top 10 words for each cluster
get_top_keywords(10)

# map clusters to appropriate labels 
cluster_map = {0: "administration", 1: "engineering", 2: "services and operations", 3: "processes", 4: "cooperate business", 5: "financial", 6: "information technology"}
# apply mapping
data2['cluster'] = data2['cluster'].map(cluster_map)

# set image size
plt.figure(figsize=(12, 7))
# set a title
plt.title("TF-IDF + KMeans Job Category Clustering", fontdict={"fontsize": 18})
# set axes names
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
# create scatter plot with seaborn, where hue is the class used to group the data
sns.scatterplot(data=data2, x='x0', y='x1', hue='cluster', palette="viridis")
plt.show()

