
# coding: utf-8

# In[3]:


from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[6]:


import pandas as pd
with open("nba_2013.csv", 'r') as csvfile:
    nba = pd.read_csv(csvfile)
    
nba.head(50)


# In[7]:


nba.shape


# In[8]:


nba['player'].value_counts()


# In[9]:


nba['pos'].value_counts()


# In[10]:


nba['season'].value_counts()


# In[11]:


nba['season_end'].value_counts()


# In[13]:


#Remove "Player" column as Player Name is unique and will lead to incorrect model if included
#Remove Season which is having same value for all rows
#Remove Season_end which is also having same value for all rows and is object type
nbadata = nba.iloc[:,1:29]

#nbadata.drop('player', axis=1)
nbadata.head()


# In[15]:


charcter_col = nbadata.dtypes.pipe(lambda x: x[x=='object']).index
charcter_col


# In[16]:


#Here label_mapping will be having all unique values for each column stored in a dictionary
label_mapping= {}
for c in charcter_col:
    nbadata[c], label_mapping[c] = pd.factorize(nbadata[c])


nbadata.head()


# In[17]:


nbadata.isnull().any() # identify if a column has Null Value or not


# In[18]:


nbadata.fillna(nbadata.mean(), inplace=True)


# In[19]:


nbadata.head(10)


# In[20]:


X = nbadata.drop('pts', axis=1)
Y = nbadata['pts']
X.head()


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

for K in range(25):
    K_value = K+1
    neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    print("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:",K_value)


# In[22]:


X_train[:1:]


# In[23]:


y_test[1]


# In[24]:


y_pred[1]


# In[25]:


neigh.predict(X_train[:10:])


# In[26]:


X_train[:1:]

