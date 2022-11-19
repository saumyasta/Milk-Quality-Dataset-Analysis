#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix


# In[3]:


dataset = pd.read_csv('milknew.csv')


# In[4]:


X = dataset.iloc[:, 3:13].values


# In[5]:


y = dataset.iloc[:, 3:13].values


# In[6]:


print(X)
print(y)


# In[7]:


dataset.dtypes


# #### Encoding categorical data

# In[8]:


labelencoder_X_1 = LabelEncoder()


# In[9]:


X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])


# In[10]:


labelencoder_X_2 = LabelEncoder()


# In[11]:


X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])


# In[12]:


sns.set_theme()
sns.set(rc = {'figure.figsize':(12,6)})
SEED = 2022


# In[13]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
dataset['Grade']=encoder.fit_transform(dataset['Grade'])
dataset


# #### Splitting the dataset into the Training set and Test set

# In[14]:


X_train, X_test, y_train, y_test = train_test_split(
    dataset.iloc[:,:-1], 
    dataset.iloc[:,-1],
    test_size = 0.2,
    random_state = 12345
)
X_train.dtypes


# In[15]:


print(y_train)


# In[16]:


X_test.dtypes


# In[17]:


y_train.dtypes


# In[18]:


y_test.dtypes


# #### Feature Scaling

# In[19]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train.shape,
X_test.shape,
y_train.shape,
y_test.shape)


# #### Compiling the ANN

# In[21]:


classifier = Sequential()
classifier.add(Dense(units=20, input_dim=7, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=15, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=10, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=3, kernel_initializer='uniform', activation='softmax'))
classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
survivalANN_Model=classifier.fit(X_train,y_train, batch_size=10, epochs=100, verbose=1)


# In[ ]:




