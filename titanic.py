#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.tree import DecisionTreeClassifier
import pandas as pd
df = pd.read_csv('titanic.csv')
print(df.to_string())


# In[2]:


print(df.shape)


# In[3]:


print(df.isna().sum())


# In[4]:


mean=df['Age'].mean()
print(mean)


# In[5]:


df['Age']=df['Age'].fillna(mean)


# In[6]:


df['Age']


# In[7]:


newdf = df.drop(columns = ['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'])
print(newdf.to_string())


# In[8]:


newdf1 = pd.get_dummies(newdf,dtype=int)
print(newdf1.to_string())


# In[9]:


x = newdf1.drop(columns = ['Survived'])
print(x)
y = df['Survived']
print(y)


# In[10]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size = 0.20)


# In[11]:


model = DecisionTreeClassifier()
model.fit(X_train,Y_train)


# In[12]:


print(X_train.shape)
print(X_test.shape)


# In[13]:


ans=model.predict(X_test)
print(ans)


# In[14]:


model.score(X_train,Y_train)


# In[ ]:




