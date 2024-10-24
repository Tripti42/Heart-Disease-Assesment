#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df= pd.read_csv("heart.csv")


# In[3]:


df= df.drop(['oldpeak','slp','thall'],axis=1)


# In[4]:


df.head()


# # ### Data Analysis

# # Understanding our DataSet:
# 
# #### Age : Age of the patient
# 
# #### Sex : Sex of the patient
# 
# #### exang: exercise induced angina (1 = yes; 0 = no)
# 
# #### ca: number of major vessels (0-3)
# 
# #### cp : Chest Pain type chest pain type
# 
# - Value 0: typical angina
# - Value 1: atypical angina
# - Value 2: non-anginal pain
# - Value 3: asymptomatic
# 
# #### trtbps : resting blood pressure (in mm Hg)
# 
# #### chol : cholestoral in mg/dl fetched via BMI sensor
# 
# #### fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# 
# #### rest_ecg : resting electrocardiographic results
# 
# - Value 0: normal
# - Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
# - Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
# 
# #### thalach : maximum heart rate achieved
# 
# #### target : 0= less chance of heart attack 1= more chance of heart attack

# df.shape

# In[7]:


df.isnull().sum()


# In[8]:


#### As we can see there are no null values in our Data Set


# In[9]:


df.corr()


# In[10]:


sns.heatmap(df.corr())


# In[11]:


#### As we can see our variables are not highly correlated to each other 


# In[12]:


#### We will do Uni and Bi variate analysis on our Features


# In[60]:


plt.figure(figsize=(20, 10))
plt.title("Age of Patients")
plt.xlabel("Age")
sns.countplot(x='age',data=df)


# In[14]:


#### As we can see the Patients are of Age Group 51-67years in majority


# In[61]:


plt.figure(figsize=(20, 10))
plt.title("Sex of Patients,0=Female and 1=Male")

sns.countplot(x='sex',data=df)


# In[65]:


cp_data = df['cp'].value_counts().reset_index()
cp_data.columns = ['cp_type', 'count']  # Rename the columns

# Update the cp_type values
cp_data['cp_type'] = cp_data['cp_type'].replace({
    0: 'Typical Angina',
    1: 'Atypical Angina',
    2: 'Non-anginal',
    3: 'Asymptomatic'
})

cp_data


# In[69]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Example DataFrame (replace this with your actual data)
cp_data = pd.DataFrame({
    'index': ['Type 1', 'Type 2', 'Type 3'],
    'cp': [10, 15, 5]
})

# Check the DataFrame
print(cp_data.head())
print(cp_data.columns)

plt.figure(figsize=(20, 10))
plt.title("Chest Pain of Patients")

# Creating a bar plot
sns.barplot(x=cp_data['index'], y=cp_data['cp'])

# Show the plot
plt.xlabel('Chest Pain Type')
plt.ylabel('Number of Patients')
plt.xticks(rotation=45)
plt.show()


# # We have seen how the the Chest Pain Category is distributed

# In[71]:


# Assuming df is already defined and contains a 'restecg' column
ecg_data = df['restecg'].value_counts().reset_index()
ecg_data.columns = ['index', 'count']  # Rename columns for clarity

# Update the index values using .loc
ecg_data.loc[0, 'index'] = 'normal'
ecg_data.loc[1, 'index'] = 'having ST-T wave abnormality'
ecg_data.loc[2, 'index'] = 'showing probable or definite left ventricular hypertrophy by Estes'

# Display the updated DataFrame
print(ecg_data)


# In[73]:


# Assuming ecg_data has already been defined as follows:
# ecg_data = df['restecg'].value_counts().reset_index()
# ecg_data.columns = ['index', 'count']

plt.figure(figsize=(20, 10))
plt.title("ECG Data of Patients")

# Use the correct column names
sns.barplot(x=ecg_data['index'], y=ecg_data['count'])

# Show the plot
plt.xlabel('ECG Type')  # Customize as needed
plt.ylabel('Number of Patients')  # Customize as needed
plt.xticks(rotation=45)  # Rotate x-axis labels if needed
plt.show()


# # This is our ECG Data 

# In[75]:


sns.pairplot(df, hue='output')

# Show the plot
plt.show()


# # Let us see for our Continuous Variable

# In[76]:


plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
sns.distplot(df['trtbps'], kde=True, color = 'magenta')
plt.xlabel("Resting Blood Pressure (mmHg)")
plt.subplot(1,2,2)
sns.distplot(df['thalachh'], kde=True, color = 'teal')
plt.xlabel("Maximum Heart Rate Achieved (bpm)")


# In[77]:


plt.figure(figsize=(10,10))
sns.distplot(df['chol'], kde=True, color = 'red')
plt.xlabel("Cholestrol")


# # We have done the Analysis of the data now let's have a look at out data

# In[79]:


df.head()


# # Let us do Standardisation

# In[81]:


from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
scale.fit(df)


# In[82]:


df= scale.transform(df)


# In[83]:


df=pd.DataFrame(df,columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
       'exng', 'caa', 'output'])


# In[84]:


df.head()


# # We can insert this data into our ML Models

# # We will use the following models for our predictions :
# - Logistic Regression
# - Decision Tree
# - Random Forest
# - K Nearest Neighbour

# # Let us split our data

# In[86]:


x= df.iloc[:,:-1]
x


# In[87]:


y= df.iloc[:,-1:]
y


# In[88]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)


# # Logistic Regression

# In[90]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
lbl= LabelEncoder()
encoded_y= lbl.fit_transform(y_train)


# In[91]:


logreg= LogisticRegression()
logreg = LogisticRegression()
logreg.fit(x_train, encoded_y)


# In[94]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[95]:


encoded_ytest= lbl.fit_transform(y_test)


# In[96]:


Y_pred1 = logreg.predict(x_test)
lr_conf_matrix = confusion_matrix(encoded_ytest,Y_pred1 )
lr_acc_score = accuracy_score(encoded_ytest, Y_pred1)


# In[97]:


lr_conf_matrix


# In[98]:


print(lr_acc_score*100,"%")


# # As we see the Logistic Regression Model have a 85% accuracy

# In[116]:


from sklearn.tree import DecisionTreeClassifier
tree= DecisionTreeClassifier()
tree.fit(x_train,encoded_y)
ypred2=tree.predict(x_test)
encoded_ytest= lbl.fit_transform(y_test)
tree_conf_matrix = confusion_matrix(encoded_ytest,ypred2 )
tree_acc_score = accuracy_score(encoded_ytest, ypred2)


# In[117]:


tree_conf_matrix


# In[118]:


print(tree_acc_score*100,"%")


# # Random Forest

# In[119]:


from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier()
rf.fit(x_train,encoded_y)
ypred3 = rf.predict(x_test)
rf_conf_matrix = confusion_matrix(encoded_ytest,ypred3 )
rf_acc_score = accuracy_score(encoded_ytest, ypred3)


# In[120]:


rf_conf_matrix


# In[121]:


print(rf_acc_score*100,"%")


# # K Nearest Neighbour
# 
# #### We have to select what k we will use for the maximum accuracy
# #### Let's write a function for it
# 
# 

# In[122]:


from sklearn.neighbors import KNeighborsClassifier


error_rate= []
for i in range(1,40):
    knn= KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,encoded_y)
    pred= knn.predict(x_test)
    error_rate.append(np.mean(pred != encoded_ytest))
    
    
    plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.xlabel('K Vlaue')
plt.ylabel('Error rate')
plt.title('To check the correct value of k')
plt.show()


# # As we see from the graph we should select K= 12 as it gives the best error rate

# In[123]:


knn= KNeighborsClassifier(n_neighbors=12)
knn.fit(x_train,encoded_y)
ypred4= knn.predict(x_test)

knn_conf_matrix = confusion_matrix(encoded_ytest,ypred4 )
knn_acc_score = accuracy_score(encoded_ytest, ypred4)


knn_conf_matrix


# In[124]:


print(knn_acc_score*100,"%")


# # As we see KNN gives us an accuracy of around 85% which is good

# # Let us build a proper confusion matrix for our model

# In[126]:


# Confusion Matrix of  Model enlarged
options = ["Disease", 'No Disease']

fig, ax = plt.subplots()
im = ax.imshow(lr_conf_matrix, cmap= 'Set3', interpolation='nearest')

# We want to show all ticks...
ax.set_xticks(np.arange(len(options)))
ax.set_yticks(np.arange(len(options)))
# ... and label them with the respective list entries
ax.set_xticklabels(options)
ax.set_yticklabels(options)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(options)):
    for j in range(len(options)):
        text = ax.text(j, i, lr_conf_matrix[i, j],
                       ha="center", va="center", color="black")

ax.set_title("Confusion Matrix of Logistic Regression Model")
fig.tight_layout()
plt.xlabel('Model Prediction')
plt.ylabel('Actual Result')
plt.show()
print("ACCURACY of our model is ",lr_acc_score*100,"%")


# In[115]:


## We have succesfully made our model which predicts weather a person is having a risk of Heart Disease or not with 85.7% accuracy


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




