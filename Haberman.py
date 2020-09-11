
# coding: utf-8

# # EDA on Haberman’s Cancer Survival Dataset

# # 1. Understanding the dataset
# Title: Haberman’s Survival Data
# Description: The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago’s Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# Attribute Information:
# Age of patient at the time of operation (numerical)
# Patient’s year of operation (year — 1900, numerical)
# Number of positive axillary nodes detected (numerical)
# Survival status (class attribute) :
# 1 = the patient survived 5 years or longer
# 2 = the patient died within 5 years

# # The data set has four attributes 3 are features and 1 class attribute .There are 306 instances of data.
# 1.Number of Axillary nodes(Lymph Nodes)
# 2.Age
# 3.Operation Year
# 4.Survival Status

# In[164]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
df=pd.read_csv("haberman.csv")
df.head()


# In[165]:


#number of rows and columns in the data
df.shape


# In[166]:


#Information about the data with 306 rows and all 4 columns are numerical
#There are no missing values so the data is clean and ready for the EDA
df.info()


# In[167]:


#checking for missing values
#no missing values in the data
df.isnull().sum()


# In[168]:


# The column names in our dataset
print (df.columns)


# In[169]:


print(df['Survival Status'].unique())


# In[170]:


df['Survival Status'] = df['Survival Status'].map({1:'Yes', 2:'No'})
df.head() 
#mapping the values of 1 and 2 to yes and no respectively and 
#printing the first 5 records from the dataset.


# In[171]:


df['Survival Status'].value_counts()


# In[172]:


# plot the histogram for the classes - binary classification, only two classes
count_classes = pd.value_counts(df["Survival Status"])
count_classes.plot(kind = 'bar')
plt.title("Class distribution Histogram")
plt.xlabel("Survival Status")
plt.ylabel("Frequency")
plt.show()


# # Observations:
# The value_counts() function tells how many data points for each class are present. Here, it tells how many patients survived and how many did not survive.
# Out of 306 patients, 225 patients survived and 81 did not.
# The dataset is imbalanced.

# In[173]:


#Describing statistical info about the data
df.describe()


# Observation: Age values are between 30-83.
#              Year of operation is between 1958-1969.
#              axillary nodes are between 0-52,about 75% of patients have less than or equal to 4 axillary nodes and 25% of patients are having 0 nodes.
#    

# # 1. Univariate Analysis
# The major purpose of the univariate analysis is to describe, summarize and find patterns in the single feature.

# # (a)Probability Density Function(PDF)
# Probability Density Function (PDF) is the probability that the variable takes a value x. (a smoothed version of the histogram)
# Here the height of the bar denotes the percentage of data points under the corresponding group.
# By looking at their PDF graphs and the amount of separation and overlapping between different classes, we can decide which features gives useful insight and choose that feature.

# In[174]:


sns.FacetGrid(df, hue="Survival Status", size=6)    .map(sns.distplot, "Age")    .add_legend();
plt.show();


# Observation:No major conclusion can be drawn because data points are overlapping.
#             Patients with the age less than 40 are more likely to survive than who are at 70s.
#             

# In[175]:


sns.FacetGrid(df, hue="Survival Status", size=6)    .map(sns.distplot, "Year of operation")    .add_legend();
plt.show();


# Observation:The survival status corresponding to operation year data points are overlapping, hence no conclusion about the survival status of the patient could be drawn based on the Year of operation. Except that the patient who had undergone surgery between the year 1960-1963 has higher probability of survival.

# In[176]:


sns.FacetGrid(df, hue="Survival Status", size=6)    .map(sns.distplot, "Axillary nodes")    .add_legend();
plt.show();


# From axillary nodes which is very important features The patients with axiilary nodes between 0 to 5 are more likely to survive.

# Observation:Axillary nodes is the most important feature among all which helps us in bringing insights for survival status (class label)

# # CDF

# In[177]:


survived=df.loc[df['Survival Status']=='Yes']
notsurvived=df.loc[df['Survival Status']=='No']


# In[178]:


count,edges = np.histogram(survived['Age'],bins=10,density=True)
pdf = count/(sum(count))
cdf = np.cumsum(pdf)
plt.plot(edges[1:],pdf)
plt.plot(edges[1:],cdf)
print('survived:')
print('bin_edges: {}'.format(edges))
print('pdf: {}'.format(pdf))
print('***********************************************')
plt.title('pdf and cdf of people based on age')
count,edges = np.histogram(notsurvived['Age'],bins=10,density=True)
pdf = count/(sum(count))
cdf = np.cumsum(pdf)
plt.plot(edges[1:],pdf)
plt.plot(edges[1:],cdf)
plt.legend(['pdf of survived','cdf of survived','pdf of not survived','cdf of not survived'])
print('notsurvived:')
print('bin_edges: {}'.format(edges))
print('pdf: {}'.format(pdf))
print('*********************************************')
plt.show()


# In[179]:


count,edges = np.histogram(survived['Year of operation'],bins=10,density=True)
pdf = count/(sum(count))
cdf = np.cumsum(pdf)
plt.plot(edges[1:],pdf)
plt.plot(edges[1:],cdf)
print('survived:')
print('bin_edges: {}'.format(edges))
print('pdf: {}'.format(pdf))
print('***********************************************')
plt.title('pdf and cdf of people based on year of operation')
count,edges = np.histogram(notsurvived['Year of operation'],bins=10,density=True)
pdf = count/(sum(count))
cdf = np.cumsum(pdf)
plt.plot(edges[1:],pdf)
plt.plot(edges[1:],cdf)
plt.legend(['pdf of survived','cdf of survived','pdf of not survived','cdf of not survived'])
print('notsurvived:')
print('bin_edges: {}'.format(edges))
print('pdf: {}'.format(pdf))
print('*********************************************')
plt.show()


# In[180]:


count,edges = np.histogram(survived['Axillary nodes'],bins=10,density=True)
pdf = count/(sum(count))
cdf = np.cumsum(pdf)
plt.plot(edges[1:],pdf)
plt.plot(edges[1:],cdf)
print('survived:')
print('bin_edges: {}'.format(edges))
print('pdf: {}'.format(pdf))
print('***********************************************')
plt.title('pdf and cdf of people based on axillary nodes')
count,edges = np.histogram(notsurvived['Axillary nodes'],bins=10,density=True)
pdf = count/(sum(count))
cdf = np.cumsum(pdf)
plt.plot(edges[1:],pdf)
plt.plot(edges[1:],cdf)
plt.legend(['pdf of survived','cdf of survived','pdf of not survived','cdf of not survived'])
print('notsurvived:')
print('bin_edges: {}'.format(edges))
print('pdf: {}'.format(pdf))
print('*********************************************')
plt.show()


# Observations: Almost 80% of the  survived patients have less than or equal to 5 positive lymph nodes.
#               people with age less than 35 are more likely to be survived.
#               

# # BIVARIATE ANALYSIS

# Boxplots ,violion plot ,pairplot,scatter plot

# In[181]:


plt.figure(1,figsize=(14,4))
plt.subplot(1,3,1)
sns.boxplot(x='Survival Status', y='Age', data=df)
plt.subplot(1,3,2)
sns.boxplot(x='Survival Status',y='Year of operation', data=df)
plt.subplot(1,3,3)
sns.boxplot(x='Survival Status',y='Axillary nodes', data=df)
plt.show()


# In[182]:


plt.figure(1,figsize=(14,4))
plt.subplot(1,3,1)
sns.violinplot(x='Survival Status', y='Age', data=df)
plt.subplot(1,3,2)
sns.violinplot(x='Survival Status',y='Year of operation', data=df)
plt.subplot(1,3,3)
sns.violinplot(x='Survival Status',y='Axillary nodes', data=df)
plt.show()


# Observation: People with the age less than 35 are more likely to survive.
#              No such difference in the plot for year of operation.
#              we observe that the patients with more axillary nodes are more likely to die.
#              from the plot we can see that patients having node between 1 to 25 are having  more chances of death.

# # Pair plot

# In[183]:


sns.set_style('darkgrid')
sns.pairplot(df,hue='Survival Status',size=4)
plt.show()


# Observation:No such correlation is found between features 
#             From the plot for axillary nodes and age we can observe that Younger people are most likely to have less nodes and               their chances of survival is more

# # OBSERVATION

# We observed that Axillary nodes and age are important attributes.

# Patients with age 35 or less than 35 are having higher chances of survival.

# Patients with age greater than 60 will not survive 5 years or longer.

# Patients with axillary nodes 0 - 5 are more likely to survive.(More the positive lymph nodes less chances of survival)
