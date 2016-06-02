# # Collaborative Filtering

#Import libraries
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

data=pd.read_csv("/home/vaibhavi/machine-learning/cf_mba/data/groceries.csv")
data["Quantity"]=1          #Assume that for all items only one quantity was bought

dataWide=data.pivot("Person", "item", "Quantity")           #Converting data from long to wide format

dataWide.fillna(0, inplace=True)            #Replace NA with 0

#User Based collaborative Filtering
data_purchases = dataWide.copy()
data_purchases = data_purchases.reset_index()             #To make the 'Person' field just another column and not an index
data_purchases = data_purchases.drop("Person", axis=1)

data_itemvsuser = data_purchases.copy()

# Create a place holder matrix for similarities, and fill in the user name column
user_sim = pd.DataFrame(index=data_itemvsuser.index, columns=data_itemvsuser.index)

user_sim_smaller=user_sim.iloc[:100,:100]

for i in range(0,len(user_sim_smaller.columns)) :
    # Loop through the columns for each column
    for j in range(0,len(user_sim_smaller.columns)) :
      user_sim_smaller.ix[i,j] = cosine(data_purchases.ix[i,:],data_purchases.ix[j,:])





