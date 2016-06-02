# # Collaborative Filtering #User Based collaborative Filtering

#Import libraries
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

def createDf(dindex, dcolumns):
    return pd.DataFrame(index=dindex,
                        columns=dcolumns)

def findSimilarity(dataToBeFilled, dataUsed):                   # Similarity Measure using cosine distances
    for i in range(0,len(dataToBeFilled.columns)):
        for j in range(0,len(dataToBeFilled.columns)):                #Loop through the columns for each column
            dataToBeFilled.ix[i,j] = cosine(dataUsed.ix[:,i],dataUsed.ix[:,j])

    return dataToBeFilled

# Create a place holder matrix for similarities, and fill in the user name column
def userSimilarity(data_itemvsuser,data_purchases ):
    user_sim = createDf(data_itemvsuser.index, data_itemvsuser.index)
    user_sim_smaller=user_sim.iloc[:100,:100]
    user_sim_smaller= findSimilarity(user_sim_smaller, data_purchases)

    return user_sim_smaller

def swap(a,b):
    return b,a

def userNeighbours(user_sim_smaller):       #Find neighbours to users based on similarity
    user_neighbours = createDf(user_sim_smaller.columns,range(1,11))
    for i in range(0,len(user_sim_smaller.columns)): # Loop through our similarity dataframe and fill in neighbouring item names
        user_neighbours.ix[i, :] = np.array(user_sim_smaller.ix[i,:].sort_values(ascending=True)[0:10].index)
        if user_neighbours.iloc[i,0] != i:
            c=user_neighbours.iloc[i,:][user_neighbours.iloc[i,:] == i].index.tolist()
            user_neighbours.iloc[i,0],user_neighbours.iloc[i,c[0]-1]=swap(user_neighbours.iloc[i,0], user_neighbours.iloc[i,c[0]-1])
    return user_neighbours

def userPurchases(data,user_sim_smaller,data_purchases):    #Find all items purchased by a user
    user_purchases=createDf(user_sim_smaller.columns,range(max(data.Person.value_counts())))
    for i in range(len(user_sim_smaller.columns)):
        user_purchases.iloc[i,:]=pd.Series(data_purchases.iloc[i,:][data_purchases.iloc[i,:]==1].index.tolist())
    user_purchases=user_purchases.fillna(0)
    return user_purchases

def getRecommendation(user_purchases, user_neighbours, user_sim_smaller, number): #getting 'number' number of recommendations
    recommend=createDf(user_sim_smaller.columns, range(1,11))
    for j in range(len(user_sim_smaller.columns)):
        itemslist=[]
        for i in range(2,11):
            items=[user_purchases.ix[user_neighbours.ix[j,i]][k] for k in range(10) if user_purchases.ix[user_neighbours.ix[j,i]][k]!=0]
            itemslist=list(set(itemslist+items))
        recommend.iloc[j,:]=pd.Series(itemslist[0:number+1])
    return recommend


def main():
    data=pd.read_csv("/home/vaibhavi/machine-learning/cf_mba/data/groceries.csv")
    data["Quantity"]=1          #Assume that for all items only one quantity was bought
    dataWide=data.pivot("Person", "item", "Quantity")           #Converting data from long to wide format
    dataWide.fillna(0, inplace=True)            #Replace NA with 0
    data_purchases = dataWide.copy()
    data_purchases = data_purchases.reset_index()             #To make the 'Person' field just another column and not an index
    data_purchases = data_purchases.drop("Person", axis=1)
    data_itemvsuser = data_purchases.copy()
    user_sim_smaller= userSimilarity(data_itemvsuser,data_purchases)
    user_neighbours= userNeighbours(user_sim_smaller)
    user_purchases=  userPurchases(data,user_sim_smaller,data_purchases)
    recommend=getRecommendation(user_purchases, user_neighbours, user_sim_smaller, 10)
    print(recommend.head(10))

if __name__ == '__main__':
    main()