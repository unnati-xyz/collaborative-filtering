# # Collaborative Filtering
# 
# **Item Based**: which takes similarities between items’ consumption histories

#Import libraries
import pandas as pd
from scipy.spatial.distance import cosine


def createDf(dindex, dcolumns):
    return pd.DataFrame(index=dindex,
                        columns=dcolumns)


def findSimilarity(dataToBeFilled, dataUsed):                   # Similarity Measure using cosine distances
    for i in range(0,len(dataToBeFilled.columns)):
        for j in range(0,len(dataToBeFilled.columns)):                #Loop through the columns for each column
            dataToBeFilled.ix[i,j] = cosine(dataUsed.ix[:,i],dataUsed.ix[:,j])

    return dataToBeFilled



def findNeighbours(neighbours, data, number):
    for i in range(0,len(data.columns)):
        neighbours.ix[i,:number] = data.ix[0:,i].sort_values(ascending=True)[1:number+1].index
    return neighbours



def itemBased(number, dataWide):                        # Item-based Collaborative Filtering
    data_ib = dataWide.copy()
    data_ib = data_ib.reset_index()             #To make the 'Person' field just another column and not an index
    data_ib = data_ib.drop("Person", axis=1)        # In item based collaborative filtering we do not care about the user column, hence drop it

    data_ibs = createDf(data_ib.columns,data_ib.columns)    # Create a placeholder dataframe listing item vs. item
    data_ibs = findSimilarity(data_ibs,data_ib)

    data_neighbours = createDf(data_ibs.columns,range(1,number+1))
    data_neighbours= findNeighbours(data_neighbours, data_ibs, number)
    return data_neighbours

def main():
    data=pd.read_csv("/home/vaibhavi/machine-learning/cf_mba/data/groceries.csv")
    data["Quantity"]=1          #Assume that for all items only one quantity was bought

    dataWide=data.pivot("Person", "item", "Quantity")           #Converting data from long to wide format

    dataWide.fillna(0, inplace=True)            #Replace NA with 0
    neighbours=itemBased(5, dataWide)
    print(neighbours.head(5))

if __name__ == '__main__':
    main()







