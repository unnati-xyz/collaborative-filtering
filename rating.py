
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


def getScore(history, similarities):
   return sum(history*similarities)/sum(similarities)       #similarity score between purchase history and item similarity


def findSimilarity(data_sims,data,neighbours,data_ibs,dataib):
     for i in range(0, len(data_sims.index)):
        for j in range(1, len(data_sims.columns)):           # skip the user column
             user = data_sims.index[i]
             product = data_sims.columns[j]

        if data.ix[i][j] == 1:        #we score items that the user has already consumed as 0, because there is no point recommending it again.
            data_sims.ix[i][j] = 0
        else:
            product_top_names = neighbours.ix[product][1:10]
            product_top_sims = data_ibs.ix[product].order(ascending=False)[1:10]
            user_purchases = dataib.ix[user, product_top_names]

            data_sims.ix[i][j] = getScore(user_purchases, product_top_sims)



def UserBased(number,dataWide,No_of_users,No_of_top_items):                      # User Based Collaborative Filtering
     data_sims1=dataWide.copy()
     data_sims1=dataWide.reset_index()
     data_sims=createDf(data_sims1.index,data_sims1.columns)        # Create a place holder matrix for similarities
     data_sims.ix[:, :1] = data_sims1.ix[:, :1]

     data_sims=findSimilarity(data_sims,data_sims1)

     data_recommend=createDf(data_sims.index,['user','1','2','3','4','5','6'])         #How to assign columns for general number?
     data_recommend.ix[0:,0] = data_sims.ix[:,0]

         for i in range(0,len(data_sims.index)):
          data_recommend.ix[i,1:] = data_sims.ix[i,:].order(ascending=False).ix[1:number+1,].index.transpose()
      print  data_recommend.ix[:No_of_users,:No_of_top_items]


def main_function():
    data=pd.read_csv("/home/vaibhavi/machine-learning/cf_mba/data/groceries.csv")
    data["Quantity"]=1          #Assume that for all items only one quantity was bought

    dataWide=data.pivot("Person", "item", "Quantity")           #Converting data from long to wide format

    dataWide.fillna(0, inplace=True)            #Replace NA with 0
    neighbours=itemBased(5, dataWide)
    print(neighbours.head(5))
    UserBased(6,dataWide,10,4)



main_function()







