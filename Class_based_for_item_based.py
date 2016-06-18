# Collaborative Filtering #Item Based collaborative Filtering

# Import libraries
import pandas as pd
from scipy.spatial.distance import cosine


class DataHandler:

    def get_dataframe(self, dindex, dcolumns):
        return pd.DataFrame(index=dindex, columns=dcolumns)

    def find_similarity(self, data_to_be_filled, data_used):                  # Similarity Measure using cosine distances
        for i in range(0, len(data_to_be_filled.columns)):
            for j in range(0, len(data_to_be_filled.columns)):                # Loop through the columns for each column
                data_to_be_filled.ix[i, j] = cosine(data_used.ix[:, i], data_used.ix[:, j])

        return data_to_be_filled


class Item:

    def __init__(self, file):

        self.data=pd.read_csv(file)
        self.data["Quantity"] = 1                                            # Assume that for all items only one quantity was bought
        self.dataWide = self.data.pivot("Person", "item", "Quantity")        # Converting data from long to wide format
        self.dataWide.fillna(0, inplace=True)                                # Replace NA with 0
        self.data_ib = self.dataWide.copy()
        self.data_ib = self.data_ib.reset_index()
        self.data_ib = self.data_ib.drop("Person", axis=1)
        self.fun = DataHandler()
        self.data_ibs_smaller = self.find_item_similarity()


    def find_item_similarity(self):

        self.data_ibs = self.fun.get_dataframe(self.data_ib.columns,self.data_ib.columns)
        self.data_ibs_smaller = self.data_ibs.iloc[:100, :100]
        self.data_ibs_smaller = self.fun.find_similarity(self.data_ibs_smaller,self.data_ib)

        return self.data_ibs_smaller



class Recommender(Item):

     def __init__(self):
         Item.__init__(self, "C:\Users\Tihor\PycharmProjects\untitled1\src1\groceries.csv")

     def get_recommendation(self,number):
         self.get_item_neighbours = self.fun.get_dataframe(self.data_ibs_smaller.columns,range(1,number+1))
         for i in range(0, len(self.data_ibs_smaller.columns)):
             self.get_item_neighbours.ix[i, :number] = self.data_ibs_smaller.ix[0:, i].sort_values(ascending=True)[1:number+1].index
        print(self.get_item_neighbours.head(number))


if __name__ == '__main__':
    item1 = Recommender()
    item1.get_recommendation(5);








