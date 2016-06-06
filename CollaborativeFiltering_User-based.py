# Collaborative Filtering #User Based collaborative Filtering

# Import libraries
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine


class Functions: 

    def create_df(self, dindex, dcolumns):
        return pd.DataFrame(index=dindex, columns=dcolumns)

    def find_similarity(self, data_to_be_filled, data_used):                  # Similarity Measure using cosine distances
        for i in range(0, len(data_to_be_filled.columns)):
            for j in range(0, len(data_to_be_filled.columns)):                # Loop through the columns for each column
                data_to_be_filled.ix[i, j] = cosine(data_used.ix[:, i],data_used.ix[:, j])

        return data_to_be_filled

    def swap(a,b):
        return b,a
    

class Run:

    def __init__(self, file):

        self.data=pd.read_csv(file)
        self.data["Quantity"] = 1                                               # Assume that for all items only one quantity was bought
        self.dataWide = self.data.pivot("Person", "item", "Quantity")           # Converting data from long to wide format
        self.dataWide.fillna(0, inplace=True)                                   # Replace NA with 0
        self.data_purchases = self.dataWide.copy()
        self.data_purchases = self.data_purchases.reset_index()
        self.data_purchases = self.data_purchases.drop("Person", axis=1)  # To make the 'Person' field just another column and not an index
        self.data_itemvsuser = self.data_purchases.copy()

    def main(self):
        user1 = Filter()
        user1.get_recommendation(10, 10)


class User(Run):

    def __init__(self):

        self.fun = Functions()
        self.user_sim_smaller = self.user_similarity()
        self.user_neighbours = self.user_neighbours()
        self.user_purchases =  self.user_purchases()

    def user_similarity(self):

        self.user_sim = self.fun.create_df(self.data_itemvsuser.index, self.data_itemvsuser.index)
        self.user_sim_smaller = self.user_sim.iloc[:100, :100]
        self.user_sim_smaller = self.fun.find_similarity(self.user_sim_smaller, self.data_purchases)

        return self.user_sim_smaller

    def user_neighbours(self):       #Find neighbours to users based on similarity

        self.user_neighbours = self.fun.create_df(self.user_sim_smaller.columns,range(1, 11))
        for i in range(0, len(self.user_sim_smaller.columns)):       # Loop through our similarity dataframe and fill in neighbouring item names
            self.user_neighbours.ix[i, :] = np.array(self.user_sim_smaller.ix[i, :]
                                                     .sort_values(ascending = True)[0:10].index)
            if self.user_neighbours.iloc[i ,0] != i:
                c=self.user_neighbours.iloc[i, :][self.user_neighbours.iloc[i, :] == i].index.tolist()
                self.user_neighbours.iloc[i, 0],self.user_neighbours.iloc[i, c[0] - 1]\
                    = self.fun.swap(self.user_neighbours.iloc[i, 0], self.user_neighbours.iloc[i, c[0] - 1])
        return self.user_neighbours

    def user_purchases(self):      # Find all items purchased by a User

        self.user_purchases = self.fun.create_df(self.user_sim_smaller.columns,range(max(self.data.Person.value_counts())))
        for i in range(len(self.user_sim_smaller.columns)):
           self.user_purchases.iloc[i, :] = pd.Series(self.data_purchases.iloc[i, :][self.data_purchases.iloc[i, :] == 1]
                                                      .index.tolist())
           self.user_purchases = self.user_purchases.fillna(0)
        return self.user_purchases


class Filter(User, Run):

    def __init__(self):
        Run.__init__(self, "/home/vaibhavi/machine-learning/cf_mba/data/groceries.csv")
        User.__init__(self)

    def get_recommendation(self, number, user_number):            # Getting 'number' number of recommendations
        self.recommend = self.fun.create_df(self.user_sim_smaller.columns, range(1, 11))
        for j in range(len(self.user_sim_smaller.columns)):
            self.itemslist = []
            for i in range(2, 11):
                self.items = [self.user_purchases.ix[self.user_neighbours.ix[j, i]][k] for k in range(10)
                            if self.user_purchases.ix[self.user_neighbours.ix[j, i]][k] != 0]
                self.itemslist = list(set(self.itemslist + self.items))
            self.recommend.iloc[j, :] = pd.Series(self.itemslist[0:number + 1])
        print(self.recommend.iloc[user_number, :])


if __name__ == '__main__':
    Run("/home/vaibhavi/machine-learning/cf_mba/data/groceries.csv").main()