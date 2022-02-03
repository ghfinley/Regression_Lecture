# Running a Regression in Python 

# Can use packages statsmodels to do the regression like stata and R

import pandas #package to manipulate the dataframe

from sklearn import linear_model

dataset = pandas.read_csv("dataset.csv")

#print(dataset)

target = dataset.iloc[:,0].values  #This puts the 1st collumn in a differnt dataframe [R,C], .values puts in a matrix

#print(target)

data = dataset.iloc[:,3:9].values #This puts the dependant variables in a data set called data, need to go to 9 to index 8
print(data)

# Linear Model - using this model to predict, the machine is learing based on the data that we give it. 

machine = linear_model.LinearRegression()
print(machine)
machine.fit(data, target)
print(machine)


#Prediction based on new data

new_data = [
[-0.5,1.1,0.88,0.4,3,0],
[0.6,1.4,-0.2,1,2,1]
]

new_target = machine.predict(new_data)
print(new_target)