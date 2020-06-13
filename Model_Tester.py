import matplotlib as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sks
from datetime import timedelta
#LOADING THE DATATSET
data = pd.read_csv('covid_19 4-June.csv')

#preliminary check on the data
print("reading")
print(data.head(10))


#converting to date time format for further operations
data['ObservationDate']=pd.to_datetime(data['ObservationDate'] , infer_datetime_format=True)
print(data['ObservationDate'].dtype)


#since we have only one blank coloumn we will drop it
#also we dont need last update coloumn
data.drop(["Province/State"],axis=1,inplace=True)
data.drop(["SNo"],axis=1,inplace=True)
data.drop(["Last Update"],axis=1,inplace=True)
print("dropped3")
print(data.head())
print("nasum")
print(data.isna().sum())
print("no of uniq")
unique_value = data['Country/Region'].nunique()
print("No of Countries in the csv = ",unique_value)

#changing the data format
df=data

#now we group data based on countries
print("Observation Date and Country/Region grouping are applied ")
df2=df.groupby(['Country/Region','ObservationDate'], as_index=True, axis=0).aggregate(

   {'Confirmed' :sum ,
     'Deaths' :sum ,
      'Recovered' :sum}

).reset_index()
new=pd.DataFrame(df2)
print(new.head(20))


#Now we create a user-input function
#In which the user defines the country and it show the plot of country

print(new.head(20))
name=input('Please Enter The COUNTRY Name = ')
print('The entered Country was = ' , name)
user=new.loc[new['Country/Region'] == name ]
print("Head for user \n" , user.head(20))

#creating the days column#flase positive warning
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
print("user observation dates \n" , user['ObservationDate'])
#pd.to_datetime(user['ObservationDate'])
user['Days'] = user['ObservationDate']
print("USER HEAD \n" ,user.head(10))
min_date = user['ObservationDate'].min()
max_date=df2['ObservationDate'].max()
print("Minimum Date in USER =" , max_date)
print("Minimum Date in USER =" , min_date)

columnSeriesObj = []
for i in user['Days'] :

    t=i-min_date
    columnSeriesObj.append(t)
user['Days'] = columnSeriesObj
print("Column Overwrite\n " , user['Days'].head(10))
print("Datatype = " , type(user['Days']))
user['Days'] = user['Days'].dt.days
print("DAYS NOW \n" ,user['Days'])

print("User info\n " , user.info)
print("User description \n " , user.describe())
print("DATA FED INTO MODEL \n" , user.head(10))
#model input parameters

x = user.iloc[:,-1].values

#x=x/x_len
y=user.iloc[:,-3].values

#y=y/y_len
#print("Before Reshape x \n" , x )
#print("Before Reshape y \n" , y )
x= x.reshape(-1,1)
y=y.reshape(-1,1)




#fitting the Percceptron model for confirmed cases
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5 ,shuffle= False)


plt.scatter(x_train, y_train, color = 'black')
plt.plot(x_test , y_test , color='black')



plt.title('Full Data')
plt.xlabel('Days From 1st Case')
plt.ylabel('confirmed Cases')

plt.show()

#creating a score dataset
cnames = ['Accuracy' , 'Size' , 'alpha'  ]
acc_matrix=pd.DataFrame(columns =cnames )
print(acc_matrix.head())
acc_lst=[]
i_lst=[]
nr_list=[]
act_list=[]
solv_list=[]
iterate_list = [0.0000000001, 0.0000000002, 0.0000000003, 0.0000000004, 0.0000000005,0.0000000006, 0.0000000007, 0.0000000008, 0.0000000009,
                0.000000001, 0.000000002, 0.000000003, 0.000000004, 0.000000005,0.000000006, 0.000000007, 0.000000008, 0.000000009,
                0.00000001, 0.00000002, 0.00000003, 0.00000004, 0.00000005,0.00000006, 0.00000007, 0.00000008, 0.00000009,
                0.0000001 , 0.0000002 , 0.0000003 , 0.0000004 , 0.0000005 ,0.0000006 , 0.0000007 , 0.0000008 , 0.0000009,
                0.000001 , 0.000002 , 0.000003 , 0.000004 , 0.000005 , 0.000006 , 0.000007 , 0.000008 , 0.000009 ,
                0.00001 , 0.00002 , 0.00003 , 0.00004 , 0.00005  , 0.00006 , 0.00007 , 0.00008 , 0.00009 ,
                0.0001 , 0.0002 , 0.0003 , 0.0004 , 0.0005 , 0.0006 , 0.0007 , 0.0008 , 0.0009 ,
                0.001 , 0.002 , 0.003 , 0.004 , 0.005 ,0.006 , 0.007 , 0.008 , 0.009 ,
                0.01 , 0.02 , 0.03 , 0.04 , 0.05 ,0.06 , 0.07 , 0.08 , 0.09
                ]

#importing the nural net

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score , mean_squared_log_error , mean_squared_error
from math import sqrt

def rmse(a,b) :
  return sqrt(mean_squared_error(a, b))

print("Model Training :")

#model Testing Module


for nr in range(40 , 190,10):
 print("Nural Size = " ,nr)
 for i in iterate_list :

  mlp  = MLPRegressor(activation='relu', alpha=i , batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(nr) , learning_rate='constant', max_iter=90000000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=5, shuffle=False, solver='lbfgs', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=True)
  mlp.fit(x_train , y_train.ravel())

  predict_test = mlp.predict(x_test)
  scr2 = r2_score(y_test, predict_test)
  acc_lst.append(scr2)
  i_lst.append(i)
  nr_list.append(nr)
  print(" i = ", i, "Score = ", scr2)

print("Training Complete")
print()
acc_matrix['Accuracy'] = acc_lst
acc_matrix['Size'] = nr_list
acc_matrix['alpha'] = i_lst
acc_matrix.reset_index()

print(acc_matrix.head())
for i in acc_matrix.index:
    if acc_matrix['Accuracy'][i] == max(acc_matrix['Accuracy']):
     print("Accuracy ", acc_matrix["Accuracy"][i])
     print("Nural Size ", acc_matrix['Size'][i])
     print("aplha ="  , acc_matrix['alpha'][i])



acc_matrix.to_csv("MODEL TESTER ")
