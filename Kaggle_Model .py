import datetime
import time
import matplotlib as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sks
from datetime import timedelta
from datetime import datetime , date
import scipy
#LOADING THE DATATSET
data = pd.read_csv('covid_19 4-June.csv')
#data['Country/Region'].replace( ",", "", inplace=True)

#preliminary check on the data
print("reading")
print(data.head(10))

#describing the datatset
print("Desr")
print(data.describe())
print("Info")
data.info()
print(data['ObservationDate'].dtype)

#converting to date time format for further operations
data['ObservationDate']=pd.to_datetime(data['ObservationDate'] , infer_datetime_format=True)
print(data['ObservationDate'].dtype)

#data cleaning
#checking wether we have null values
print("nullsum")
print(data.isnull().sum())
print("nasum")
print(data.isna().sum())
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
#finding the latest date in ObservationDate

# latest_date=data['ObservationDate'].max()
# print("Last Observation Data on the csv =",latest_date)
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
#we now plot the data by finding the latest date for each state

max_date=df2['ObservationDate'].max()
print(max_date)
plt1=new.loc[new['ObservationDate'] == max_date ]
print(plt1.head(10))
#plotting the data from the result dataframe

#plotting the above data using seaborn
sns.barplot(data =plt1 , x ='Country/Region', y='Confirmed' )
plt.xlabel("Country")
plt.ylabel("Confirmed Cases")
#plt.show()

sns.barplot(data =plt1 , x ='Country/Region', y='Deaths' )
plt.xlabel("Country")
plt.ylabel("Deaths")
#plt.show()

sns.barplot(data =plt1 , x ='Country/Region', y='Recovered' )
plt.xlabel("Country")
plt.ylabel("Recovered")
#plt.show()


print("PLOT DATA \n" , plt1.head())
#plotting the confirmed cases in an ineractive worldmap
import plotly.graph_objs as go
fig =go.Figure(data = go.Choropleth(locations=plt1['Country/Region'] , #location coordinate
                               z=plt1['Confirmed'] , #data to be colored
                             locationmode="country names", #for country names
                                    colorscale='Reds' ,
                                    colorbar_title= 'Confirmed',
                                    ))
fig.update_layout(
    title = 'Confirmed Cases Across The World'
)
#fig.show()
#plotting the number of Death Cases around the world
fig =go.Figure(data = go.Choropleth(locations=plt1['Country/Region'] , #location coordinate
                               z=plt1['Deaths'] , #data to be colored
                             locationmode="country names", #for country names
                                    colorscale='Reds' ,
                                    colorbar_title= 'Deaths',
                                    ))
fig.update_layout(
    title = 'Deaths Across The World'
)
#fig.show()
#plotting the number of recovered cases around the world
fig =go.Figure(data = go.Choropleth(locations=plt1['Country/Region'] , #location coordinate
                               z=plt1['Recovered'] , #data to be colored
                             locationmode="country names", #for country names
                                    colorscale='Reds' ,
                                    colorbar_title= 'Recovered',
                                    ))
fig.update_layout(
    title = 'Recovered Cases Across The World'
)
#fig.show()

#ceating a worldwide data for better understanding
# The sum doesnt involve country since total is always the same
print("Making a global sum for each day")
g1=df2.groupby(['ObservationDate'], as_index=True, axis=0).aggregate(

   {'Confirmed' :sum ,
     'Deaths' :sum ,
      'Recovered' :sum}

).reset_index()
global_data=pd.DataFrame(g1)
print(global_data.head(20))


#creating a day counter for each day from begening
min_date_global = global_data['ObservationDate'].min()
max_date_global=global_data['ObservationDate'].max()
print("Maximum Date in Global =" , max_date_global)
print("Minimum Date in Global =" , min_date_global)

global_data['Days'] = global_data['ObservationDate']

columnSeriesObj = []
for i in global_data['Days'] :

    t=i-min_date_global
    columnSeriesObj.append(t)
global_data['Days'] = columnSeriesObj
print("Column Overwrite\n " , global_data['Days'].head(10))
print("Datatype = " , type(global_data['Days']))
global_data['Days'] = global_data['Days'].dt.days
print("DAYS NOW \n" ,global_data['Days'])
print("Gloabl Head \n" , global_data.head(20))
print("Gloabl Tail \n" , global_data.tail(20))


#Global Data Tracking
#user country time wise analysis

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_squared_error , mean_squared_log_error
from sklearn.neural_network import MLPRegressor
from math import sqrt

def rmse(a,b) :
  return sqrt(mean_squared_error(a, b))
#############################################################################################################################
#Confirmed Module


#model input parameters
X = global_data.iloc[:,-1].values
Y=global_data.iloc[:,-4].values

#print("Before Reshape x \n" , x )
#print("Before Reshape y \n" , y )
X= X.reshape(-1,1)
Y=Y.reshape(-1,1)

print("Confirmed Model :")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5 ,shuffle= False)

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
#for i in iterate_list :

# mlp_gloabal_Conf  = MLPRegressor(activation='relu', alpha=i , batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,learning_rate_init=0.0001,
#        hidden_layer_sizes=(180) , learning_rate='constant', max_iter=10000, momentum=0.9,
#        n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
#        random_state=5, shuffle=False, solver='lbfgs', tol=0.0001,
#        validation_fraction=0.1, verbose=False, warm_start=True )
# mlp_gloabal_Conf.fit(X_train , Y_train.ravel())
#
#
# predict_global_test = mlp_gloabal_Conf.predict(X_test)
# predict_gloabal_train = mlp_gloabal_Conf.predict(X_train)
# sr= r2_score(Y_test, predict_global_test)
# print(" i = " , i ,  "Score = "  , sr )
#
# print("Training Complete")
# print()
# print()
#model input parameters
D = global_data.iloc[:,-1].values
T=global_data.iloc[:,-4].values

#print("Before Reshape x \n" , x )
#print("Before Reshape y \n" , y )
D= D.reshape(-1,1)
T=T.reshape(-1,1)

cnames = ['Accuracy' , 'Size' , 'alpha']
acc_matrix=pd.DataFrame(columns =cnames )
print(acc_matrix.head())
acc_lst=[]
i_lst=[]
nr_list=[]
X_train_Death, X_test_Death, Y_train_Death, Y_test_Death = train_test_split(D, T, test_size=0.2, random_state=5 ,shuffle= False)
print("Death Model :")
for nr in range(30 , 200 ,10):
 print("Nural Size = " ,nr)
 for i in iterate_list :

  mlp_gloabal_Death  = MLPRegressor(activation='relu', alpha=i , batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,learning_rate_init=0.0001,
       hidden_layer_sizes=(nr) , learning_rate='constant', max_iter=100000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=5, shuffle=False, solver='lbfgs', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=True )
  mlp_gloabal_Death.fit(X_train_Death , Y_train_Death.ravel())
  predict_global_test_Death = mlp_gloabal_Death.predict(X_test_Death)
  predict_gloabal_train_Death = mlp_gloabal_Death.predict(X_train_Death)

  scr2 = r2_score(Y_test_Death , predict_global_test_Death )

  acc_lst.append(scr2)
  i_lst.append(i)
  nr_list.append(nr)
  print(" i = ", i, "Score = ", scr2)

print("Training Complete")
print()
acc_matrix['Accuracy']=acc_lst
acc_matrix['Size']=nr_list
acc_matrix['alpha']=i_lst

print(acc_matrix.head())
for i in acc_matrix.index :
    if acc_matrix['Accuracy'][i] == max(acc_matrix['Accuracy']) :
        print("Accuracy " , i)
        print("Nural Size " , acc_matrix['Size'][i])
        print("Alpha" ,acc_matrix['alpha'][i] )

acc_matrix.to_csv("ACC MATRIX")
#Global Recovered  Model
#Perceptron Model
# from sklearn.neural_network import MLPRegressor
# #model input parameters
# H = global_data.iloc[:,-2].values
# #
# #print("Before Reshape x \n" , x )
# #print("Before Reshape y \n" , y )
# H= H.reshape(-1,1)
# X_train_Recovered, X_test_Recovered, Y_train_Recovered, Y_test_Recovered = train_test_split(D, H, test_size=0.2,
#                                                                                             random_state=5,
#                                                                                             shuffle=False)
# print("Recovered Model :")
#
# #for i in iterate_list :
#
# mlp_gloabal_Recov  = MLPRegressor(activation='relu', alpha=i, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,learning_rate_init=0.0001,
#        hidden_layer_sizes=(100) , learning_rate='constant', max_iter=100000, momentum=0.9,
#        n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
#        random_state=5, shuffle=False, solver='lbfgs', tol=0.0001,
#        validation_fraction=0.1, verbose=False, warm_start=True )
# mlp_gloabal_Recov.fit(X_train_Recovered , Y_train_Recovered.ravel())
#
# predict_global_test_Recovered = mlp_gloabal_Recov.predict(X_test_Recovered)
#
#
# scr3=  r2_score(Y_test_Recovered , predict_global_test_Recovered )
# print(" i = ", i, "Score = ", scr3)
#
# print("Training Complete")
# print()
# print()