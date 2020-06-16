import matplotlib as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sks
from datetime import timedelta
#LOADING THE DATATSET
data = pd.read_csv('covid_19_data_29_April.csv')

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
#name=input('Please Enter The COUNTRY Name = ')
name ='US' #("Please Enter The Country = ")
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

#model input parameters

x = user.iloc[:,-1].values

#x=x/x_len
y=user.iloc[:,-4].values

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
plt.title('Data Before Prediction')
plt.xlabel('Days From 1st Case')
plt.ylabel('Cases')

plt.show()

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

mlp  = MLPRegressor(activation='relu', alpha=0.006 , batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(150) , learning_rate='constant', max_iter=10000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=5, shuffle=False, solver='lbfgs', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)
mlp.fit(x_train , y_train.ravel())

predict_test = mlp.predict(x_test)
predict_train = mlp.predict(x_train)


from sklearn.metrics import r2_score , mean_squared_log_error , mean_squared_error
from math import sqrt

def rmse(a,b) :
  return sqrt(mean_squared_error(a, b))

print("STD  Test Set", np.std(y_test))

print("STD  Prediction Test Set", np.std(predict_test))

print("R2 Score Test="  , r2_score(y_test , predict_test ))

#print("MLG SCORE " , mean_squared_log_error(y_test, predict_test))

print("MSE Train"  , mean_squared_error(y_train, predict_train))

print("RMSE Train" , rmse(y_train, predict_train))

print("MSE Test" , mean_squared_error(y_test, predict_test))

print("RMSE Test" , rmse(y_test, predict_test))


plt.scatter(x_train, y_train, color = 'black')
plt.plot(x_test , y_test , color='black')

plt.plot(x_train, predict_train, color = 'blue')
plt.plot(x_test, predict_test, color = 'blue')

plt.title('Model Fit')
plt.xlabel('Days From 1st Case')
plt.ylabel('confirmed Cases')

plt.show()

#plotting for the future os user given country
#Counrty Model
#predicting th future dates

h= user['Days'].max()
ctry_fut = []

for i in range(11) :
    h+=1
    ctry_fut.append(h)

ctry_fut=np.array(ctry_fut)
ctry_fut= ctry_fut.reshape(-1,1)

print("List of Future Dates After Reshape = \n" , ctry_fut)
#prediction for the future dates for both gloabal
#Global Model

print("Last Update = " , user['ObservationDate'].max())

plt.scatter(x, y, color = 'blue')
plt.plot(ctry_fut, mlp.predict(ctry_fut), color = 'red')


plt.title('Future for Country')
plt.xlabel('Days')
plt.ylabel('Confirmed Cases')
#plt.show()

print("Future Values = " , mlp.predict(ctry_fut))

#ceating a worldwide data for better understanding

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


#now finding the pattern in gloabl confirmed usoing perceptron
#model input parameters

X = global_data.iloc[:,-1].values
Y=global_data.iloc[:,-4].values

#print("Before Reshape x \n" , x )
#print("Before Reshape y \n" , y )
X= X.reshape(-1,1)
Y=Y.reshape(-1,1)


#modelling a linea regression model for before the inflection point

#fitting the logistic growth curve for confirmed cases
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5 ,shuffle= False)


#scaling the features


#scaler = StandardScaler()

#print("Before Scaling y_train \n" , y_train)
#print("Before Scaling y_test \n" , y_test)

#scaler.fit(Y_train) #learning the feature on train
#Y_train=scaler.transform(Y_train)
#Y_test = scaler.transform(Y_test)
#
# Y_train = np.log(Y_train)
# Y_test = np.log(Y_test)

#print("After Scaling y_train \n" , y_train)
#print("After Scaling y_test \n" , y_test)


#importing the nural net

mlp_gloabal  = MLPRegressor(activation='relu', alpha=0.003 , batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,learning_rate_init=0.0001,
       hidden_layer_sizes=(180) , learning_rate='constant', max_iter=10000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=5, shuffle=False, solver='lbfgs', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False )
mlp_gloabal.fit(X_train , Y_train.ravel())


predict_global_test = mlp_gloabal.predict(X_test)
predict_gloabal_train = mlp_gloabal.predict(X_train)







print("R2 Score Test="  , r2_score(Y_test , predict_global_test ))

print("MLG SCORE " , mean_squared_log_error(Y_test, predict_global_test))

print("MSE Train"  , mean_squared_error(Y_train, predict_gloabal_train))

print("RMSE Train" , rmse(Y_train, predict_gloabal_train))

print("MSE Test" , mean_squared_error(Y_test, predict_global_test))

print("RMSE Test" , rmse(Y_test, predict_global_test))



plt.scatter(X_train, Y_train, color = 'black')
plt.plot(X_train, predict_gloabal_train, color = 'blue')
plt.plot(X_test, predict_global_test, color = 'blue')
plt.plot(X_test , Y_test , color='black')

plt.title('Model Fit Global')
plt.xlabel('Days From 1st Case')
plt.ylabel('confirmed Cases')
#plt.show()

print("Last Update = " , global_data['ObservationDate'].max())

#predicting th future dates
m= global_data['Days'].max()
Obs_date_fut = []

for i in range(11) :
    m+=1
    Obs_date_fut.append(m)



Obs_date_fut=np.array(Obs_date_fut)
Obs_date_fut= Obs_date_fut.reshape(-1,1)

print("List of Future Dates After Reshape = \n" , Obs_date_fut)
#prediction for the future dates for both gloabal
#Global Model



plt.scatter(X, Y, color = 'blue')
plt.plot(Obs_date_fut, mlp_gloabal.predict(Obs_date_fut), color = 'red')


plt.title('Future Global')
plt.xlabel('Days')
plt.ylabel('Confirmed Cases')
#plt.show()

print("Future Values = " , mlp_gloabal.predict(ctry_fut))
