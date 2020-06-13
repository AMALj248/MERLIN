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
#name=input('Please Enter The COUNTRY Name = ')
name = input("Please Enter The Country = ")
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

#plotting the data to fing outlier

#scatter plot for outlier detection

#plt.scatter(user['Days'] , user['Confirmed'])
#plt.xlabel("The days from 1st case")
#plt.ylabel("The Confirmed Number of Cases")
#plt.show()

#seeing the z score
from scipy  import stats

#z = np.abs(stats.zscore(user['Days']))
#print(z)
#threshold = 3
#print(np.where(z > 3))

#model input parameters

x = user.iloc[:,-1].values
y=user.iloc[:,-4].values

#print("Before Reshape x \n" , x )
#print("Before Reshape y \n" , y )
x= x.reshape(-1,1)
y=y.reshape(-1,1)


#modelling a linea regression model for before the inflection point

#fitting the logistic growth curve for confirmed cases
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5 ,shuffle= False)


#scaling the features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#print("Before Scaling y_train \n" , y_train)
#print("Before Scaling y_test \n" , y_test)

scaler.fit(y_train) #learning the feature on train
y_train=scaler.transform(y_train)
y_test = scaler.transform(y_test)

#print("After Scaling y_train \n" , y_train)
#print("After Scaling y_test \n" , y_test)


#importing the nural net
from sklearn.neural_network import MLPRegressor
mlp  = MLPRegressor(activation='relu', alpha=0.02 , batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(50) , learning_rate='constant', max_iter=10000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=5, shuffle=False, solver='lbfgs', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False )
mlp.fit(x_train , y_train.ravel())


predict_test = mlp.predict(x_test)
predict_train = mlp.predict(x_train)


from sklearn.metrics import r2_score , mean_squared_log_error , mean_squared_error


score=r2_score(y_test ,predict_test )
print("R2 Score Test="  , score)
#print("MLG SCORE " , mean_squared_log_error(y_test, predict_test))
#print("Difference in value" , y_test - predict_test)
print("MSE Train"  , mean_squared_error(y_train, predict_train))
print("MSE Test" , mean_squared_error(y_test, predict_test))


plt.scatter(x_train, y_train, color = 'black')
plt.plot(x_train, predict_train, color = 'blue')
plt.plot(x_test, predict_test, color = 'blue')
plt.plot(x_test , y_test , color='black')
plt.title('Model Fit')
plt.xlabel('Days')
plt.ylabel('confirmed Adjusted')
plt.show()

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


scaler = StandardScaler()

#print("Before Scaling y_train \n" , y_train)
#print("Before Scaling y_test \n" , y_test)

scaler.fit(Y_train) #learning the feature on train
Y_train=scaler.transform(Y_train)
Y_test = scaler.transform(Y_test)

#print("After Scaling y_train \n" , y_train)
#print("After Scaling y_test \n" , y_test)


#importing the nural net

mlp_gloabal  = MLPRegressor(activation='relu', alpha=0.003 , batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(150) , learning_rate='constant', max_iter=10000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=5, shuffle=False, solver='lbfgs', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False )
mlp_gloabal.fit(X_train , Y_train.ravel())


predict_global_test = mlp_gloabal.predict(X_test)
predict_gloabl_train = mlp_gloabal.predict(X_train)




score2=r2_score(Y_test ,predict_global_test )
print("R2 Score Test="  , score2)
#print("MLG SCORE " , mean_squared_log_error(y_test, predict_test))
#print("Difference in value" , y_test - predict_test)
print("MSE Train"  , mean_squared_error(Y_train, predict_gloabl_train))
print("MSE Test" , mean_squared_error(Y_test, predict_global_test))


plt.scatter(X_train, Y_train, color = 'black')
plt.plot(X_train, predict_gloabl_train, color = 'blue')
plt.plot(X_test, predict_global_test, color = 'blue')
plt.plot(X_test , Y_test , color='black')
plt.title('Model Fit')
plt.xlabel('Days')
plt.ylabel('confirmed Adjusted')
plt.show()

#predicting th future dates
global_data_date_max = global_data['Days'].max()
Obs_date_fut = []
m=global_data_date_max
for i in range(61) :
    m+=1
    Obs_date_fut.append(m)
print(Obs_date_fut)
print("List of Future Dates = \n" , Obs_date_fut)
