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
import plotly.graph_objs as go
import plotly.express as px
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
print(new.head())
#we now plot the data by finding the latest date for each state

max_date=df2['ObservationDate'].max()
print(max_date)
plt1=new.loc[new['ObservationDate'] == max_date ]
print(plt1.head())
#plotting the data from the result dataframe
#convert to plotly
#plotting the above data using seaborn
sns.barplot(data =plt1 , x ='Country/Region', y='Confirmed' )
plt.xlabel("Country")
plt.ylabel("Confirmed Cases")
plt.show()

sns.barplot(data =plt1 , x ='Country/Region', y='Deaths' )
plt.xlabel("Country")
plt.ylabel("Deaths")
plt.show()

sns.barplot(data =plt1 , x ='Country/Region', y='Recovered' )
plt.xlabel("Country")
plt.ylabel("Recovered")
plt.show()

#making a stacked bar graph
fig = go.Figure(data = [
    go.Bar( name ="Confirmed",x=plt1['Country/Region'] , y=plt1['Confirmed'] ),
    go.Bar(name='Recovered',x=plt1['Country/Region'] , y=plt1['Recovered']),
    go.Bar(name='Deaths',x=plt1['Country/Region'] , y=plt1['Deaths'])
]
)
fig.update_layout(barmode = 'group',xaxis_tickangle=-45,title_text='Country Wise Data')
fig.show()

#Pie chart
fig = px.pie(data_frame=plt1 , names = 'Country/Region' ,values='Confirmed', title = 'Country Wise Confirmed' )
fig.show()
fig = px.pie(data_frame=plt1 , names = 'Country/Region' ,values='Recovered', title = 'Country Wise Recovered' )
fig.show()
fig = px.pie(data_frame=plt1 , names = 'Country/Region' ,values='Deaths', title = 'Country Wise Deaths' )
fig.show()



print("PLOT DATA \n" , plt1.head())
#plotting the confirmed cases in an ineractive worldmap

fig =go.Figure(data = go.Choropleth(locations=plt1['Country/Region'] , #location coordinate
                               z=plt1['Confirmed'] , #data to be colored
                             locationmode="country names", #for country names
                                    colorscale='Reds' ,
                                    colorbar_title= 'Confirmed',
                                    ))
fig.update_layout(
    title = 'Confirmed Cases Across The World'
)
fig.show()
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
fig.show()
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
fig.show()

#making bubble graph for country wise data data

#confirmed bubble chart
fig = px.scatter( data_frame=plt1 ,x="Country/Region" , y="Confirmed" ,  color = 'Country/Region' , hover_name = 'Country/Region' ,size ="Confirmed" , size_max=80)
fig.show()

#recovered bubble chart
fig = px.scatter( data_frame=plt1 ,x="Country/Region" , y="Recovered" ,  color = 'Country/Region' , hover_name = 'Country/Region' ,size ="Recovered" , size_max=80)
fig.show()

#death bubble chart
fig = px.scatter( data_frame=plt1 ,x="Country/Region" , y="Deaths" ,  color = 'Country/Region' , hover_name = 'Country/Region' ,size ="Deaths" , size_max=80)
fig.show()

#creating a custom dataset for a few top countries
cnames = ['Country/Region' , 'ObservationDate' , 'Confirmed','Deaths','Recovered']
cstm=pd.DataFrame(columns =cnames )
print(cstm.head())
Country_lst=[]
Date_lst=[]
Cnf_list=[]
Dth_list=[]
Recv_list=[]
for i in plt1.index:
    if plt1["Country/Region"][i] in ['US', 'India','UK','Brazil','Spain','Italy','Mainland China','France','Germany',''] :
        Country_lst.append(plt1['Country/Region'][i])
        Date_lst.append(plt1['ObservationDate'][i])
        Cnf_list.append(plt1['Confirmed'][i])
        Dth_list.append(plt1['Deaths'][i])
        Recv_list.append(plt1['Recovered'][i])

cstm['Country/Region'] = Country_lst
cstm['ObservationDate'] = Date_lst
cstm['Confirmed'] = Cnf_list
cstm['Deaths'] = Dth_list
cstm['Recovered'] = Recv_list
cstm.reset_index()
print(cstm.head(10))

fig =px.sunburst( data_frame=cstm , path=['Country/Region','Confirmed', 'Deaths', 'Recovered'],
                  title="Top Countries Data In Sunburst Confirmed ,Deaths ,Recovered" )
fig.show()

fig = px.scatter_3d( data_frame=cstm , x=cstm['Country/Region'] ,y=cstm['Confirmed'],z=cstm['Deaths'])
fig.show()


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
print("Column Overwrite\n " , global_data['Days'].head())
print("Datatype = " , type(global_data['Days']))
global_data['Days'] = global_data['Days'].dt.days
print("DAYS NOW \n" ,global_data['Days'])
print("Gloabl Head \n" , global_data.head())



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
sns.relplot(x='Days',y='Confirmed',kind='line',data=global_data)
plt.title("Confirmed Around The World")
plt.setp(plt.xticks()[1], rotation=30, ha='right') # ha is the same as horizontalalignment
plt.show()

#Global Confirmed Model
#Perceptron Model

#model input parameters
X = global_data.iloc[:,-1].values
Y=global_data.iloc[:,-4].values

#print("Before Reshape x \n" , x )
#print("Before Reshape y \n" , y )
X= X.reshape(-1,1)
Y=Y.reshape(-1,1)




#fitting the logistic growth curve for confirmed cases


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

mlp_gloabal_Conf  = MLPRegressor(activation='relu', alpha=2e-10 , batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,learning_rate_init=0.0001,
       hidden_layer_sizes=(130) , learning_rate='constant', max_iter=100000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=5, shuffle=False, solver='lbfgs', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)
mlp_gloabal_Conf.fit(X_train , Y_train.ravel())


predict_global_test = mlp_gloabal_Conf.predict(X_test)
predict_gloabal_train = mlp_gloabal_Conf.predict(X_train)


print("R2 Score Test="  , r2_score(Y_test , predict_global_test ))

print("MLG SCORE Test = " , mean_squared_log_error(Y_test, predict_global_test))

print("MSE Train ="  , mean_squared_error(Y_train, predict_gloabal_train))

print("RMSE Train =" , rmse(Y_train, predict_gloabal_train))

print("MSE Test =" , mean_squared_error(Y_test, predict_global_test))

print("RMSE Test =" , rmse(Y_test, predict_global_test))



plt.scatter(X_train, Y_train, color = 'black')
plt.plot(X_train, predict_gloabal_train, color = 'blue')
plt.plot(X_test, predict_global_test, color = 'blue')
plt.plot(X_test , Y_test , color='black')

plt.title('Model Fit Global Confirmed')
plt.xlabel('Days From 1st Case')
plt.ylabel('Confirmed Cases')
plt.show()

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
#Global Model Confimred


plt.scatter(X, Y, color = 'blue')
plt.plot(Obs_date_fut, mlp_gloabal_Conf.predict(Obs_date_fut), color = 'red')

plt.title('Future Global Confirmed')
plt.xlabel('Days From 1st Case')
plt.ylabel('Confirmed Cases')
plt.show()

##########################################################################################################################
#Deaths Around The World
sns.relplot(x='Days',y='Deaths',kind='line',data=global_data)
plt.title("Deaths Around The World")
plt.setp(plt.xticks()[1], rotation=30, ha='right') # ha is the same as horizontalalignment
plt.show()
#Global Confirmed Model
#Perceptron Model
from sklearn.neural_network import MLPRegressor
#model input parameters
D = global_data.iloc[:,-1].values
T=global_data.iloc[:,-3].values

#print("Before Reshape x \n" , x )
#print("Before Reshape y \n" , y )
D= D.reshape(-1,1)
T=T.reshape(-1,1)


#modelling a linea regression model for before the inflection point

#fitting the logistic growth curve for confirmed cases


X_train_Death, X_test_Death, Y_train_Death, Y_test_Death = train_test_split(D, T, test_size=0.2, random_state=5 ,shuffle= False)
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

mlp_gloabal_Death  = MLPRegressor(activation='relu', alpha=0.0008 , batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,learning_rate_init=0.0001,
       hidden_layer_sizes=(130) , learning_rate='constant', max_iter=100000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=5, shuffle=False, solver='lbfgs', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)
mlp_gloabal_Death.fit(X_train_Death , Y_train_Death.ravel())


predict_global_test_Death = mlp_gloabal_Death.predict(X_test_Death)
predict_gloabal_train_Death = mlp_gloabal_Death.predict(X_train_Death)




print("R2 Score Test="  , r2_score(Y_test_Death , predict_global_test_Death ))

print("MLG SCORE Test = " , mean_squared_log_error(Y_test_Death, predict_global_test_Death))

print("MSE Train ="  , mean_squared_error(Y_train_Death, predict_gloabal_train_Death))

print("RMSE Train =" , rmse(Y_train_Death, predict_gloabal_train_Death))

print("MSE Test =" , mean_squared_error(Y_test_Death, predict_global_test_Death))

print("RMSE Test =" , rmse(Y_test_Death, predict_global_test_Death))



plt.scatter(X_train_Death, Y_train_Death, color = 'black')
plt.plot(X_train_Death, predict_gloabal_train_Death, color = 'blue')
plt.plot(X_test_Death, predict_global_test_Death, color = 'blue')
plt.plot(X_test_Death , Y_test_Death , color='black')

plt.title('Model Fit Global Deaths')
plt.xlabel('Days From 1st Case')
plt.ylabel('Deaths')
plt.show()


#prediction for the future dates for both gloabal
#Global Death Confimred


plt.scatter(D, T, color = 'blue')
plt.plot(Obs_date_fut, mlp_gloabal_Death.predict(Obs_date_fut), color = 'red')

plt.title('Future Global Deaths')
plt.xlabel('Days From !st Case')
plt.ylabel('Deaths')
plt.show()



#########################################################################################################################
#Recovered Around The World
sns.relplot(x='Days',y='Recovered',kind='line',data=global_data)
plt.title("Recovered Around The World")
plt.setp(plt.xticks()[1], rotation=30, ha='right') # ha is the same as horizontalalignment
plt.show()

#Global Confirmed Model
#Perceptron Model
from sklearn.neural_network import MLPRegressor
#model input parameters
H = global_data.iloc[:,-2].values
#
#print("Before Reshape x \n" , x )
#print("Before Reshape y \n" , y )
H= H.reshape(-1,1)
#


#modelling a linea regression model for before the inflection point

#fitting the logistic growth curve for confirmed cases


X_train_Recovered, X_test_Recovered, Y_train_Recovered, Y_test_Recovered = train_test_split(D, H, test_size=0.2, random_state=5 ,shuffle= False)
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

mlp_gloabal_Recov  = MLPRegressor(activation='relu', alpha=2e-10 , batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,learning_rate_init=0.0001,
       hidden_layer_sizes=(180) , learning_rate='constant', max_iter=100000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=5, shuffle=False, solver='lbfgs', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)
mlp_gloabal_Recov.fit(X_train_Recovered , Y_train_Recovered.ravel())


predict_global_test_Recovered = mlp_gloabal_Recov.predict(X_test_Recovered)
predict_gloabal_train_Recovered = mlp_gloabal_Recov.predict(X_train_Recovered)


print()
print()
print("R2 Score Test="  , r2_score(Y_test_Recovered , predict_global_test_Recovered ))

print("MLG SCORE Test = " , mean_squared_log_error(Y_test_Recovered, predict_global_test_Recovered))

print("MSE Train ="  , mean_squared_error(Y_train_Recovered, predict_gloabal_train_Recovered))

print("RMSE Train =" , rmse(Y_train_Recovered, predict_gloabal_train_Recovered))

print("MSE Test =" , mean_squared_error(Y_test_Recovered, predict_global_test_Recovered))

print("RMSE Test =" , rmse(Y_test_Recovered, predict_global_test_Recovered))



plt.scatter(X_train_Recovered, Y_train_Recovered, color = 'black')
plt.plot(X_train_Recovered, predict_gloabal_train_Recovered, color = 'blue')
plt.plot(X_test_Recovered, predict_global_test_Recovered, color = 'blue')
plt.plot(X_test_Recovered , Y_test_Recovered , color='black')

plt.title('Model Fit Global Recovered')
plt.xlabel('Days From 1st Case')
plt.ylabel('Recovered')
plt.show()


#prediction for the future dates for both gloabal
#Global Recovered Future


plt.scatter(D, H, color = 'blue')
plt.plot(Obs_date_fut, mlp_gloabal_Recov.predict(Obs_date_fut), color = 'red')

plt.title('Future Global Recovered')
plt.xlabel('Days From 1st Case')
plt.ylabel('Recovered')
plt.show()

##########################################################################################################################
##########################################################################################################################

#creating the country wise model for confirmed

#we have already created df2 for countrywise grouping
#Now we create a user-input function
#In which the user defines the country and it show the plot of country

new=pd.DataFrame(df2)
print(new.head(20))

#user input function
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


#########################################################################################################################
#Confirmed In the country "US"
sns.relplot(x='Days',y='Confirmed',kind='line',data=user)
plt.title("Confirmed In The Country")
plt.setp(plt.xticks()[1], rotation=30, ha='right') # ha is the same as horizontalalignment
plt.show()

#Global Confirmed Model
#Perceptron Model
from sklearn.neural_network import MLPRegressor
#model input parameters
Ctry_dates = user.iloc[:,-1].values
Ctry_target = user.iloc[:,-4].values
#
#print("Before Reshape x \n" , x )
#print("Before Reshape y \n" , y )
Ctry_dates= Ctry_dates.reshape(-1,1)
Ctry_target = Ctry_target.reshape(-1, 1)

#trian test split
X_train_Ctry, X_test_Ctry, Y_train_Ctry, Y_test_Ctry = train_test_split(Ctry_dates, Ctry_target, test_size=0.2, random_state=5 ,shuffle= False)

#importing the nural net

mlp_Ctry_Conf  = MLPRegressor(activation='relu', alpha=9e-09 , batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,learning_rate_init=0.0001,
       hidden_layer_sizes=(170) , learning_rate='constant', max_iter=100000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=5, shuffle=False, solver='lbfgs', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)
mlp_Ctry_Conf.fit(X_train_Ctry , Y_train_Ctry.ravel())


predict_Ctry_test_Conf = mlp_Ctry_Conf.predict(X_test_Ctry)
predict_Ctry_train_Conf = mlp_Ctry_Conf.predict(X_train_Ctry)


print()
print()
print("Country Confirmed")
print("R2 Score Test="  , r2_score(Y_test_Ctry , predict_Ctry_test_Conf ))

print("MLG SCORE Test = " , mean_squared_log_error(Y_test_Ctry, predict_Ctry_test_Conf))

print("MSE Train ="  , mean_squared_error(Y_train_Ctry, predict_Ctry_train_Conf))

print("RMSE Train =" , rmse(Y_train_Ctry, predict_Ctry_train_Conf))

print("MSE Test =" , mean_squared_error(Y_test_Ctry, predict_Ctry_test_Conf))

print("RMSE Test =" , rmse(Y_test_Ctry, predict_Ctry_test_Conf))



plt.scatter(X_train_Ctry, Y_train_Ctry, color = 'black')
plt.plot(X_train_Ctry, predict_Ctry_train_Conf, color = 'blue')
plt.plot(X_test_Ctry, predict_Ctry_test_Conf, color = 'blue')
plt.plot(X_test_Ctry , Y_test_Ctry , color='black')

plt.title('Model Fit Country Confirmed')
plt.xlabel('Days From 1st Case')
plt.ylabel('Confirmed')
plt.show()


#prediction for the future dates for both gloabal
#Global Confirmed Future


plt.scatter(Ctry_dates, Ctry_target, color = 'blue')
plt.plot(Obs_date_fut, mlp_Ctry_Conf.predict(Obs_date_fut), color = 'red')

plt.title('Future Country Confirmed')
plt.xlabel('Days From 1st Case')
plt.ylabel('Confirmed')
plt.show()

##########################################################################################################################

#recovered In the country "US"
sns.relplot(x='Days',y='Recovered',kind='line',data=user)
plt.title("Recovered In The Country")
plt.setp(plt.xticks()[1], rotation=30, ha='right') # ha is the same as horizontalalignment
plt.show()

#Perceptron Model
from sklearn.neural_network import MLPRegressor
#model input parameters
Ctry_dates = user.iloc[:,-1].values
Ctry_target = user.iloc[:,-2].values
#
#print("Before Reshape x \n" , x )
#print("Before Reshape y \n" , y )
Ctry_dates= Ctry_dates.reshape(-1,1)
Ctry_target = Ctry_target.reshape(-1, 1)

#trian test spli
X_train_Ctry, X_test_Ctry, Y_train_Ctry, Y_test_Ctry = train_test_split(Ctry_dates, Ctry_target, test_size=0.2, random_state=5 ,shuffle= False)

#importing the nural net

mlp_Ctry_Conf  = MLPRegressor(activation='relu', alpha=0.0008 , batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,learning_rate_init=0.0001,
       hidden_layer_sizes=(140) , learning_rate='constant', max_iter=100000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=5, shuffle=False, solver='lbfgs', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)
mlp_Ctry_Conf.fit(X_train_Ctry , Y_train_Ctry.ravel())


predict_Ctry_test_Conf = mlp_Ctry_Conf.predict(X_test_Ctry)
predict_Ctry_train_Conf = mlp_Ctry_Conf.predict(X_train_Ctry)


print()
print()
print("Country Recovered")
print("R2 Score Test="  , r2_score(Y_test_Ctry , predict_Ctry_test_Conf ))

print("MLG SCORE Test = " , mean_squared_log_error(Y_test_Ctry, predict_Ctry_test_Conf))

print("MSE Train ="  , mean_squared_error(Y_train_Ctry, predict_Ctry_train_Conf))

print("RMSE Train =" , rmse(Y_train_Ctry, predict_Ctry_train_Conf))

print("MSE Test =" , mean_squared_error(Y_test_Ctry, predict_Ctry_test_Conf))

print("RMSE Test =" , rmse(Y_test_Ctry, predict_Ctry_test_Conf))



plt.scatter(X_train_Ctry, Y_train_Ctry, color = 'black')
plt.plot(X_train_Ctry, predict_Ctry_train_Conf, color = 'blue')
plt.plot(X_test_Ctry, predict_Ctry_test_Conf, color = 'blue')
plt.plot(X_test_Ctry , Y_test_Ctry , color='black')

plt.title('Model Fit Country Recovered')
plt.xlabel('Days From 1st Case')
plt.ylabel('Recovered')
plt.show()


#prediction for the future dates for both gloabal
#Global Recovered Future


plt.scatter(Ctry_dates, Ctry_target, color = 'blue')
plt.plot(Obs_date_fut, mlp_Ctry_Conf.predict(Obs_date_fut), color = 'red')

plt.title('Future Country Recovered')
plt.xlabel('Days From 1st Case')
plt.ylabel('Recovered')
plt.show()
############################################################################################################################
#Deaths In the country "US"
# sns.relplot(x='Days',y='Confirmed',kind='line',data=user)
# plt.title("Deaths In The Country")
# plt.setp(plt.xticks()[1], rotation=30, ha='right') # ha is the same as horizontalalignment
# #plt.show()
#
# #Perceptron Model
# from sklearn.neural_network import MLPRegressor
# #model input parameters
# Ctry_dates = user.iloc[:,-1].values
# Ctry_target = user.iloc[:,-3].values
# #
# #print("Before Reshape x \n" , x )
# #print("Before Reshape y \n" , y )
# Ctry_dates= Ctry_dates.reshape(-1,1)
# Ctry_target = Ctry_target.reshape(-1, 1)
#
# #trian test spli
# X_train_Ctry, X_test_Ctry, Y_train_Ctry, Y_test_Ctry = train_test_split(Ctry_dates, Ctry_target, test_size=0.2, random_state=5 ,shuffle= False)
#
# #importing the nural net
#
# mlp_Ctry_Conf  = MLPRegressor(activation='relu', alpha=9e-09 , batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,learning_rate_init=0.0001,
#        hidden_layer_sizes=(170) , learning_rate='constant', max_iter=100000, momentum=0.9,
#        n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
#        random_state=5, shuffle=False, solver='lbfgs', tol=0.0001,
#        validation_fraction=0.1, verbose=False, warm_start=True)
# mlp_Ctry_Conf.fit(X_train_Ctry , Y_train_Ctry.ravel())
#
#
# predict_Ctry_test_Conf = mlp_Ctry_Conf.predict(X_test_Ctry)
# predict_Ctry_train_Conf = mlp_Ctry_Conf.predict(X_train_Ctry)
#
#
# print()
# print()
# print("Country Deaths")
# print("R2 Score Test="  , r2_score(Y_test_Ctry , predict_Ctry_test_Conf ))
#
# print("MLG SCORE Test = " , mean_squared_log_error(Y_test_Ctry, predict_Ctry_test_Conf))
#
# print("MSE Train ="  , mean_squared_error(Y_train_Ctry, predict_Ctry_train_Conf))
#
# print("RMSE Train =" , rmse(Y_train_Ctry, predict_Ctry_train_Conf))
#
# print("MSE Test =" , mean_squared_error(Y_test_Ctry, predict_Ctry_test_Conf))
#
# print("RMSE Test =" , rmse(Y_test_Ctry, predict_Ctry_test_Conf))
#
#
#
# plt.scatter(X_train_Ctry, Y_train_Ctry, color = 'black')
# plt.plot(X_train_Ctry, predict_Ctry_train_Conf, color = 'blue')
# plt.plot(X_test_Ctry, predict_Ctry_test_Conf, color = 'blue')
# plt.plot(X_test_Ctry , Y_test_Ctry , color='black')
#
# plt.title('Model Fit Country Deaths')
# plt.xlabel('Days From 1st Case')
# plt.ylabel('Deaths')
# plt.show()
#
#
# #prediction for the future dates for both gloabal
# #Global Recovered Future
#
#
# plt.scatter(Ctry_dates, Ctry_target, color = 'blue')
# plt.plot(Obs_date_fut, mlp_Ctry_Conf.predict(Obs_date_fut), color = 'red')
#
# plt.title('Future Country Deaths')
# plt.xlabel('Days From 1st Case')
# plt.ylabel('Deaths')
# plt.show()