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
import datetime
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_squared_error , mean_squared_log_error
from sklearn.neural_network import MLPRegressor
from math import sqrt
import csv
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
#making a stacked bar graph
fig = go.Figure(data = [
    go.Bar( name ="Confirmed",x=plt1['Country/Region'] , y=plt1['Confirmed'] ),
    go.Bar(name='Recovered',x=plt1['Country/Region'] , y=plt1['Recovered']),
    go.Bar(name='Deaths',x=plt1['Country/Region'] , y=plt1['Deaths'])
]
)
fig.update_layout(barmode = 'group',xaxis_tickangle=-45,title_text='Country Wise Data')
#fig.show()

#Pie chart
fig = px.pie(data_frame=plt1 , names = 'Country/Region' ,values='Confirmed', title = 'Country Wise Confirmed' )
#fig.show()
fig = px.pie(data_frame=plt1 , names = 'Country/Region' ,values='Recovered', title = 'Country Wise Recovered' )
#fig.show()
fig = px.pie(data_frame=plt1 , names = 'Country/Region' ,values='Deaths', title = 'Country Wise Deaths' )
#fig.show()



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

#making bubble graph for country wise data data

#confirmed bubble chart
fig = px.scatter( data_frame=plt1 ,x="Country/Region" , y="Confirmed" ,  color = 'Country/Region' , hover_name = 'Country/Region' ,size ="Confirmed" , size_max=80)
#fig.show()

#recovered bubble chart
fig = px.scatter( data_frame=plt1 ,x="Country/Region" , y="Recovered" ,  color = 'Country/Region' , hover_name = 'Country/Region' ,size ="Recovered" , size_max=80)
#fig.show()

#death bubble chart
fig = px.scatter( data_frame=plt1 ,x="Country/Region" , y="Deaths" ,  color = 'Country/Region' , hover_name = 'Country/Region' ,size ="Deaths" , size_max=80)
#fig.show()

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
#fig.show()

fig = px.scatter_3d( data_frame=cstm , x=cstm['Country/Region'] ,y=cstm['Confirmed'],z=cstm['Deaths'])
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
print("Column Overwrite\n " , global_data['Days'].head())
print("Datatype = " , type(global_data['Days']))
global_data['Days'] = global_data['Days'].dt.days
print("DAYS NOW \n" ,global_data['Days'])
print("Gloabl Head \n" , global_data.head())


#AUTO TRAINING MODEL

####################################
#Trcking Function For The model paprameters in a cvs

def Track_MERLIN(matrix) :
 with open("MERLIN_Trcaker.csv", "a", newline="") as f:
  writer = csv.writer(f)
  writer.writerows(matrix.values)

#making a model tracker dataframe
cnames = ['Accuracy','Model_Number' ,'Run_Date','Country']
Tracker_matrix = pd.DataFrame(columns=cnames)
print(Tracker_matrix.head())
###########################################################################################################################
#Auto Tester Module
def merlin ( dataset , x_tr , x_tst , y_tr ,y_tst ,act_fun , sov_fun,model_number,country ) :
# creating a score dataset
 cnames = ['Accuracy', 'Size', 'alpha','Activation_Function','Solver']
 acc_matrix = pd.DataFrame(columns=cnames)
 print(acc_matrix.head())
 acc_lst = []
 i_lst = []
 nr_list = []
 fun1=[]
 fun2=[]
 mdl_lst = []
 bst_scr = []
 dat_lst = []

 iterate_list = [0.0000000001, 0.0000000002, 0.0000000003, 0.0000000004, 0.0000000005, 0.0000000006, 0.0000000007,
                    #0.0000000008, 0.0000000009,
                    #0.000000001, 0.000000002, 0.000000003, 0.000000004, 0.000000005, 0.000000006, 0.000000007,
                    #0.000000008, 0.000000009,
                    #0.00000001, 0.00000002, 0.00000003, 0.00000004, 0.00000005, 0.00000006, 0.00000007, 0.00000008,
                    #0.00000009,
                    #0.0000001, 0.0000002, 0.0000003, 0.0000004, 0.0000005, 0.0000006, 0.0000007, 0.0000008, 0.0000009,
                    #0.000001, 0.000002, 0.000003, 0.000004, 0.000005, 0.000006, 0.000007, 0.000008, 0.000009,
                    #0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009,
                    #0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
                    #0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                    #0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09
                    ]

# importing the nural net

 from sklearn.neural_network import MLPRegressor
 from sklearn.metrics import r2_score

# model Testing Module

 for nr in range(110, 120, 20):
  print("Nural Size = ", nr)
  for i in iterate_list:
   mlp = MLPRegressor(activation=act_fun, alpha=i, batch_size='auto', beta_1=0.9,
                               beta_2=0.999, early_stopping=False, epsilon=1e-08,
                               hidden_layer_sizes=(nr), learning_rate='constant', max_iter=90000000, momentum=0.9,
                               n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
                               random_state=5, shuffle=False, solver=sov_fun, tol=0.0001,
                               validation_fraction=0.1, verbose=False, warm_start=True )
   mlp.fit(x_tr, y_tr.ravel())

   predict_test = mlp.predict(x_tst)
   scr = r2_score(y_tst, predict_test)
   acc_lst.append(scr)
   i_lst.append(i)
   nr_list.append(nr)
   fun1.append(act_fun)
   fun2.append(sov_fun)
   print(" i = ", i, "Score = ", scr)

 print("Training Complete")
 print()
 acc_matrix['Accuracy'] = acc_lst
 acc_matrix['Size'] = nr_list
 acc_matrix['alpha'] = i_lst
 acc_matrix['Activation_Function'] = fun1
 acc_matrix['Solver'] = fun2
 run_date = date.today()

 acc_matrix.reset_index()

 print(acc_matrix.head())
 for i in acc_matrix.index:
  if acc_matrix['Accuracy'][i] == max(acc_matrix['Accuracy']):
   print("Best Parameters For The Model Is\n")
   print("Accuracy ", acc_matrix["Accuracy"][i])
   print("Nural Size ", acc_matrix['Size'][i])
   print("aplha =", acc_matrix['alpha'][i])
   print("Activation Function =", acc_matrix['Activation_Function'][i])
   print("Solver =", acc_matrix['Solver'][i])
   bst = acc_matrix["Accuracy"][i]
   mdl_lst.append(model_number)
   bst_scr.append(bst)
   dat_lst.append(run_date)
   Tracker_matrix['Accuracy'] = bst_scr
   Tracker_matrix['Model_Number'] = mdl_lst
   Tracker_matrix['Run_Date'] = dat_lst
   Tracker_matrix['Country'] = country
   Track_MERLIN(Tracker_matrix)
   return acc_matrix['Size'][i], acc_matrix['alpha'][i], acc_matrix['Activation_Function'][i], acc_matrix['Solver'][i]
####################################################################################################################################

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





def rmse(a,b) :
  return sqrt(mean_squared_error(a, b))

#Global Data Tracking

#############################################################################################################################
#Confirmed Module
sns.relplot(x='Days',y='Confirmed',kind='line',data=global_data)
plt.title("Confirmed Around The World")
plt.setp(plt.xticks()[1], rotation=30, ha='right') # ha is the same as horizontalalignment
#plt.show()

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

#trainig and implementing the MLP model
sz_1,alp_1,act_1,slv_1 =merlin(global_data,X_train,  X_test, Y_train, Y_test,'relu','lbfgs' , 1,'Global_Conf')

mlp_gloabal_Conf  = MLPRegressor(activation=act_1, alpha=alp_1 , batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,learning_rate_init=0.0001,
       hidden_layer_sizes=(sz_1) , learning_rate='constant', max_iter=100000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=5, shuffle=False, solver=slv_1, tol=0.0001,
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
#Global Model Confimred


plt.scatter(X, Y, color = 'blue')
plt.plot(Obs_date_fut, mlp_gloabal_Conf.predict(Obs_date_fut), color = 'red')

plt.title('Future Global Confirmed')
plt.xlabel('Days From 1st Case')
plt.ylabel('Confirmed Cases')
#plt.show()

##########################################################################################################################
#Deaths Around The World
sns.relplot(x='Days',y='Deaths',kind='line',data=global_data)
plt.title("Deaths Around The World")
plt.setp(plt.xticks()[1], rotation=30, ha='right') # ha is the same as horizontalalignment
#plt.show()
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
#Training and deploying the model
sz_2,alp_2,act_2,slv_2 =merlin(global_data,X_train_Death, X_test_Death, Y_train_Death, Y_test_Death ,'relu','lbfgs', 2,'Global_Death')

mlp_gloabal_Death  = MLPRegressor(activation=act_2, alpha=alp_2 , batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,learning_rate_init=0.0001,
       hidden_layer_sizes=(sz_2) , learning_rate='constant', max_iter=100000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=5, shuffle=False, solver=slv_2, tol=0.0001,
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
#plt.show()


#prediction for the future dates for both gloabal
#Global Death Confimred


plt.scatter(D, T, color = 'blue')
plt.plot(Obs_date_fut, mlp_gloabal_Death.predict(Obs_date_fut), color = 'red')

plt.title('Future Global Deaths')
plt.xlabel('Days From !st Case')
plt.ylabel('Deaths')
#plt.show()



#########################################################################################################################
#Recovered Around The World
sns.relplot(x='Days',y='Recovered',kind='line',data=global_data)
plt.title("Recovered Around The World")
plt.setp(plt.xticks()[1], rotation=30, ha='right') # ha is the same as horizontalalignment
#plt.show()

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
#Training and deploying the mlp
sz_3,alp_3,act_3,slv_3 =merlin(global_data,X_train_Recovered, X_test_Recovered, Y_train_Recovered, Y_test_Recovered ,'relu','lbfgs', 3,'Global_Recov')

mlp_gloabal_Recov  = MLPRegressor(activation=act_3, alpha=alp_3 , batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,learning_rate_init=0.0001,
       hidden_layer_sizes=(sz_3) , learning_rate='constant', max_iter=100000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=5, shuffle=False, solver=slv_3, tol=0.0001,
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
#plt.show()


#prediction for the future dates for both gloabal
#Global Recovered Future


plt.scatter(D, H, color = 'blue')
plt.plot(Obs_date_fut, mlp_gloabal_Recov.predict(Obs_date_fut), color = 'red')

plt.title('Future Global Recovered')
plt.xlabel('Days From 1st Case')
plt.ylabel('Recovered')
#plt.show()

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
#plt.show()

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

#Training and deploying the mlp
sz_4,alp_4,act_4,slv_4 =merlin(user,X_train_Ctry, X_test_Ctry, Y_train_Ctry, Y_test_Ctry,'relu','lbfgs', 4, name+str('Conf'))

mlp_Ctry_Conf  = MLPRegressor(activation=act_4, alpha=alp_4 , batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,learning_rate_init=0.0001,
       hidden_layer_sizes=(sz_4) , learning_rate='constant', max_iter=100000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=5, shuffle=False, solver=slv_4, tol=0.0001,
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
#plt.show()


#prediction for the future dates for both gloabal
#Global Confirmed Future


plt.scatter(Ctry_dates, Ctry_target, color = 'blue')
plt.plot(Obs_date_fut, mlp_Ctry_Conf.predict(Obs_date_fut), color = 'red')

plt.title('Future Country Confirmed')
plt.xlabel('Days From 1st Case')
plt.ylabel('Confirmed')
#plt.show()

##########################################################################################################################

#recovered In the country "US"
sns.relplot(x='Days',y='Recovered',kind='line',data=user)
plt.title("Recovered In The Country")
plt.setp(plt.xticks()[1], rotation=30, ha='right') # ha is the same as horizontalalignment
#plt.show()

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

#Training and deploying the mlp
sz_5,alp_5,act_5,slv_5 =merlin(user,X_train_Ctry, X_test_Ctry, Y_train_Ctry, Y_test_Ctry,'relu','lbfgs', 5, name+str('Recov'))

mlp_Ctry_Conf  = MLPRegressor(activation=act_5, alpha=alp_5 , batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,learning_rate_init=0.0001,
       hidden_layer_sizes=(sz_5) , learning_rate='constant', max_iter=100000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=5, shuffle=False, solver=slv_5, tol=0.0001,
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
#plt.show()


#prediction for the future dates for both gloabal
#Global Recovered Future


plt.scatter(Ctry_dates, Ctry_target, color = 'blue')
plt.plot(Obs_date_fut, mlp_Ctry_Conf.predict(Obs_date_fut), color = 'red')

plt.title('Future Country Recovered')
plt.xlabel('Days From 1st Case')
plt.ylabel('Recovered')
#plt.show()
############################################################################################################################
#Deaths In the country "US"
sns.relplot(x='Days',y='Deaths',kind='line',data=user)
plt.title("Deaths In The Country")
plt.setp(plt.xticks()[1], rotation=30, ha='right') # ha is the same as horizontalalignment
#plt.show()

#Perceptron Model
# from sklearn.neural_network import MLPRegressor
# #model input parameters
Ctry_dates = user.iloc[:,-1].values
Ctry_target = user.iloc[:,-3].values
# #
# #print("Before Reshape x \n" , x )
##print("Before Reshape y \n" , y )
Ctry_dates= Ctry_dates.reshape(-1,1)
Ctry_target = Ctry_target.reshape(-1, 1)

#trian test split
X_train_Ctry_Dth, X_test_Ctry_Dth, Y_train_Ctry_Dth, Y_test_Ctry_Dth = train_test_split(Ctry_dates, Ctry_target, test_size=0.2, random_state=5 ,shuffle= False)

#Training and deploying the mlp
sz_6,alp_6,act_6,slv_6 =merlin(user,X_train_Ctry_Dth, X_test_Ctry_Dth, Y_train_Ctry_Dth, Y_test_Ctry_Dth,'relu','lbfgs', 6, name+str('Death'))

mlp_Ctry_Dth  = MLPRegressor(activation=act_6, alpha=alp_6 , batch_size='auto', beta_1=0.9,
        beta_2=0.999, early_stopping=False, epsilon=1e-08,learning_rate_init=0.0001,
        hidden_layer_sizes=(sz_6) , learning_rate='constant', max_iter=100000, momentum=0.9,
        n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
        random_state=5, shuffle=False, solver=slv_6, tol=0.0001,
        validation_fraction=0.1, verbose=False, warm_start=True)
mlp_Ctry_Dth.fit(X_train_Ctry_Dth , Y_train_Ctry_Dth.ravel())

predict_Ctry_test_Conf_Dth = mlp_Ctry_Dth.predict(X_test_Ctry_Dth)
predict_Ctry_train_Conf_Dth = mlp_Ctry_Dth.predict(X_train_Ctry_Dth)

print()
print()
print("Country Deaths")
print("R2 Score Test="  , r2_score(Y_test_Ctry_Dth , predict_Ctry_test_Conf_Dth ))

print("MLG SCORE Test = " , mean_squared_log_error(Y_test_Ctry_Dth, predict_Ctry_test_Conf_Dth))

print("MSE Train ="  , mean_squared_error(Y_train_Ctry_Dth, predict_Ctry_train_Conf_Dth))

print("RMSE Train =" , rmse(Y_train_Ctry_Dth, predict_Ctry_train_Conf_Dth))

print("MSE Test =" , mean_squared_error(Y_test_Ctry_Dth, predict_Ctry_test_Conf_Dth))

print("RMSE Test =" , rmse(Y_test_Ctry_Dth, predict_Ctry_test_Conf_Dth))

plt.scatter(X_train_Ctry_Dth, Y_train_Ctry_Dth, color = 'black')
plt.plot(X_train_Ctry_Dth, predict_Ctry_train_Conf_Dth, color = 'blue')
plt.plot(X_test_Ctry_Dth, predict_Ctry_test_Conf_Dth, color = 'blue')
plt.plot(X_test_Ctry_Dth , Y_test_Ctry_Dth , color='black')

plt.title('Model Fit Country Deaths')
plt.xlabel('Days From 1st Case')
plt.ylabel('Deaths')
#plt.show()

plt.scatter(Ctry_dates, Ctry_target, color = 'blue')
plt.plot(Obs_date_fut, mlp_Ctry_Dth.predict(Obs_date_fut), color = 'red')

plt.title('Future Country Deaths')
plt.xlabel('Days From 1st Case')
plt.ylabel('Deaths')
#plt.show()



