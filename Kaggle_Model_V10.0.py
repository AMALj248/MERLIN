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
from sklearn.preprocessing import MinMaxScaler , StandardScaler ,MaxAbsScaler , QuantileTransformer , Normalizer
from sklearn.neural_network import MLPRegressor
from math import sqrt
import csv

import timeit
import concurrent.futures
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




#################################################################
#Test/Train Changer
ff =pd.read_csv('MERLIN_TRACKER.csv')
def Split_Maker(model_name , split_value ) :
 mx_dt=ff['Run_Date'].max()
 for row in ff.itertuples():
   if (model_name not in list(ff['Model_Name']) or ff.empty):
    print()
    print()
    print("############################################## First Run Detected ##################################################")
    print()
    print()
    return  split_value
   elif ((row.Model_Name == model_name) ) :
    if ((row.Accuracy) >= 0.90) :
     print("################################################ High Accuracy Model Detected ##########################################" ,split_value )
     return split_value
    if ((row.Accuracy) <= 0.850 ) :
     if((split_value) >= 0.05) :
      print()
      print()
      print("############################################## Split value is reduced ##############################################",split_value-0.05)
      print()
      print()
      return split_value
     else :
      print()
      print()
      print("############################################## Model Is Useless At this point ########################################",split_value)
      print()
      print()
      return split_value



###########################################################################################################################
#Auto Tester Module
def merlin ( dataset , x_tr , x_tst , y_tr ,y_tst ,act_fun , sov_fun,model_number,country ) :

# creating a score dataset
 cnames = ['Accuracy', 'Model_Number','Run_Date','Model_Name','RMSE_Train' , 'RMSE_Test','MSE_Train','MSE_Test' , 'alpha','Nural_Size','Activation_Function','Solver']
 acc_matrix = pd.DataFrame(columns=cnames)
 print(acc_matrix.head())
 acc_lst = []
 i_lst = []
 nr_list = []
 fun1=[]
 fun2=[]
 mdl_lst = []
 RMSE_TRN = []
 RMSE_TST = []
 MSE_TRN = []
 MSE_TST = []
 b1_LST=[]
 b2_LST=[]
 acc_chk = 1

##CHECKIN AND RUNNING THE OLD VALUES FOR FASTER EXCECUTION
 if (ff.empty):
    print('CSV file is empty')
    csv_var = -1
 else:
    print('CSV file is not empty')
    csv_var = 0

 ##CHECKIN AND RUNNING THE OLD VALUES FOR FASTER EXCECUTION
 if (csv_var ==0):
     for row in ff.itertuples() :
      if (row.Model_Name == country) :
       acc_chk = -1
       if (row.Accuracy > 0.950) :

        print("#######################IMPORTANT#############" ,row.Accuracy  )
        print("################################# Pre-Trained Model Detected #####################################################")
        print("################################ Alpha ##################################################", row.alpha , row.Nural_Size , row.Activation_Function , row.Solver)
        return row.Nural_Size, row.alpha, row.Activation_Function, row.Solver , -1

     print("Normal Country =" , country)
     print("String Country =" ,str(country))
     print(type(country))


 if ( (csv_var == -1)  or (acc_chk == -1) or (country not in list(ff['Model_Name']))  ) :
   print("####################################### Training The Model ########################################################")
   iterate_list = [0.0000000001, 0.0000000002, 0.0000000003, 0.0000000004, 0.0000000005, 0.0000000006, 0.0000000007,0.0000000008, 0.0000000009,
                     0.000000001, 0.000000002, 0.000000003, 0.000000004, 0.000000005, 0.000000006, 0.000000007,0.000000008, 0.000000009,
                     0.00000001, 0.00000002, 0.00000003, 0.00000004, 0.00000005, 0.00000006, 0.00000007, 0.00000008,0.00000009,
                     0.0000001, 0.0000002, 0.0000003, 0.0000004, 0.0000005, 0.0000006, 0.0000007, 0.0000008, 0.0000009,
                     0.000001, 0.000002, 0.000003, 0.000004, 0.000005, 0.000006, 0.000007, 0.000008, 0.000009,
                     0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009,
                     0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
                     0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                     0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09
                     ]
   from sklearn.neural_network import MLPRegressor
   from sklearn.metrics import r2_score
   for nr in range(90, 150 ,10):
    print("Nural Size = ", nr)
    for i in iterate_list:
     mlp = MLPRegressor(activation=act_fun, alpha=i, early_stopping=True,
                                   hidden_layer_sizes=(nr), learning_rate='constant', max_iter=100000, momentum=0.9,
                                   n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
                                   random_state=5, shuffle=False, solver=sov_fun, tol=0.0001,
                                   validation_fraction=0.1, verbose=False, warm_start=True )
     mlp.fit(x_tr, y_tr.ravel())

     predict_test = mlp.predict(x_tst)
     predict_train = mlp.predict(x_tr)

     scr = r2_score(y_tst, predict_test)
     MSE_Train = "{:.2f}".format(mean_squared_error(y_tr, predict_train))
     RMSE_Train = "{:.2f}".format(rmse(y_tr, predict_train))
     MSE_Test = "{:.2f}".format(mean_squared_error(y_tst, predict_test))
     RMSE_Test = "{:.2f}".format(rmse(y_tst, predict_test))
     run_date = date.today()

     RMSE_TRN.append(RMSE_Train)
     RMSE_TST.append(RMSE_Test)
     MSE_TRN.append(MSE_Train)
     MSE_TST.append(MSE_Test)
     acc_lst.append(scr)
     i_lst.append(i)
     nr_list.append(nr)
     fun1.append(act_fun)
     fun2.append(sov_fun)
     b1_LST.append(country)
     b2_LST.append(run_date)
     mdl_lst.append(model_number)
     print(" i = ", i, "Score = ", scr)
     if(scr>=0.984) :
       break
   print("Training Complete")
   print()
   acc_matrix['Accuracy'] = acc_lst
   acc_matrix['Model_Number'] = mdl_lst
   acc_matrix['Run_Date'] = b2_LST
   acc_matrix['Model_Name'] = b1_LST
   acc_matrix['RMSE_Train'] = RMSE_TRN
   acc_matrix['RMSE_Test'] = RMSE_TST
   acc_matrix['MSE_Test'] = MSE_TST
   acc_matrix['MSE_Train'] = MSE_TRN
   acc_matrix['alpha'] = i_lst
   acc_matrix['Nural_Size'] = nr_list
   acc_matrix['Activation_Function'] = fun1
   acc_matrix['Solver'] = fun2
   acc_matrix.reset_index()


   print("DATAFRAME =\n", acc_matrix.values)
   print("DATAFRAME =\n", acc_matrix.columns)

   for row in acc_matrix.itertuples() :
    print(row)
    if row.Accuracy == acc_matrix['Accuracy'].max() :
     ss=acc_matrix.iloc[row.Index]
     print("Dataframe Format =\n" , ss)
     Merlin_Tracker(ss)
     print("######## APPENDED ##########" ,country)
     print("Hey Look at MEEE!")
     print(row.Nural_Size,row.alpha, row.Activation_Function,row.Solver)
     print()
     return row.Nural_Size, row.alpha, row.Activation_Function, row.Solver , 0

####################################################################################################################################

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#

#CSV Writing Functions

def FeedIn(Accuracy,Model_Number,Model_Name,RMSE_Train,RMSE_Test,MSE_Train,MSE_Test,alpha,Nural_Size,Activation_Function,Solver) :
    cnames = ['Accuracy', 'Model_Number', 'Run_Date', 'Model_Name', 'RMSE_Train', 'RMSE_Test', 'MSE_Train', 'MSE_Test',
              'alpha', 'Nural_Size', 'Activation_Function', 'Solver']
    acc_matrix_cstm = pd.DataFrame(columns=cnames)
    print(acc_matrix_cstm.head())
    acc_lst = []
    i_lst = []
    nr_list = []
    fun1 = []
    fun2 = []
    mdl_lst = []
    RMSE_TRN = []
    RMSE_TST = []
    MSE_TRN = []
    MSE_TST = []
    b1_LST = []
    b2_LST = []

    run_date = date.today()
    RMSE_TRN.append(RMSE_Train)
    RMSE_TST.append(RMSE_Test)
    MSE_TRN.append(MSE_Train)
    MSE_TST.append(MSE_Test)
    acc_lst.append(Accuracy)
    i_lst.append(alpha)
    nr_list.append(Nural_Size)
    fun1.append(Activation_Function)
    fun2.append(Solver)
    b1_LST.append(Model_Name)
    b2_LST.append(run_date)
    mdl_lst.append(Model_Number)

    acc_matrix_cstm['Accuracy'] = acc_lst
    acc_matrix_cstm['Model_Number'] = mdl_lst
    acc_matrix_cstm['Run_Date'] = b2_LST
    acc_matrix_cstm['Model_Name'] = b1_LST
    acc_matrix_cstm['RMSE_Train'] = RMSE_TRN
    acc_matrix_cstm['RMSE_Test'] = RMSE_TST
    acc_matrix_cstm['MSE_Test'] = MSE_TST
    acc_matrix_cstm['MSE_Train'] = MSE_TRN
    acc_matrix_cstm['alpha'] = i_lst
    acc_matrix_cstm['Nural_Size'] = nr_list
    acc_matrix_cstm['Activation_Function'] = fun1
    acc_matrix_cstm['Solver'] = fun2
    acc_matrix_cstm.reset_index()
    print("Hey Amal = " , acc_matrix_cstm.head())
    print("Alpha Values = ", acc_matrix_cstm['alpha'])
    with open("MERLIN_TRACKER.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(acc_matrix_cstm.values)




def Merlin_Tracker ( acc_data):


    with open("MERLIN_TRACKER.csv" , "a" , newline='') as f:
        writer = csv.writer(f)
        writer.writerow(map(lambda x: x , pd.Series(acc_data)))



def rmse(a,b) :
  return sqrt(mean_squared_error(a, b))


#Score Evaluation Function
def Evaluate_Model(train, test, Y_train, Y_test):
    print("Evaluate Model R2= \n" ,r2_score(Y_test, test),"MSE TRAIN =", mean_squared_error(Y_train, train),"RMSE TRAIN ", rmse(Y_train, train), \
         "MSE TEST",  mean_squared_error(Y_test, test), "RMSE TEST",  rmse(Y_test, test))
    return r2_score(Y_test, test), mean_squared_error(Y_train, train), rmse(Y_train, train), \
           mean_squared_error(Y_test, test),   rmse(Y_test, test)


#Global Data Tracking
#Scaling the input and output
Scaler = MinMaxScaler(feature_range=(0,1))


#model input parameters
D = global_data.iloc[:,-1].values

D= D.reshape(-1,1)
D = Scaler.fit_transform(D)





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
print("GLOBAL DATA =\n" , global_data.head(20))
H = global_data.iloc[:,-2].values
#
#print("Before Reshape x \n" , x )
#print("Before Reshape y \n" , y )
H= H.reshape(-1,1)
H = Scaler.fit_transform(H)
#


#modelling a linea regression model for before the inflection point

#fitting the logistic growth curve for confirmed cases


X_train_Recovered, X_test_Recovered, Y_train_Recovered, Y_test_Recovered = train_test_split(D, H, test_size=Split_Maker('Global_Recov',0.2), random_state=5 ,shuffle= False)
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
#with concurrent.futures.ProcessPoolExecutor() as executor :
sz_3,alp_3,act_3,slv_3,flg =merlin(global_data,X_train_Recovered, X_test_Recovered, Y_train_Recovered, Y_test_Recovered ,'relu','lbfgs', 3,'Global_Recov')

mlp_gloabal_Recov  = MLPRegressor(activation=act_3, alpha=alp_3 , early_stopping=True,learning_rate_init=0.0001,
       hidden_layer_sizes=(sz_3) , learning_rate='constant', max_iter=100000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=5, shuffle=False, solver=slv_3, tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)
mlp_gloabal_Recov.fit(X_train_Recovered , Y_train_Recovered.ravel())


predict_global_test_Recovered = mlp_gloabal_Recov.predict(X_test_Recovered)
predict_gloabal_train_Recovered = mlp_gloabal_Recov.predict(X_train_Recovered)

if (flg == -1) :
    v1,v2,v3,v4,v5=Evaluate_Model(predict_gloabal_train_Recovered,predict_global_test_Recovered,Y_train_Recovered, Y_test_Recovered)
    print(v1,v2,v3,v4,v5)
    FeedIn(v1,3,'Global_Recov',v3,v5,v2,v4,alp_3,sz_3,act_3,slv_3)
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





