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
data = pd.read_csv('covid_19_data 16_May.csv')
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
import plotly.express as px
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

#making a bubble chart for better data understanding
#confirmed bubble chart
fig = px.scatter( data_frame=plt1 ,x="Country/Region" , y="Confirmed" ,  color = 'Country/Region' , hover_name = 'Country/Region' ,size ="Confirmed" , size_max=80)
fig.show()

#recovered bubble chart
fig = px.scatter( data_frame=plt1 ,x="Country/Region" , y="Recovered" ,  color = 'Country/Region' , hover_name = 'Country/Region' ,size ="Recovered" , size_max=80)
fig.show()

#death chart
fig = px.scatter( data_frame=plt1 ,x="Country/Region" , y="Deaths" ,  color = 'Country/Region' , hover_name = 'Country/Region' ,size ="Deaths" , size_max=80)
fig.show()

#Now we create a user-input function
#In which the user defines the country and it show the plot of country

print(new.head(20))
name=input('Please Enter The COUNTRY Name = ')
print('The entered Country was = ' , name)
user=new.loc[new['Country/Region'] == name ]
print("Head for user \n" , user.head(20))

#Now plotting the Country history for user defined Country
#converting the datatype of x for plotting ease
plt_data=user
pd.to_datetime(plt_data['ObservationDate'])
print("NEW =\n" , plt_data.head())

sns.relplot(x='ObservationDate',y='Confirmed',kind='line',data=plt_data)
plt.setp(plt.xticks()[1], rotation=30, ha='right') # ha is the same as horizontalalignment
plt.show()

#passing X, Y inputs
#before passing we need to create a new column day counter which starts from minimum date in the dateset
#this is now on the user data where it dosent work
#it cant fin .dt.days function

#flase positive warning
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
print("user observation dates \n" , user['ObservationDate'])
pd.to_datetime(user['ObservationDate'])
user['Days'] = user['ObservationDate']
print("USER HEAD \n" ,user.head(10))
min_date = user['Days'].min()
print("Minimum Date in USER =" , max_date)
print("Minimum Date in USER =" , min_date)

#creating the days column
columnSeriesObj = []
for i in user['ObservationDate'] :

    t=i-min_date
    columnSeriesObj.append(t)
user['Days'] = columnSeriesObj
print("Column Overwrite\n " , user['Days'].head(10))
print("Datatype = " , type(user['Days']))
user['Days'] = user['Days'].dt.days
print("DAYS NOW \n" ,user['Days'])

#model input parameters

x= user['Days']
y=user['Confirmed']
print("Before Reshape \n" , x )
x= x.values.reshape(-1,1)
print("X after reshape \n" , x )


#now we try to predict the confirmed cases in the data using linear regression

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x  , y , test_size=0.1 , random_state=0 , shuffle=False)
#shuffle has to be pased as False or else the x values will be randomly selected
#linear regression does not work for date type data , we need to convert it to numerical data


#passing model inputs and trainng the model
from sklearn.linear_model import LinearRegression
ln_rg=LinearRegression()
ln_rg.fit(x_train , y_train)

#predicting test set values with test set
ln_rg_pred = ln_rg.predict(x_test)
y_pred = ln_rg_pred

# #plotting the scatter plot  between y_actual and y_predicited
# plt.scatter(y_test, y_pred , c='green')
# plt.xlabel;("Actual  Confirmed")
# plt.ylabel("predicted Confirmed")
# plt.title(" True vs Predicted value : Linear Regression ")
#plt.show()

#R2 error

r2_score1 = sks.metrics.r2_score(y_test , y_pred )
print("R2 Score for the Linear Regression  model is : " , r2_score1)

#creating a polynomial fit model
#first creating a polynomial feature
from sklearn.preprocessing import PolynomialFeatures
poly_feature = PolynomialFeatures(degree=3)

x_train_poly = poly_feature.fit_transform(x_train)

#fitting the model
poly_model = LinearRegression()
poly_model.fit(x_train_poly, y_train)

#prediciting the model on test data
y_poly_pred = poly_model.predict(poly_feature.fit_transform(x_test))

#R2 error

r2_score2 = sks.metrics.r2_score(y_test , y_poly_pred )
print("R2 Score for Polynomial Model of 5Th degree is : " , r2_score2)
#predicting the number of confirmed cases for the user input country
#we will compare the prediciton from both linear regression and ploynomial fit

x_days_max =user['Days'].max()
print("max data = " ,x_days_max)
pred_dates=[]
for x in range(61) :
    q= x+x_days_max
    pred_dates.append(q)

#print("FUTURE DATES =\n" , pred_dates)
#we have to convert list to numpy array to reshape it
pred_dates_array= np.array(pred_dates)
future_dates= pred_dates_array.reshape(-1,1)
print("FUTURE DATES AFTER RESHAPE  =\n" , future_dates)

#POLYNOMIAL MODEL FUTURE PREDICTION
bf = poly_model.predict(poly_feature.fit_transform(future_dates))
poly_fut_pred=bf.astype(int)
print("Y POLYNIMIAL PREDICTION FOR THE FUTURE \n" , poly_fut_pred)

#LINEAR MODEL FUTURE PREDICTION
cf=ln_rg.predict(future_dates)
ln_fut_pred=cf.astype(int)
print("Y LINEARA REGRESION PREDICTION FOR THE FUTURE \n" , ln_fut_pred)
print(pred_dates)

#combining the two lists into a single dataframe

final_plot_linear=pd.DataFrame(
    {'Days' : pred_dates,
    'Confirmed' :ln_fut_pred
     }
)
print("Final DataFrame = \n", final_plot_linear.head())

#drawing the final plot for the next 20 days from th polynomial model
import plotly.express as px
fig= px.line( final_plot_linear , x='Days' , y='Confirmed')
#fig.show()
print("Tail\n",final_plot_linear.tail())
print("Last Date = ", user['ObservationDate'].max())
print("First Date = ", user['ObservationDate'].min())

#updating the dataframe to include dates for easy understanding

user_date_max = user['ObservationDate'].max()
Obs_date_fut = []
m=user_date_max
for i in range(61) :
    m+=timedelta(days=1)
    Obs_date_fut.append(m)
print(Obs_date_fut)
final_plot_linear['Dates'] = Obs_date_fut
print("final_plot = \n" , final_plot_linear.head())

#plotting with the new dates column
import plotly.express as px
fig= px.line( final_plot_linear , x='Dates' , y='Confirmed')
fig.update_layout(title_text = ("THE CONFIRMED CASES BY LINEAR  MODEL"))
#fig.show()


#now for polynomial prediction

#combining the two lists into a single dataframe

final_plot_poly=pd.DataFrame(
    {'Days' : pred_dates,
    'Confirmed' :poly_fut_pred
     }
)
print("Final DataFrame = \n", final_plot_poly.head())
user_date_max = user['ObservationDate'].max()
Obs_date_fut = []
m=user_date_max
for i in range(61) :
    m+=timedelta(days=1)
    Obs_date_fut.append(m)
print(Obs_date_fut)
final_plot_poly['Dates'] = Obs_date_fut
print("final_plot = \n" , final_plot_poly.head())

#plotting with the new dates column
import plotly.express as px
fig= px.line( final_plot_poly , x='Dates' , y='Confirmed')
fig.update_layout(title_text = ("THE CONFIRMED CASES BY POLYNOMIAL MODEL 29 th APRIL"))
#fig.show()

#modelling the model with the scatter points
# Visualising the Polynomila Regression results
plt.scatter(x_test, y_test, color='blue')

plt.plot(x_test, poly_model.predict(poly_feature.fit_transform(x_test)), color='red')
plt.title('Polynomial  Regression')
plt.xlabel('Dates')
plt.ylabel('Confirmed')

plt.show()

# Visualising the Polynomila Regression results
plt.scatter(x_test, y_test, color='blue')

plt.plot(x_test, ln_rg.predict(x_test), color='red')
plt.title('Linear  Regression')
plt.xlabel('Dates')
plt.ylabel('Confirmed')

plt.show()


#now there are two problems
#1 polynomial model is overfitted
#2 linear regression model has to be used but has low accuraccy
#So we use enembling Metod for better predictions and low error

#Average Ensembling
# finalpred = (ln_rg_pred+y_poly_pred)/2
#
# #the r score for simple average
# from scipy.optimize import minimize
# r2_score3 = sks.metrics.r2_score(y_test , finalpred )
# print("R2 Score for The Average Model is : " , r2_score3)
# #finalpred1=[]
# #Weighhted Aveage Ensembling
# # def opt(x) :
# #     x1=x[0]
# #     x2=x[1]
#
#finalpred1 = (x1*ln_rg_pred+x2*y_poly_pred)
# #     r2_score4 = sks.metrics.r2_score(y_test, finalpred1)
# #     score = 1-r2_score4
# #     return score
# # x0 =(1,1)
# # sol = minimize(opt  , x0 , method = 'Nelder-Mead')
# # print(sol)
# # print(sol.x)
# #r2_score4 = sks.metrics.r2_score(y_test , finalpred1 )
# #print("R2 Score for The Average Model is : " , r2_score4)
#
# finalpred1 = (-0.00870389*ln_rg_pred+1.0145018*y_poly_pred)
# r2_score4 = sks.metrics.r2_score(y_test , finalpred1 )
# print("R2 Score for The Weighted  Average Model is : " , r2_score4)
# #single array has to be converted to int
# print(finalpred1)
#
#
# #astype() needs a mew variable name
# f2=finalpred1.astype(int)
# print(f2)
#
# #creating the weighted average model for the next 21 days
# print("The dates = \n" ,future_dates)
# finalpred_actual = ( -0.00870389*ln_rg.predict(future_dates)
#                      +
#                      1.0145018*poly_model.predict(poly_feature.fit_transform(future_dates)))
#
# print(finalpred_actual.astype(int))
#
# #fitting it to the graphing dataframe to plot it
# final_plot['Avg_Model'] = finalpred_actual
#
# fig= px.line( final_plot , x='Dates' , y='Avg_Model')
# fig.update_layout(title_text = ("THE CONFIRMED CASES VIA WEIGHTED AVERAGE MODEL"))
# fig.show()

#creating a logistic regression model
# from sklearn.linear_model import LogisticRegression
# logisticRgr = LogisticRegression()
#
# logisticRgr.fit(x_train , y_train)
# pred_values1=logisticRgr.predict(x_test)
#
# # # Use score method to get accuracy of model
# scor = sks.metrics.r2_score(y_test , pred_values1 )
# print("LOGISTIC REGRESSION SCORE =" , scor)

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


X = global_data.iloc[:,-1].values
Y=global_data.iloc[:,-4].values

#print("Before Reshape x \n" , x )
#print("Before Reshape y \n" , y )
#X= X.reshape(-1,1)
#Y=Y.reshape(-1,1)


#modelling a linea regression model for before the inflection point

#fitting the logistic growth curve for confirmed cases

