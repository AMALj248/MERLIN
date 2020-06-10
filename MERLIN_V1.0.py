import datetime
import time
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sks

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
data['ObservationDate']=pd.to_datetime(data['ObservationDate'])
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
result=new.loc[new['ObservationDate'] == max_date ]
print(result.head(10))

#plotting the data from the result dataframe
#making bubble graph for worldwide data
import plotly.express as px
import plotly.graph_objects as go
#confirmed bubble chart
fig = px.scatter( data_frame=result ,x="Country/Region" , y="Confirmed" ,  color = 'Country/Region' , hover_name = 'Country/Region' ,size ="Confirmed" , size_max=80)
#fig.show()

#recovered bubble chart
fig = px.scatter( data_frame=result ,x="Country/Region" , y="Recovered" ,  color = 'Country/Region' , hover_name = 'Country/Region' ,size ="Recovered" , size_max=80)
#fig.show()

#death chart
fig = px.scatter( data_frame=result ,x="Country/Region" , y="Deaths" ,  color = 'Country/Region' , hover_name = 'Country/Region' ,size ="Deaths" , size_max=80)
#fig.show()


#making a stacked bar graph
fig = go.Figure(data = [
    go.Bar( name ="Confirmed",x=result['Country/Region'] , y=result['Confirmed'] ),
    go.Bar(name='Recovered',x=result['Country/Region'] , y=result['Recovered']),
    go.Bar(name='Deaths',x=result['Country/Region'] , y=result['Deaths'])
]
)
fig.update_layout(barmode = 'group',xaxis_tickangle=-45,title_text='Country Wise Data')
#fig.show()

#Pie chart
fig = px.pie(data_frame=result , names = 'Country/Region' ,values='Confirmed', title = 'Country Wise Confirmed' )
#fig.show()
fig = px.pie(data_frame=result , names = 'Country/Region' ,values='Recovered', title = 'Country Wise Recovered' )
#fig.show()
fig = px.pie(data_frame=result , names = 'Country/Region' ,values='Deaths', title = 'Country Wise Deaths' )
#fig.show()
cnames = ['Country/Region' , 'ObservationDate' , 'Confirmed','Deaths','Recovered']
cstm=pd.DataFrame(columns =cnames )
print(cstm.head())
Country_lst=[]
Date_lst=[]
Cnf_list=[]
Dth_list=[]
Recv_list=[]
for i in result.index:
    if result["Country/Region"][i] in ['US', 'India','UK','Brazil','Spain','Italy','Mainland China','France','Germany'] :
        Country_lst.append(result['Country/Region'][i])
        Date_lst.append(result['ObservationDate'][i])
        Cnf_list.append(result['Confirmed'][i])
        Dth_list.append(result['Deaths'][i])
        Recv_list.append(result['Recovered'][i])

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