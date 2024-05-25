#importing libraries
import numpy as np
import pandas as pd

#importing zomato missing values dataset
df=pd.read_csv('zomato_dataset.csv')

#Remove the missing values from the data.
#replacing the unrecognised values with null values
df.replace("?",np.nan,inplace=True)
df.isnull().sum()

#Choose the right Data Imputation (replacement strategy) for the missing data.
#checking data types of columns
df.dtypes

#converting columns country code , avg cost for two , price range ,votes ,longitude, latitude and aggregate rating to numeric data type and applying knn imputation technique
#using backward fill technique to impute data in the columns city, address, locality and locality verbose
#using forward fill technique to impute data in the columns rating color, rating text, restraunt name, cuisines and currency
#converting columns to numeric data type
df["Country Code"]=pd.to_numeric(df["Country Code"])
df["Longitude"]=pd.to_numeric(df["Longitude"])
df["Latitude"]=pd.to_numeric(df["Latitude"])
df["Average Cost for two"]=pd.to_numeric(df["Average Cost for two"])
df["Price range"]=pd.to_numeric(df["Price range"])
df["Aggregate rating"]=pd.to_numeric(df["Aggregate rating"])
df["Votes"]=pd.to_numeric(df["Votes"])

#using forward fill technique to impute data in the columns rating color, rating text, restraunt name, cuisines and currency
df['Rating color'].fillna(method='ffill',inplace=True)
df['Rating text'].fillna(method='ffill',inplace=True)
df['Restaurant Name'].fillna(method='ffill',inplace=True)
df['Cuisines'].fillna(method='ffill',inplace=True)
df['Currency'].fillna(method='ffill',inplace=True)

#using backward fill technique to impute data in the columns city, address, locality and locality verbose
df['City'].fillna(method='bfill',inplace=True)
df['Address'].fillna(method='bfill',inplace=True)
df['Locality'].fillna(method='bfill',inplace=True)
df['Locality Verbose'].fillna(method='bfill',inplace=True)
df.isna().sum()

#Use KNN-imputer to impute missing values in data.
#creating a dataframe of numeric data
Before_imputation_df=pd.DataFrame(data=[],columns=['Country Code','Longitude','Latitude','Average Cost for two','Price range','Aggregate rating','Votes'])
Before_imputation_df["Country Code"]=df["Country Code"]
Before_imputation_df["Longitude"]=df["Longitude"]
Before_imputation_df["Latitude"]=df["Latitude"]
Before_imputation_df["Average Cost for two"]=df["Average Cost for two"]
Before_imputation_df["Price range"]=df["Price range"]
Before_imputation_df["Aggregate rating"]=df["Aggregate rating"]
Before_imputation_df["Votes"]=df["Votes"]
Before_imputation_df

#importing the knn imputer
from sklearn.impute import KNNImputer

#creating an imputation object
imputer=KNNImputer(n_neighbors=2)

#performing KNN Imputation on dataset
After_imputation=imputer.fit_transform(Before_imputation_df)
After_imputation_df=pd.DataFrame(After_imputation,columns=Before_imputation_df.columns)
After_imputation_df

df["Country Code"]=After_imputation_df["Country Code"]
df["Longitude"]=After_imputation_df["Longitude"]
df["Latitude"]=After_imputation_df["Latitude"]
df["Average Cost for two"]=After_imputation_df["Average Cost for two"]
df["Price range"]=After_imputation_df["Price range"]
df["Aggregate rating"]=After_imputation_df["Aggregate rating"]
df["Votes"]=After_imputation_df["Votes"]
#checking for null values after imputation
df.isna().sum()

#Perform the data normalization using (a) min-max scaler technique (b) z-score normalization technique.

#performing scaling techinques on After_imputation_df as it has only numeric data types

#importing the min_max_scaler technique
from sklearn.preprocessing import MinMaxScaler

#creating a copy of the dataframe
Before_minmax_scale_df=After_imputation_df

#creating an normalization object
min_max_scaler=MinMaxScaler()                   #y=(x-min)/(max-min)
After_minmax_scale = min_max_scaler.fit_transform(Before_minmax_scale_df)
After_minmax_scale_df=pd.DataFrame(After_minmax_scale,columns=Before_minmax_scale_df.columns)
After_minmax_scale_df

#importing the z score technique
from sklearn.preprocessing import StandardScaler
#creating a copy of the dataframe
Before_zscore_scale_df=After_imputation_df

#creating an normalization object
zscore_scaler=StandardScaler()
After_zscore_scale = zscore_scaler.fit_transform(Before_zscore_scale_df)
After_zscore_scale_df=pd.DataFrame(After_zscore_scale,columns=Before_zscore_scale_df.columns)

After_zscore_scale_df
