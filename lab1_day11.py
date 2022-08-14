# To ignore warnings
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


boston = datasets.load_boston()
X_boston,y_boston = boston.data, boston.target
print('Shape of data:', X_boston.shape, y_boston.shape)
print('Keys:', boston.keys())
print('Feature names:',boston.feature_names)
#Q1: Create a dataframe and Save that dataset inside it.
df=pd.DataFrame(X_boston,columns=['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT'])
df['Price']=y_boston

#Q2: Print the head rows of the dataframe.
print(df.head())

#Q3: Use histogram to show the distribution of House Prices.
plt.hist(df['Price'])
plt.show()

#Q4: Use a heatmap to show the correlation between features and the target labels.
sns.heatmap(df.corr(),annot=True)
plt.show()

#Q5: Use a lmplot to draw the relations between price and LSTAT.
sns.lmplot(data=df,x='Price',y='LSTAT')
plt.show()

#Q6: Use a lmplot to draw the relations between price and RM.
sns.lmplot(data=df,x='Price',y='RM')
plt.show()
print(df.columns)

#Q7: Split the dataset into Train and Test sets with test_size=30% and random_state=23.
x_tr,x_ts,y_tr,y_ts=train_test_split(X_boston,y_boston,test_size=0.3,random_state=23)
print(x_tr.shape,x_ts.shape,y_tr.shape,y_ts.shape)
#Q8: Build a Linear Regression Model.
model=LinearRegression()
model.fit(x_tr,y_tr)
#Q10: Evaluate the model.

print(model.intercept_)
X_boston=pd.DataFrame(X_boston,columns=['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT'])
coeff_df = pd.DataFrame(model.coef_,X_boston.columns,columns=['Coefficient'])
print(coeff_df)
y_pred=model.predict(x_ts)
Real_Values = np.array(y_ts)
print(y_pred,'\n',Real_Values)

plt.scatter(Real_Values,y_pred)
plt.show()

sns.distplot((y_ts-y_pred))
plt.show()
#Q11: Use evaluation metrics MAE, MSE, RMSE and R^2.

print('MAE:', metrics.mean_absolute_error(y_ts, y_pred))
print('MSE:', metrics.mean_squared_error(y_ts, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_ts, y_pred)))
print('R^2:', metrics.r2_score(y_ts, y_pred))
