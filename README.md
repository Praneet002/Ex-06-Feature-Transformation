# Ex-06-Feature-Transformation

### AIM
To read the given data and perform Feature Transformation process and save the data to a file.

### EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

### ALGORITHM
STEP 1:
Read the given Data

STEP 2:
Clean the Data Set using Data Cleaning Process

STEP 3:
Apply Feature Transformation techniques to all the features of the data set

STEP 4:
Print the transformed features

### PROGRAM:
# NAME: PRANEET S
# REG NO: 212221230078

```
### Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

### Reading CSV File

df=pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Semester 3/19AI403 _Intro to DS/Exp_6/Data_to_Transform.csv")
df
Basic Process
df.head()

df.info()

df.describe()

df.tail()

df.shape

df.columns

df.isnull().sum()

df.duplicated()

### Before Transformation
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

### Log Transformation
df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()


df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

### Reciprocal Transformation

df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

### Square Root Transformation

df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()
Power Transformation
df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()


from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")

df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

### Quantile Transformation

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')

df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
```

### OUTPUT:

### Reading CSV File
![output](1.png)

### Basic Process
## Head
![output](2.png)
## Info
![output](3.png)
## Describe
![output](4.png)
## Tail
![output](5.png)
## Shape
![output](6.png)
## Columns
![output](7.png)
## Null Values
![output](8.png)
## Duplicate Values
![output](9.png)

### Before Transformation
## Highly Positive Skew
![output](10.png)
## Highly Negative Skew
![output](11.png)
## Moderate Positive Skew
![output](12.png)
## Moderate Negative Skew
![output](13.png)

### Log Transformation
## Highly Positive Skew
![output](14.png)
## Moderate Positive Skew
![output](15.png)

### Reciprocal Transformation
## Highly Positive Skew
![output](16.png)

### Square Root Transformation
## Highly Positive Skew
![output](17.png)

### Power Transformation
## Moderate Positive Skew
![output](18.png)
## Moderate Negative Skew
![output](19.png)

### Quantile Transformation
## Moderate Negative Skew
![output](20.png)

### RESULT:
Thus feature transformation is done for the given dataset.

