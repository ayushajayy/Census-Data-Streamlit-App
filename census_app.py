import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import censusdata
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
 
st.set_option('deprecation.showPyplotGlobalUse', False)

background_image = """
<style>
[data-testid="stAppViewContainer"]{
  background-image: url('https://wallpaperaccess.com/full/307193.jpg');  
  background-size: cover;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)

st.image(image='jose-martin-ramirez-carrasco-45sjAjSjArQ-unsplash.jpg')

st.title('Census Data Exploration')

st.header('County Level Summaries')
st.write('This analysis will use data aggregated on the county level for the variables: Gini Index (income inequality index), Vacant Housing, Percent Unemployed and Median Family Income.')


### SECTION 1: Querying and Cleaning Data ###

# Function to query the CensusData API based on the selected year
def ApiQuery(selected_year):
    try:
        df = censusdata.download('acs5', selected_year,
                                censusdata.censusgeo([('state', '*'),('county', '*')]),var=
                                ['B19083_001E','B19113_001E','B23025_003E','B23025_005E','B01001_001E','B25004_001E'])
    except:
        st.error('Could not query data for the selected year.')

    df = df.reset_index()
    df.columns = ['Location','gini_index','median_family_income','employed','unemployed','population',
                'vacant_housing']

    return df
 
selected_year = st.slider('Select a year for Census data', 2011, 2021, 2018)  
df = ApiQuery(selected_year)
 
def FindFipsId(df):
    df.Location = df.Location.astype(str)
    df = df[df['Location'].str.contains("Puerto Rico:")==False]
    state_fips = lambda a: a[a.find('state:')+6:a.find('state:')+8]
    df['state_fips'] = df['Location'].apply(state_fips)
    county_fips = lambda a: a[a.find('county:')+7:a.find('county:')+11]
    df['county_fips'] = df['Location'].apply(county_fips)
    df['fips'] = df.state_fips+df.county_fips
    df.fips = df.fips.astype('int32')
    county_name = lambda a: a[:a.find('County,')+6]
    df['county_name'] = df['Location'].apply(county_name)
    df['state_fips'] = pd.to_numeric(df.state_fips)
    df['county_fips'] = pd.to_numeric(df.county_fips)
    df = df.drop('Location',axis=1)    
    return df

def PctUnemployed(df):
    df['percent_unemployed'] = df.unemployed / df.employed * 100
    return df

def OnlyColumns(df):
    df = df[['gini_index','vacant_housing','percent_unemployed', 'median_family_income']]
    df.columns = ['Gini Index', 'Vacant Housing', 'Percent Unemployed', 'Median Family Income']
    return df
 
df = FindFipsId(df)
df = PctUnemployed(df)
df = OnlyColumns(df)


### SECTION 2: Display the data.

st.subheader(f'Descriptive Statistics for Median Family Income in {selected_year}')
st.write(df['Median Family Income'].describe().round())

# Histogram for median family income
st.subheader('Matplotlib Histogram of Median Family Income')
plt.figure(figsize=(10, 6))
plt.hist(df['Median Family Income'], bins=30, color='skyblue', edgecolor='black')
plt.title('County Level Income Distribution')
plt.xlabel('Median Family Income')
plt.ylabel('Frequency')
plt.grid(True)
st.pyplot()
 
df['Income Quartile'] = pd.qcut(df['Median Family Income'], q=4, labels=False, precision=0, duplicates='raise')
df['Income Quartile'] = df['Income Quartile'] + 1

## Scatterplot for median family income by quartile
st.subheader(f'Matplotlib Scatterplot of Median Family Income by Percent Unemployed in {selected_year}')
plt.figure(figsize=(10, 6))
plt.scatter(df['Median Family Income'], df['Percent Unemployed'], c=df['Income Quartile'], cmap='viridis', s=100)
plt.title(f'Median Family Income versus Unemployment in {selected_year}')
plt.xlabel('Median Family Income')
plt.ylabel('Percent Unemployed')
plt.colorbar(label='Income Quartile')
st.pyplot()

# Checkbox for lmplot analysis
st.subheader(f'Please select an X and Y value for analysis in {selected_year}' )
x_axis = st.selectbox(
    'Pick an X-axis value:',
     ['Gini Index', 'Vacant Housing', 'Percent Unemployed', 'Median Family Income'])
y_axis = st.selectbox(
    'Pick a Y-axis value:',
     ['Percent Unemployed','Gini Index', 'Vacant Housing','Median Family Income'])

st.subheader(f'Matplotlib lmplot analysis by Income Quartile in {selected_year}')
plt.figure(figsize=(12, 8))
for quartile in range(1, 5):
    plt.subplot(2, 2, quartile)
    quartile_data = df[df['Income Quartile'] == quartile]
    plt.scatter(quartile_data[x_axis], quartile_data[y_axis], label=f'Quartile {quartile}', s=100)
    plt.title(f'Income Quartile {quartile}')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
plt.suptitle(f'lmplot analysis by Income Quartile in {selected_year}', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
st.pyplot()

# SECTION 3: Data Insights

st.header('Data Insights')

# Correlation Matrix
st.subheader('Correlation Matrix')
correlation_matrix = df.corr()
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
st.pyplot()

st.subheader('Key Insights')

st.write("1. **Income Quartiles and Gini Index:** There is a negative correlation between income quartiles and the Gini Index, suggesting that higher-income quartiles are associated with lower income inequality.")

st.write("2. **Geographical Patterns:** Explore geographical patterns by using the map. Look for clusters of counties with high/low income, unemployment, or other metrics.")

st.write("3. **Vacant Housing and Median Family Income:** Counties with higher median family income tend to have lower percentages of vacant housing, indicating a potential link between economic prosperity and housing occupancy.")

st.write("4. **Income and Education:** Investigate the relationship between median family income and educational attainment in counties. Are higher-income counties associated with higher levels of education?")

st.write("5. **Outliers:** Identify and explore any outliers in the data. For instance, are there counties with unusually high or low Gini Index values compared to their income quartiles?")

# Regression Analysis
st.header('Regression Analysis')
st.subheader('Predict Gini Index based on Median Family Income')

# Select features for regression
features = ['Median Family Income', 'Percent Unemployed']  
 
df_regression = df.dropna(subset=features + ['Gini Index'])
 
X = df_regression[features]
y = df_regression['Gini Index']
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
model = LinearRegression()
model.fit(X_train, y_train)

 
y_pred = model.predict(X_test)

 
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

 
st.write(f'Mean Absolute Error: {mae:.2f}')
st.write(f'Mean Squared Error: {mse:.2f}')
st.write(f'Root Mean Squared Error: {rmse:.2f}')

# Create a scatter plot with the regression line
plt.figure()
plt.scatter(X_test['Median Family Income'], y_test, color='black', label='Actual')
plt.scatter(X_test['Median Family Income'], y_pred, color='blue', label='Predicted')
plt.xlabel('Median Family Income')
plt.ylabel('Gini Index')
plt.legend()
st.pyplot()
   
# Logistic Regression

df['HighIncome'] = (df['Median Family Income'] > df['Median Family Income'].median()).astype(int)
 
features_logistic = ['Median Family Income', 'Percent Unemployed']
 
df_logistic = df.dropna(subset=features_logistic + ['HighIncome'])

 
X_logistic = df_logistic[features_logistic]
y_logistic = df_logistic['HighIncome']

 
X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(
    X_logistic, y_logistic, test_size=0.2, random_state=42
)

# Create a pipeline with imputation and logistic regression
pipeline_logistic = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('logistic_regression', LogisticRegression())
])
 
pipeline_logistic.fit(X_train_logistic, y_train_logistic)

# Make predictions on the test set
y_pred_logistic = pipeline_logistic.predict(X_test_logistic)

# Evaluate the logistic regression model
accuracy_logistic = accuracy_score(y_test_logistic, y_pred_logistic)
precision_logistic = precision_score(y_test_logistic, y_pred_logistic)
recall_logistic = recall_score(y_test_logistic, y_pred_logistic)
f1_logistic = f1_score(y_test_logistic, y_pred_logistic)

# Display classification metrics for logistic regression
st.subheader('Logistic Regression Metrics:')
st.write(f'Accuracy: {accuracy_logistic:.2f}')
st.write(f'Precision: {precision_logistic:.2f}')
st.write(f'Recall: {recall_logistic:.2f}')
st.write(f'F1 Score: {f1_logistic:.2f}')

# Create a confusion matrix for logistic regression
conf_matrix_logistic = confusion_matrix(y_test_logistic, y_pred_logistic)
st.subheader('Confusion Matrix (Logistic Regression)')
st.write(conf_matrix_logistic)

# Visualize the distribution of the binary target variable
st.subheader('Distribution of HighIncome')
st.bar_chart(df['HighIncome'].value_counts())

df.to_csv('output_file.csv', index=False)
