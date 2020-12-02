# Titanic-ML-Analysis
This python script imports data from 'train.csv', a file containing information for around 900 people on board the titanic.
Using pandas dataframes, the data is analyzed to display the correlation between different factors (age, gender, class, fare, etc) and survival rate.
The script then applies 9 different machine learning algorithms to determine which is the best at predicting casualties.
The file 'test.csv' is incommplete (no casualty information) as is completed by running one of the 9 algorithms (as of right now, Random Forest has the best performance)

Submitting the challenge back to kaggle yielded an accuracy rating of 77%, which is pretty good given the limited size of the data set and prescence of outliers.
Data, and more information regarding this project can be found at https://www.kaggle.com/c/titanic/data
