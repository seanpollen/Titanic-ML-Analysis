import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import style

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_cols = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
test_cols = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

train_df.columns = train_cols
test_df.columns = test_cols

# Categorical: Survived, Sex, Embarked, Pclass
# Numerical: Age, Fare, SibSp, Parch
# Not Important: Cabin, PassengerId, Ticket
drop_cols1 = ['PassengerId', 'Ticket', 'Cabin']
drop_cols2 = ['Ticket', 'Cabin']
train_df = train_df.drop(drop_cols1, axis=1)
test_df = test_df.drop(drop_cols2, axis=1)

print(train_df.info())
# Age contains 100+ NULL
# Embarked contains 2 NULL

print(train_df.describe())
print(train_df.describe(include=['O']))
# 38% of people survived,
# Average passenger was 29 y/o
# more men than women
# S most common location of departure

# Check correlation between class and survival
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'],
                                               as_index=False).mean().sort_values(by='Survived', ascending=False))
print()
# Check correlation between sex and survival
print(train_df[['Sex', 'Survived']].groupby(['Sex'],
                                            as_index=False).mean().sort_values(by='Survived', ascending=False))
print()
# Check correlation between siblings/spouses and survival
print(train_df[['SibSp', 'Survived']].groupby(['SibSp'],
                                              as_index=False).mean().sort_values(by='Survived', ascending=False))
print()
# Check correlation between children/parents
print(train_df[['Parch', 'Survived']].groupby(['Parch'],
                                              as_index=False).mean().sort_values(by='Survived', ascending=False))
# plot correlation between survival and age for each class
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
plt.xlabel("Age")
plt.ylabel("Casualties")
plt.title("Age, Class, and Death")
plt.show()

# plot correlation between place of departure and mortality rate
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
plt.show()

# create new category "Title" to store prefix
combine = [train_df, test_df]
for data in combine:
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

print(pd.crosstab(train_df['Title'], train_df['Sex']))
# there are a bunch of wack prefixes (reverend?), replace them with Rare to prevent overfitting

for data in combine:
    data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

# check correlation of title and chance of survival
print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

# map title to numerical values
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# drop name
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

# replace Sex with a numerical mapping
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

# we will guess the age of everyone that has a null age value
guess_ages = np.zeros((2,3))

for dataset in combine:
    for i in range(0, 2):  # male or female
        for j in range(0, 3): # upper middle lower class
            guess_df = dataset[(dataset['Sex'] == i) &
                               (dataset['Pclass'] == j + 1)]['Age'].dropna()

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1),
                        'Age'] = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)

# create ordinal values for age
for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4
train_df.head()

combine = [train_df, test_df]

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived',
                                                                                                ascending=False)

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

freq_port = train_df.Embarked.dropna().mode()[0]

# fill null values with mode
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# correlation of survival for each port
print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                                  ascending=False))

# convert to numerical
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# fill null values in fare
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

# create fare band and print correlation between fare and survival
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand',
                                                                                                  ascending=True))
# convert to ordinal value
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

X_train = train_df.drop("Survived", axis=1).copy()
X_test = train_df.drop("Survived", axis=1).copy()
Y_train = train_df["Survived"]

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print(acc_log)

# Analyze predictions on factors
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

print(coeff_df.sort_values(by='Correlation', ascending=False))
plt.close()
plt.figure(figsize=(18, 8))
plt.title("Would You Survive the Titanic?")
plt.xlabel("Variable")
plt.ylabel("Correlation with Survival")
ax = sns.barplot(data=coeff_df, x='Feature', y='Correlation', order=coeff_df.sort_values('Correlation',
                                                                                         ascending=False).Feature)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.1f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                size=15,
                xytext=(0, -12),
                textcoords='offset points')
plt.show()


# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

# K Nearest Neighbours
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

 # generate dataframe of alg effectiveness
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'SGD', 'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_linear_svc, acc_decision_tree]})
print(models.sort_values(by='Score', ascending=False))
plt.close()

# plot model chart
plt.figure(figsize=(18, 5))
plt.title("Machine Learning Algorithm Effectiveness: Titanic")
plt.xlabel("Algorithm")
plt.ylabel("Accuracy (%)")
ax = sns.barplot(data=models, x='Model', y='Score', order=models.sort_values('Score', ascending=False).Model)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.1f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                size=15,
                xytext=(0, -12),
                textcoords='offset points')
plt.show()

X_test = test_df.drop('PassengerId', axis=1).copy()
# choose an algorithm to create a csv of guesses
# Y_pred = logreg.predict(X_test)
# Y_pred = svc.predict(X_test)
# Y_pred = gaussian.predict(X_test)
# Y_pred = perceptron.predict(X_test)
# Y_pred = linear_svc.predict(X_test)
# Y_pred = knn.predict(X_test)
# Y_pred = sgd.predict(X_test)
# Y_pred = decision_tree.predict(X_test)
# Y_pred = random_forest.predict(X_test)

guess = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": Y_pred
})

guess.to_csv('guess.csv', index=False)
