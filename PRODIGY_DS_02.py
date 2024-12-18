# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#------------------------------------------------------------------

# Load the Titanic dataset
titanic_data = pd.read_csv(r"C:\Users\fatem\Desktop\Prodigy\Task 02\titanic\train.csv")

# Display dataset information
print("Dataset Information:")
print(titanic_data.info())

# Display column-wise missing values
missing_values = titanic_data.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Check for duplicates
duplicate_rows = titanic_data.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_rows}\n")

# Display a sample of the data
print("\nSample Data:")
display(titanic_data.head())

#---------------------------------------------------------------------

# Data Cleaning
# Fill missing 'Age' values with the median grouped by 'Pclass' and 'Sex'
titanic_data['Age'] = titanic_data.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

# Drop 'Cabin' column due to excessive missing data
titanic_data.drop('Cabin', axis=1, inplace=True)

# Fill missing 'Embarked' values with the most common embarkation point
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# Display dataset information
print("Dataset Information:")
print(titanic_data.info())

# Display column-wise missing values
missing_values = titanic_data.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Check for duplicates
duplicate_rows = titanic_data.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_rows}\n")

#-----------------------------------------------------------------------

# EDA - Survival Rates by Gender
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.barplot(x='Sex', y='Survived', data=titanic_data, ci=None, palette="pastel")
plt.title('Survival Rates by Gender')
plt.ylabel('Survival Rate')
plt.xlabel('Gender')
plt.show()

# EDA - Survival Rates by Passenger Class
plt.figure(figsize=(8, 5))
sns.barplot(x='Pclass', y='Survived', data=titanic_data, ci=None, palette="pastel")
plt.title('Survival Rates by Passenger Class')
plt.ylabel('Survival Rate')
plt.xlabel('Passenger Class')
plt.show()

# EDA - Survival Rates by Embarkation Point
plt.figure(figsize=(8, 5))
sns.barplot(x='Embarked', y='Survived', data=titanic_data, ci=None, palette="pastel")
plt.title('Survival Rates by Embarkation Point')
plt.ylabel('Survival Rate')
plt.xlabel('Embarkation Point')
plt.show()

# EDA - Age Distribution by Survival
plt.figure(figsize=(8, 5))
sns.histplot(data=titanic_data, x='Age', hue='Survived', kde=True, bins=30, palette="pastel", alpha=0.6)
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['Did not survive', 'Survived'])
plt.show()

# EDA - Fare Distribution by Survival
plt.figure(figsize=(8, 5))
sns.histplot(data=titanic_data, x='Fare', hue='Survived', kde=True, bins=30, palette="pastel", alpha=0.6)
plt.title('Fare Distribution by Survival')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['Did not survive', 'Survived'])
plt.show()
