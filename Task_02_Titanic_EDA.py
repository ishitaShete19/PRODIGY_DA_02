
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load dataset
print("Loading dataset...")
df = pd.read_csv('train.csv')
print("First 5 rows:")
print(df.head())

# Basic info
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Drop irrelevant columns
df.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Convert categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Check after cleaning
print("\nCleaned Data Info:")
print(df.info())

# Create 'outputs' folder for plots
os.makedirs("outputs", exist_ok=True)

# EDA Visualizations
print("\nGenerating EDA plots...")

# Survival Rate by Sex
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Sex')
plt.savefig('outputs/survival_by_sex.png')
plt.clf()

# Survival by Passenger Class
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.savefig('outputs/survival_by_class.png')
plt.clf()

# Age Distribution
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution of Passengers')
plt.savefig('outputs/age_distribution.png')
plt.clf()

# Correlation Heatmap
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.savefig('outputs/correlation_matrix.png')
plt.clf()

# Summary Insights
print("\nKey Insights:")
print("- Females had a significantly higher survival rate than males.")
print("- Passengers in 1st class were more likely to survive.")
print("- Younger passengers had slightly better survival chances.")
print("- Strong correlations observed between Sex, Pclass, and Survived.")

print("\nAll plots saved in the 'outputs' directory.")
