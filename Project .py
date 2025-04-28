import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load the dataset
df = pd.read_csv("C:\\Users\\murth\\OneDrive\\Desktop\\Python Toolbox Project\\data.csv")

# Set base theme
sns.set_theme(style="whitegrid")

# Data preprocessing
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
df['Stay Length'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

# 1. Display basic info
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# 2. Bar Plot - Blood Type
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Blood Type', hue='Blood Type', palette='Set1', legend=False)
plt.title("Blood Type Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Blood Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 3. Medical Conditions
plt.figure(figsize=(8, 5))
sns.countplot(data=df, y='Medical Condition', hue='Medical Condition',
              order=df['Medical Condition'].value_counts().index,
              palette='Dark2', legend=False)
plt.title("Most Common Medical Conditions", fontsize=14)
plt.tight_layout()
plt.show()

# 4. Line graph - Avg billing per year
df['Year'] = df['Date of Admission'].dt.year
yearly_avg = df.groupby('Year')['Billing Amount'].mean().reset_index()
plt.figure(figsize=(8, 5))
sns.lineplot(data=yearly_avg, x='Year', y='Billing Amount', marker='o', color='teal')
plt.title("Avg Billing Amount Over Years")
plt.tight_layout()
plt.show()

# 5. Billing Amount by Admission Type
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Admission Type', y='Billing Amount',
            hue='Admission Type', palette='coolwarm', legend=False)
plt.title("Billing Amount by Admission Type", fontsize=14)
plt.tight_layout()
plt.show()

# 6. FacetGrid - Age by Gender
g = sns.FacetGrid(df, col="Gender", height=5, aspect=1.2)
g.map_dataframe(sns.histplot, x="Age", kde=True, hue="Gender", multiple="stack",
                palette="Set3", edgecolor="black")
g.set_axis_labels("Age", "Frequency")
g.fig.suptitle("Age Distribution by Gender", fontsize=16, fontweight='bold')
g.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()

# 7. Encoded Categorical Correlation Matrix
df['Gender_Code'] = df['Gender'].astype('category').cat.codes
df['Admission Type_Code'] = df['Admission Type'].astype('category').cat.codes
plt.figure(figsize=(8, 6))
sns.heatmap(df[['Age', 'Billing Amount', 'Stay Length', 'Gender_Code', 'Admission Type_Code']].corr(),
            annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix (Encoded)", fontsize=14)
plt.tight_layout()
plt.show()

# 8. Pairplot of Numerical Features
sns.pairplot(df[['Age', 'Billing Amount', 'Room Number']], diag_kind='hist',
             plot_kws={'color': 'indigo'})
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()

# 9. Pairplot with Hue (Gender)
sns.pairplot(df[['Age', 'Billing Amount', 'Room Number', 'Gender']],
             hue='Gender', palette='cool')
plt.show()

# 10. Pie Chart - Blood Type Distribution
blood_counts = df['Blood Type'].value_counts() 
plt.figure(figsize=(7, 7))
plt.pie(blood_counts, 
        labels=blood_counts.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=sns.color_palette('Paired'))
plt.title("Patient Distribution by Blood Group")
plt.axis('equal')
plt.show()

# 11. T-test for Billing Amount by Gender
male_bills = df[df['Gender'] == 'Male']['Billing Amount']
female_bills = df[df['Gender'] == 'Female']['Billing Amount']
t_stat, p_val = stats.ttest_ind(male_bills, female_bills, equal_var=False)
print(f"\nT-test result: t = {t_stat:.2f}, p = {p_val:.4f}")
