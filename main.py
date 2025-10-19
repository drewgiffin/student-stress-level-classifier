import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data_path = "student_lifestyle_dataset.csv"

def main():
    # loading
    df = load_data()
    
    # preprocessing
    df_clean = preprocess_data(df)
    
    # exploratory data analysis
    # draw_plots(df_clean)
    
    # feature engineering
    normalize_features(df_clean)
    
    # separate features and target
    X = df_clean.drop('Stress_Level', axis=1)
    y_raw = df_clean['Stress_Level']
        
    # encode target
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # split into train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )   
    
    # sanity check
    print("Classes:", le.classes_)
    print("y_train distribution:", pd.Series(y_train).value_counts(normalize=True))
    print("y_test distribution:", pd.Series(y_test).value_counts(normalize=True))
    print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)
    
    
def load_data():
    df = pd.read_csv(data_path, encoding="ascii", delimiter=",")
    #removing uneeded feature
    df.drop("Student_ID", axis=1, inplace=True)
    return df

def inspect_data(df):
    print("Info:")
    print(df.info())
    print("\n")
    
    print("Head:")
    print(df.head())
    print("\n")
    
    print("Description:")
    print(df.describe(include="all"))
    print("\n")

def clean_data(df):
    # print("Missing values:")
    # print(df.isnull().sum())
    # print("\n")
    
    df.dropna(inplace=False)
    return df

def order_data_stress_level(df):
    df["Stress_Level"] = pd.Categorical(
        df["Stress_Level"],
        categories=["Low", "Moderate", "High"],
        ordered=True
    )

def display_feature_distributions_histogram(df):
    df.hist(bins=20, figsize=(10,8))
    plt.suptitle("Feature Distributions")
    plt.show()
    
def display_scatter_plot_matrix(df):
    sns.pairplot(df, hue="Stress_Level")
    plt.suptitle("Pair Plot of Numerical Features", y=1.02)
    plt.show()
    
def display_correlation_heatmap(df):
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()
    
def display_feature_boxplots(df):
    for col in df.select_dtypes(include=[np.number]).columns:
        sns.boxplot(x="Stress_Level", y=col, data=df)
        plt.title(f"{col} by Stress Level")
        plt.show()

def draw_plots(df):
    display_feature_distributions_histogram(df)
    display_scatter_plot_matrix(df)
    display_correlation_heatmap(df)
    display_feature_boxplots(df)

def preprocess_data(df):
    df_clean = clean_data(df)
    order_data_stress_level(df_clean)
    return df_clean

def normalize_features(df):
    scaler = MinMaxScaler()
    df[["Study_Hours_Per_Day"]] = scaler.fit_transform(df[["Study_Hours_Per_Day"]])
    df[["Extracurricular_Hours_Per_Day"]] = scaler.fit_transform(df[["Extracurricular_Hours_Per_Day"]])
    df[["Sleep_Hours_Per_Day"]] = scaler.fit_transform(df[["Sleep_Hours_Per_Day"]])
    df[["Social_Hours_Per_Day"]] = scaler.fit_transform(df[["Social_Hours_Per_Day"]])
    df[["Physical_Activity_Hours_Per_Day"]] = scaler.fit_transform(df[["Physical_Activity_Hours_Per_Day"]])
    df[["GPA"]] = scaler.fit_transform(df[["GPA"]])

main()