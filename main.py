import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


data_path = "student_lifestyle_dataset.csv"

def main():
    # loading
    df = load_data()
    
    # preprocessing
    df_clean = preprocess_data(df)
    
    # exploratory data analysis
    # draw_plots(df_clean)
    
    # separate features and target
    le = LabelEncoder()
    X, y = separate_features_and_target(df_clean, le)
    
    # split into train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )   
    
    # feature engineering
    X_train_normalized, X_test_normalized = normalize_features(X_train, X_test)
    
    # training
    model = train_logistic_regression(X_train_normalized, y_train)
    
    # prediction
    y_pred = predict_target(model, X_test_normalized)

    # evaluation
    evaluate_model(model, X, y_pred, y_test, le)
    
    draw_confusion_matrix(y_test, y_pred, le)

def draw_confusion_matrix(y_test, y_pred, le):
    y_test_decoded = le.inverse_transform(y_test)
    y_pred_decoded = le.inverse_transform(y_pred)

    cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=le.classes_)

    # Plot
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("images/confusion_matrix.png", dpi=300)  # Save for README
    plt.show()

def predict_target(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def separate_features_and_target(df, le):
    X = df.drop('Stress_Level', axis=1)
    y_raw = df['Stress_Level']
    # encode target
    y = le.fit_transform(y_raw)
    return X, y

def evaluate_model(model, X, y_pred, y_test, le):
    feature_names = X.columns
    
    # Evaluate
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0]
    })
    print(feature_importance.sort_values(by='Coefficient', ascending=False))

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        solver='lbfgs',
        max_iter=10000
    )
    model.fit(X_train, y_train)
    return model  
 
def load_data():
    df = pd.read_csv(data_path, encoding="ascii", delimiter=",")
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
    print("Missing values:")
    print(df.isnull().sum())
    print("\n")
    
    print("Duplicate rows in dataset:")
    print(df.duplicated().sum())
    print("\n")
    
    df.dropna(inplace=True)
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
    #removing uneeded feature
    df.drop("Student_ID", axis=1, inplace=True)
    df_clean = clean_data(df)
    order_data_stress_level(df_clean)
    return df_clean

def normalize_features(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit only on training data
    X_test_scaled = scaler.transform(X_test)    
    return X_train_scaled, X_test_scaled

main()