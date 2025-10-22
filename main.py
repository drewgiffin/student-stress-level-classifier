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
    X, y = separate_features_and_target(df_clean)
    
    # split into train and test data
    accuracy_scores = []
    # run training many times using different splits to get an average accuracy score 
    for i in range(1000):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=i
        )
    
        # pre training processing
        X_train_normalized, X_test_normalized = normalize_features(X_train, X_test)
        
        # training
        model = train_logistic_regression(X_train_normalized, y_train)
        
        # prediction
        y_pred = predict_target(model, X_test_normalized)

        # evaluation
        le = get_label_encoder(df_clean)
        # draw_feature_importance(model, X)
        # draw_confusion_matrix(y_test, y_pred, le)
        # draw_classification_report(y_test, y_pred, le)
        accuracy = get_accuracy(y_test, y_pred)
        accuracy_scores.append(accuracy)
    print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")
    print(f"Samples: {len(accuracy_scores)}")

def get_accuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def get_label_encoder(df):
    le = LabelEncoder()
    le.classes_ = np.array(df['Stress_Level'].cat.categories)
    return le

def draw_classification_report(y_test, y_pred, le):
    report = classification_report(
        y_test, y_pred, output_dict=True, target_names=le.classes_
    )
    df_report = pd.DataFrame(report).transpose()
    
    metrics_df = df_report.loc[le.classes_, ["precision", "recall", "f1-score"]]
    
    ax = metrics_df.plot(
        kind="bar",
        figsize=(8, 5),
        rot=0,
        color=["#4C72B0", "#55A868", "#C44E52"]
    )
    
    plt.title("Classification Report Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height:.2f}",
            (p.get_x() + p.get_width() / 2, height),
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    plt.tight_layout()
    plt.show()
    
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
    plt.show()

def predict_target(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def separate_features_and_target(df): 
    X = df.drop('Stress_Level', axis=1)
    y = df['Stress_Level'].cat.codes
    return X, y

def draw_feature_importance(model, X):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': -model.coef_[0]
    })

    feature_importance['abs_coef'] = feature_importance['Coefficient'].abs()
    feature_importance = feature_importance.sort_values(by='abs_coef', ascending=False) 
    feature_importance = feature_importance.iloc[::-1]

    colors = ['green' if c > 0 else 'red' for c in feature_importance['Coefficient']]

    plt.figure(figsize=(8,6))
    plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], color=colors)
    plt.xlabel("Coefficient (Impact on Stress Level)")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.show()

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
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
    # print("Missing values:")
    # print(df.isnull().sum())
    # print("\n")
    
    # print("Duplicate rows in dataset:")
    # print(df.duplicated().sum())
    # print("\n")
    
    df_clean = df.dropna(inplace=False)
    return df_clean

def remove_outliers(df):
    df_clean = df.copy()
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        # print("No numeric columns detected.")
        return df_clean
    
    mask = np.ones(len(df_clean), dtype=bool)
    
    for col in numeric_cols:
        col_data = pd.to_numeric(df_clean[col], errors='coerce')
        
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask &= col_data.between(lower_bound, upper_bound)
    
    df_clean = df_clean[mask]
    
    # print(f"Removed {len(df) - len(df_clean)} outliers across {len(numeric_cols)} numeric columns.")
    
    return df_clean

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
    df.drop("GPA", axis=1, inplace=True)
    df.drop("Extracurricular_Hours_Per_Day", axis=1, inplace=True)
    df.drop("Social_Hours_Per_Day", axis=1, inplace=True)
    df_clean = clean_data(df)
    order_data_stress_level(df_clean)
    df_clean = remove_outliers(df_clean)
    return df_clean

def normalize_features(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit only on training data
    X_test_scaled = scaler.transform(X_test)    
    return X_train_scaled, X_test_scaled

main()