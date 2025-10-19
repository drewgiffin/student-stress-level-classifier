import pandas as pd

data_path = "student_lifestyle_dataset.csv"

def main():
    df = load_data()
    inspect_data(df)
    df_clean = clean_data(df)
    
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
    
    df_clean = df.dropna(inplace=False)
    return df_clean

main()