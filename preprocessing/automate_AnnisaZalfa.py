import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def get_col_types(df):
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return cat_cols, num_cols

def handle_missing_values(df):
    cat_cols, num_cols = get_col_types(df)
    
    # Imputasi Numerical dengan Median
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
            
    # Imputasi Categorical dengan Mode
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
            
    print("Missing values handled.")
    return df

def preprocessing(df, columns_to_drop=None):
    # 1. Drop Kolom yang tidak perlu di awal
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        print(f"Dropped columns: {columns_to_drop}")

    # 2. Tangani duplikat
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {initial_rows - len(df)} duplicate rows.")

    # 3. Tangani missing values
    df = handle_missing_values(df)

    # 4. Re-identifikasi kolom setelah drop
    cat_cols, num_cols = get_col_types(df)

    # 5. Encoding Categorical
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    print("Categorical columns encoded.")

    # 6. Scaling Numerical
    scaler = MinMaxScaler()
    if num_cols:
        df[num_cols] = scaler.fit_transform(df[num_cols])
        print("Numerical columns scaled.")

    return df, label_encoders, scaler

# Main Execution
if __name__ == "__main__":
    # Load dataset dengan menangani tanda tanya '?' sebagai NaN
    data_path = r'D:\DICODING\MSML\Eksperimen_SML_Annisa Zalfa\raw_dataset.csv'
    dataset = pd.read_csv(data_path, na_values='?') 
    cols_to_drop = [] 
    
    processed_df, encoders, scaler = preprocessing(dataset.copy(), cols_to_drop)
    processed_df.to_csv(r'D:\DICODING\MSML\Eksperimen_SML_Annisa Zalfa\processed_dataset.csv', index=False)
    print("\nProcessed Data Sample:")
    print(processed_df.head())
    