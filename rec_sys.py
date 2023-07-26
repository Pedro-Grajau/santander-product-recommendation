from sklearn.preprocessing import LabelEncoder
import pandas as pd

def label_encoder_products(df):
    label_encoder = LabelEncoder()
    label_encoder.fit(df.added_product)
    df['encoded_products'] = label_encoder.\
        transform(df['added_product'])
    return df, label_encoder

def load_test_dataset():
    df_test = pd.read_csv("processed_test.csv")
    df_products = pd.read_csv("added_products.csv")
    df_products, label_encoder = label_encoder_products(df_products)
    return df_test, df_products, label_encoder