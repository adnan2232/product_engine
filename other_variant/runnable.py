import pandas as pd
from ProductMatcher import ProductMatcher
import warnings
warnings.filterwarnings('ignore')

def top_three_selling(df):

    return df.sort_values(by="sales", ascending=False).iloc[:3]


def top_three_low_price(df):

    return df.sort_values(by="price").iloc[:3]


def top_three_high_price(df):

    return df.sort_values(by="price", ascending=False).iloc[:3]

if __name__ == '__main__':
    df = pd.read_csv("sales_data.csv",index_col=0)
    df['brand'] = df["brand"].str.replace('\r','')
    se = ProductMatcher(df)
    
    while True:

        print("For exiting you can press ctrl+d or simply write exit\n")
        try:
            text = input("Query: ")
        except:
            print("Exiting the program")
            break

        if text.lower().strip() == "exit":
            print("Exiting the program")
            break
        print()

        result_df = se.state_space_search(text,9)
       

        df_top_sell = top_three_selling(result_df)
        

        df_top_high_price = top_three_high_price(result_df)
        

        df_top_low_price = top_three_low_price(result_df)

        df_top_mr = result_df.head(5)

        print(
            f"Top Selling Products:\n{df_top_sell[['sku','sales']].to_string()}\n"
        )
        print(
            f"Lowest Price Products:\n{df_top_low_price[['sku','price']].to_string()}\n"
        )
        print(
            f"Highest Price Products:\n{df_top_high_price[['sku','price']].to_string()}\n"
        )

        print(
            f"Top 5 matched results:\n{df_top_mr['sku'].to_string(header=False)}\n"
        )