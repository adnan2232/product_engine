from search_engine.search_engine import SmallSearchEngine
import pandas as pd


def retrieve_result(df: pd.DataFrame, text: str) -> pd.DataFrame:
    abbrevations = {
        "cooler":"cooling appliance",
        "heater":"heating appliance",
        "headset":"headphone",
    }
    df_brand = se.exact_match(df, "brand_lower", text, method="max_win_score")
    df_product = se.partial_match(
        df_brand, "product_line_clean", text, method="combine_score", abb=abbrevations
    )
    df_res = se.inverse_partial_match(df_product, "sku", text)
    return df_res


def top_three_selling(df):

    return df.sort_values(by="sales", ascending=False).iloc[:3]


def top_three_low_price(df):

    return df.sort_values(by="price").iloc[:3]


def top_three_high_price(df):

    return df.sort_values(by="price", ascending=False).iloc[:3]


def other_sku(df, ids):
    return df.loc[~df.index.isin(ids)]


if __name__ == "__main__":

    se = SmallSearchEngine()
    df = se.read_df_parquet("sales_data.parquet")
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
        result_df = retrieve_result(df, text)
        others = []

        df_top_sell = top_three_selling(result_df)
        others.extend(df_top_sell.index)

        df_top_high_price = top_three_high_price(result_df)
        others.extend(df_top_high_price.index)

        df_top_low_price = top_three_low_price(result_df)
        others.extend(df_top_low_price.index)

        df_others = other_sku(result_df, others)

        print(
            f"Top Selling Products:\n{df_top_sell[['sku','sales']].to_string()}\n"
        )
        print(
            f"Lowest Price Products:\n{df_top_low_price[['sku','price']].to_string()}\n"
        )
        print(
            f"Highest Price Products:\n{df_top_high_price[['sku','price']].to_string()}\n"
        )

        if not df_others.empty:
            print(
                f"Other Products you may also like:\n{df_others['sku'].to_string(header=False)}\n"
            )
