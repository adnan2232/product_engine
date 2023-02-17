from typing import Union
import pandas as pd
import numpy as np
import re

class SearchEngine:

    def __init__(self,df_or_path: Union[str,pd.DataFrame], index_col:int=-1, replacement: dict[re.Pattern,str] = None) -> None:

        self.df: pd.DataFrame
        self.replacement = replacement if replacement else dict()

        if isinstance(df_or_path,str):
            try:
                if df_or_path.split(".")[1] == "csv":
                    self.set_df_csv(df_or_path,index_col=index_col)

                elif df_or_path.split(".")[1] == "parquet":
                    self.set_df_parquet(df_or_path)

                else:
                    raise FileNotFoundError("Looks like your path file extension is not supported")

            except IndexError:
                raise IndexError("File extension not found in your path")
        
        elif isinstance(df_or_path,pd.DataFrame):
            self.set_df(df_or_path)
        
        else:
            raise Exception("Passed df_or_path should be path to data or a dataframe")

    def set_df(self,df: pd.DataFrame) -> None:
        self.df = df

    def set_df_parquet(self, path: str) -> None:

        self.df = pd.read_parquet(path)

    def set_df_csv(self, path: str, index_col:int=-1) -> None:

        if index_col >-1:
            self.df = pd.read_csv(path,index_col=index_col)
        else:
            self.df = pd.read_csv(path)

    def text_to_set(self,query: str, replacement: dict[re.Pattern,str] ={}) -> set[str]:

        if not replacement:
            replacement = self.replacement

        for key,val in replacement.items():

            query = re.sub(key,val,query)

        return set(query.strip().split(" "))

    def exact_match(self, category: str, bag_of_words: Union[list,set,tuple], df: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
        
        assert hasattr(self,"df") or not df.empty, "No data to query on please set data frame using set_df_parquet, set_df_csv or set_df or pass data frame"

        if df.empty:
            df = self.df

        ind = df[category].isin(bag_of_words)

        if ind.any():
            return df.loc[ind].copy()

        return df.copy()



    