from Levenshtein import ratio
from nltk.stem import WordNetLemmatizer
from typing import Union
import pandas as pd
import numpy as np
import nltk
import re

class SearchEngine:
    def __init__(self) -> None:
        nltk.download("wordnet")
        self.lemma = WordNetLemmatizer()

    def read_df_parquet(self, path: str) -> pd.DataFrame:
        df = pd.read_parquet(path)
        return df

    def read_df_csv(self, path: str, index_col:int=-1) -> pd.DataFrame:

        if index_col >-1:
            df = pd.read_csv(path,index_col=index_col)
        else:
            df = pd.read_csv(path)
        
        return df

    #max_win_score uses window size of category words and calculate Levenshtein similarity ratio (Windows shrink at the end)
    #if score is >= 0.5 particuar brand df is return else all brands df
    def max_win_score(self ,cats:str, txt_ls:list) -> dict[str,float]:
        txt_n = len(txt_ls)
        cat_scores = {cat:0 for cat in cats}
        for cat in cats:
            cat_ls = cat.split(" ")
            n = len(cat_ls)

            for i in range(txt_n):
                temp = " ".join(txt_ls[i:i+n])
                cat_scores[cat] = max(cat_scores[cat],ratio(cat.lower(),temp.lower()))

        return cat_scores

    #average_score take cartesian cross product, calculate Levenshtein similarity ratio
    #and average max similarity ratio for each item
    #if score is >= 0.5 particuar brand df is return else all brands df
    def average_score(self,cats: str,txt_ls:list) -> dict[str,float]:

        cat_scores = {cat:0 for cat in cats}
        for cat in cats:
            cat_ls = cat.split(" ")
            score = {word:0 for word in cat_ls}
            
            for word_cat in cat_ls:
                for word_txt in txt_ls:
                    score[word_cat] = max(score[word_cat],ratio(word_cat,word_txt))
        
            cat_scores[cat] = np.mean(list(score.values()))
    
        return cat_scores
                

    #exact_match function first try exact matching of brand name in search text and return that brand or  dataframe
    #if no exact match found, partial match is done using average_score or max_win_score

    def exact_match(self, df:pd.DataFrame, column_name:str, txt:str, method:str ="average_score")->pd.DataFrame:
        txt_ls = txt.lower().split(" ")
        ind = df[column_name].isin(txt_ls)
        
        if ind.any():
            return df[ind]
        
        if method=="average_score":
            tp = self.average_score(df[column_name].unique(),txt_ls)
        else:
            tp = self.max_win_score(df[column_name].unique(),txt_ls)
            
        ele = max(tp.items(),key= lambda x:x[1])
        return df.loc[df[column_name]==ele[0]].copy() if ele[1]>=0.5 else df.copy()

    #partial match return top_scoring product_lines using average_score or max_win_score
    def partial_match(self, df:pd.DataFrame, column_name:str ,txt:str ,top_ele:int=3, method:str="average_score", lemmatize:bool=True)-> pd.DataFrame:
        
        if lemmatize:
            txt_ls = [self.lemma.lemmatize(word) for word in txt.lower().split(" ")]
        else:
            txt_ls = txt.lower().split(" ")

        if method=="average_score":
            tp = self.average_score(df[column_name].unique(),txt_ls)
        elif method=="max_win_score":
            tp = self.max_win_score(df[column_name].unique(),txt_ls)
        else:
            raise ValueError(f"No scoring metircs name: {method}\nAvailable scoring metrics are: average_score and max_win_score")
            
        elements = [x for x,y in sorted(tp.items(),key = lambda x: x[1],reverse=True)[:top_ele]]
        ind = df[column_name].isin(elements)
        
        return df[ind].copy()


    