from nltk.stem import WordNetLemmatizer
from Levenshtein import ratio
import pandas as pd
import numpy as np
import nltk
import re


class SmallSearchEngine:
    def __init__(self) -> None:

        nltk.download("wordnet",quiet=True)
        self.lemma = WordNetLemmatizer()

    def read_df_parquet(self, path: str) -> pd.DataFrame:

        df = pd.read_parquet(path)
        return df

    def read_df_csv(self, path: str, index_col: int = -1) -> pd.DataFrame:

        if index_col > -1:
            df = pd.read_csv(path, index_col=index_col)
        else:
            df = pd.read_csv(path)

        return df

    #Converts text to list by splitting,lowering,lemmatizing and optionally changing abbrevation of words
    def text_to_list(
        self, txt: str, splitter: str = " ", lower: bool = True, lemmatize: bool = True, *args, **kwargs
    ) -> list[str]:

        if lower:
            txt = txt.lower()

        if lemmatize:
            res = [
                self.lemma.lemmatize(word) for word in txt.split(splitter)
            ]  # Word map to root form
        else:
            res =  txt.strip().split(splitter)


        if "abb" in kwargs:

            for i in range(len(res)):
                if res[i] in kwargs["abb"]:
                    
                    res[i] = kwargs["abb"][res[i]]

        return res
                
    #Special Character Seperator such & and /
    def special_char_sep(self,txt:str,splitter:str)->list[str]:
    
        temp = txt.split(splitter)
        
        if len(temp)<=1:
            return temp
        
        start_extra, end_extra = "", ""
        
        extra = self.text_to_list(temp[0].strip(),splitter=" ")
        if len(extra)>1:
            start_extra = extra[0]
            temp[0] = extra[1]
        
        extra = self.text_to_list(temp[-1].strip(),splitter=" ")
        if len(extra)>1:
            end_extra = extra[1]
            temp[-1] = extra[0]
            
        for i in range(len(temp)):
            temp[i] = (start_extra+" "+temp[i].strip()+" "+end_extra).strip()
        
        return temp
    
    # max_win_score uses window size of category words and calculate Levenshtein similarity ratio
    # if score is >= 0.5 particuar brand df is return else all brands df
    def max_win_score(self, cats: list[str], txt_ls: list) -> dict[str, float]:

        txt_n = len(txt_ls)
        cat_scores = {cat: 0 for cat in cats}

        for cat in cats:
            cat_ls = self.special_char_sep(cat,splitter="&")
        
            if len(cat_ls)<=1:
                cat_ls = self.special_char_sep(cat,splitter="/")

            for inter_cat in cat_ls:
                
                n = len(inter_cat.split(" "))
                for i in range(txt_n-n+1):
                    temp = " ".join(txt_ls[i:i+n])
                    cat_scores[cat] = max(cat_scores[cat],ratio(inter_cat.lower(),temp.lower()))

        return cat_scores

    # calculate max average score by permuting all possible combination of words pair and selecting max pair score for each cat
    def perm_avg_score(self, cat_ls: list[str], txt_ls: list[str]) -> np.float64:

        score = {word: 0 for word in cat_ls}

        for word_cat in cat_ls:

            for word_txt in txt_ls:

                score[word_cat] = max(score[word_cat], ratio(word_cat, word_txt))

        return np.mean(list(score.values()))

    # average_score calculate Levenshtein similarity ratio for each category with search text
    # and average max similarity ratio for each category
    def average_score(self, cats: list[str], txt_ls: list, lemmatize: bool=True,*args,**kwargs) -> dict[str, float]:

        cat_scores = {cat: 0 for cat in cats}

        for cat in cats:

            cat_ls = self.special_char_sep(cat,splitter="&")
        
            if len(cat_ls)<=1:
                cat_ls = self.special_char_sep(cat,splitter="/")
                
            for inner_cat in cat_ls:

                inner_cat_ls = self.text_to_list(inner_cat, splitter=" ",lower=True,lemmatize=lemmatize,*args,**kwargs)
                cat_scores[cat] = max(cat_scores[cat],self.perm_avg_score(inner_cat_ls,txt_ls))

        return cat_scores

    # this method combines both max_win_score and average_score technique
    # it moves window of length cat words over search text (ordered)
    # each window calculates unordered average score of search text words inside the window with words in categories
    def combine_score(self, cats: list[str], txt_ls: list[str], lemmatize:bool=True,*args,**kwargs) -> dict[str, float]:
   
        txt_n = len(txt_ls)
        cat_scores = {cat: 0 for cat in cats}

        for cat in cats:

            cat_ls = self.special_char_sep(cat,splitter="&")
        
            if len(cat_ls)<=1:
                cat_ls = self.special_char_sep(cat,splitter="/")
                
            for inner_cat in cat_ls:
              
                inner_cat_ls = self.text_to_list(inner_cat, splitter=" ",lower=True,lemmatize=lemmatize,*args,**kwargs)
                n = len(inner_cat_ls)
                
                for i in range(txt_n-n+1):
                    temp = self.perm_avg_score(inner_cat_ls,txt_ls[i:i+n])
                    cat_scores[cat] = max(cat_scores[cat],temp)

        return cat_scores

    #Method for selecting appropiate score calculator
    def calculate_score(
        self, df: pd.DataFrame, column_name: str, txt_ls: list[str], method: str, lemmatize: bool=True,*args, **kwargs
    ) -> dict[str, float]:

        assert (
            method == "average_score"
            or method == "max_win_score"
            or method == "combine_score"
        ), f"No scoring metircs name: {method}\nAvailable scoring metrics are: average_score, max_win_score and combine_score"

        if method == "average_score":
            return self.average_score(df[column_name].unique(), txt_ls,lemmatize,*args,**kwargs)
        elif method == "max_win_score":
            return self.max_win_score(df[column_name].unique(), txt_ls,*args, **kwargs)
        else:
            return self.combine_score(df[column_name].unique(), txt_ls,lemmatize,*args, **kwargs)

    # exact_match function first try exact matching of brand name in search text and return that brand dataframe
    # if no exact match found, partial match is done using average_score, max_win_score or combine scoring
    def exact_match(
        self,
        df: pd.DataFrame,
        column_name: str,
        txt: str,
        method: str = "max_win_score",
    ) -> pd.DataFrame:

        txt_ls = self.text_to_list(txt, lemmatize=False)
        ind = df[column_name].isin(txt_ls)

        if ind.any():
            return df[ind]

        tp = self.calculate_score(df, column_name, txt_ls, method,lemmatize=False)

        ele = max(tp.items(), key=lambda x: x[1])

        return df.loc[df[column_name] == ele[0]].copy() if ele[1] >= 0.75 else df.copy()

    # partial match return top_scoring product_lines using average_score or max_win_score
    def partial_match(
        self,
        df: pd.DataFrame,
        column_name: str,
        txt: str,
        method: str = "combine_score",
        lemmatize: bool = True,
        *args,
        **kwargs
    ) -> pd.DataFrame:
        
        txt_ls = self.text_to_list(txt,splitter=" ",lower=True,lemmatize=lemmatize,*args,**kwargs)

        tp = self.calculate_score(df, column_name, txt_ls, method,lemmatize=lemmatize,*args,**kwargs)

        tp = sorted(tp.items(), key=lambda x: x[1], reverse=True)

        ind = df[column_name].isin(
            [x for x, y in tp if y > tp[0][1]-0.1] if tp[0][1] > 0.65 else [x for x, y in tp]
        )
        return df[ind].copy()

    # Above methods based on scoring categories, this method score records based on search text
    # it uses threshold of atleast n-1 words (score = (txt_n-1)/txt_n)
    # and for more precision it restricts number of records for any threshold
    # if no_of_records > threshold for score > (txt_n-1)/txt_n it will stop further searching (score=(txt_n-1)/txt_n)
    # 1/txt_n is for variance
    def inverse_partial_match(
        self, df: pd.DataFrame, column: str, txt: str
    ) -> pd.DataFrame:

        txt_ls = self.text_to_list(txt)
        txt_n = len(txt_ls)
        filter_vals = df[column].apply(lambda x: self.text_to_list(x))

        thresholds = {np.around(i, decimals=2): [] for i in np.arange(0, 1.05, 0.1)}

        for sku in filter_vals.index:
            txt_score = [0 for _ in range(txt_n)]
            for i in range(txt_n):
                for word in filter_vals[sku]:

                    txt_score[i] = max(txt_score[i], ratio(txt_ls[i], word))
            txt_score.sort(reverse=True)

            avg = np.mean(txt_score)
            thresholds[round(avg, 1)].append(sku)

        res_id = []
        for threshold in np.arange(1, (txt_n - 1) / txt_n, -0.1):
            res_id.extend(thresholds[np.around(threshold, 2)])
            if len(res_id) >= 5:
                return df.loc[res_id].copy()

        return df.loc[res_id].copy() if len(res_id) > 0 else df.copy()
