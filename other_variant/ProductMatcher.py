from fuzzywuzzy import fuzz
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
import numpy as np

unexp_cat_alias = {
    r'air conditioner':['ac'],
    r'washing machine':['wm'],
    r'desktop':['pc'],
    r'gaming hardware':['game'],
    r'gaming software':['game'],
    r'computer bag':['backpack','carrycase','bag'],
    r'earphone|headphone':['headset','earbud'],
    r'tv lcd':['tv','lcd','led','television'],
    r'smart phone':['mobile phone','phone mobile'],
    r'(phone mobile)|(mobile phone)':['smart phone','mobile phone','phone mobile'],
    r'cooling appliance':['cooler']
}

unexp_brand_alias={
    r'morphyrich':['morphy'],
    r'motorola':['moto'],
    r'hitachitv':['hitachi'],
    r'cromaadonis':['croma'],
    r'brilyant':['catz'],
    r'sony':['ps4'],
    r'ubisoft':['ps4'],
    r'wb':['ps4']
}

class ProductMatcher:
    
    def __init__(self,df: pd.DataFrame,*args,**kwargs)->None:
        self.df = df
        self.brand_alias: dict[str,list[str]] = None
        self.cat_alias: dict[str,list[str]] = None
        
        if 'brand_alias' in kwargs:
            self.brand_alias = kwargs['brand_alias']
        else:
            self.brand_alias = unexp_brand_alias

        if 'cat_alias' in kwargs:
            self.cat_alias = kwargs['cat_alias']
        else:
            self.cat_alias = self.create_cat_alias(unexp_cat_alias)

    def brand_matcher(self,query:str)->list[tuple[str,int]]:
        
        brands = self.df['brand'].unique()
        brand_alias = self.brand_alias
        res = set()
        #matching exact brand name
        search_query = re.sub(r'\-|\'|e-','',query.lower())
        for brand in brands:

            search_brand = re.sub(r'\-|\'','',brand.lower())

            if re.search(r'\b'+search_brand+r'\b',search_query):

                res.add((brand,1))
                continue

            if search_brand in brand_alias and\
            re.search('|'.join(r'\b'+sim+r'\b' for sim in brand_alias[search_brand]),search_query):
                
                res.add((brand,1))

        #matching without spaces
        search_query = re.sub(r'\-|\'|e-','',query.lower())

        for brand in brands:

            search_brand= re.sub(r'\-|\'| ','',brand.lower())
            score = fuzz.partial_ratio(search_brand,search_query)/100

            if search_brand in brand_alias:
                for alias in brand_alias[search_brand]:
                    score = max(score,fuzz.partial_ratio(alias,search_query)/100)

            res.add((brand,score))
  
        return sorted(res,reverse=True,key=lambda x: x[1])[:3]
    
    def cat_matcher(self,query,brand_filter:set[str])->list[tuple[str,int]]:
        if brand_filter:
            brand_df = self.df.loc[self.df['brand'].apply(lambda x: x in brand_filter)]
        else:
            brand_df = self.df

        cat_alias = self.cat_alias
        cats = brand_df['product_line'].unique()
        res = set()
        search_query = re.sub(r'\-|/|\&','',query.lower())
        search_query = re.sub(r' +',' ',search_query).strip()
        #exact match
        for cat in cats:

            if re.search(r'\b'+cat.lower()+r'\b',search_query):
                res.add((cat,1))
                continue

            if re.search('|'.join(r'\b'+alias+r'\b' for alias in cat_alias[cat]),search_query):
                res.add((cat,1))
                continue
            
            for token in search_query.split(' '):
                    
                if not(re.search(r'[aeiou]',token) and len(token)>1):
                    continue
         
                if cat.lower().startswith(token):
                    res.add((cat,1))
                    break
                

        # partial matching using partial ratio
        for cat in cats:
            score = 0
            for alias in cat_alias[cat]:
                score = max(score,fuzz.partial_ratio(alias,search_query)/100)
            
            res.add((cat,score))
      
        return sorted(res,reverse=True,key=lambda x:x[1])[:3]
    
    def create_cat_alias(self,unexp_cat_alias:dict[str,list[str]]):
        
        cat_alias = {}
        wl = WordNetLemmatizer()


        for cat in self.df['product_line'].unique():
            temp = [val.strip() for val in re.split(r'&|/',cat)]

            if len(temp[0].split(' '))>1:
                prefix = temp[0].split(' ')[0]
                for i in range(1,len(temp)):
                    temp[i] = prefix+' '+temp[i]

            elif len(temp[-1].split(' '))>1:
                suffix = temp[-1].split(' ')[-1]
                for i in range(len(temp)-1):
                    temp[i] += ' '+suffix

            cat_alias[cat] = []
            if cat in unexp_cat_alias:
                cat_alias[cat].extend(unexp_cat_alias[cat])

            for alias in  temp:

                root_form = ' '.join(wl.lemmatize(word.lower()) for word in alias.split(' '))
                if re.search(r'\(.*\)',root_form):
                    cat_alias[cat].append(root_form)
                    root_form = re.sub(r'\(.*\)','',root_form).strip()

                for key,values in unexp_cat_alias.items():
                    if re.search(key,root_form):
                        for value in values:
                            cat_alias[cat].append(re.sub(key,value,root_form).strip())

                cat_alias[cat].append(root_form)
                
        return cat_alias
    
    def sku_search(self,query:str,brand_filter:set[str],cat_filter:set[str])->pd.Index:
        
        search_space = self.df.loc[
            self.df.apply(
                lambda x: x['brand'] in brand_filter and x['product_line'] in cat_filter,
                axis=1
            )
        ]
    
        search_space['sku'] = search_space['sku'].apply(
            lambda x: re.sub(
                r' +',' ',
                re.sub(r'\-|/|\&','',x.lower())
            ).strip()
        )
        
        query = re.sub(
            r' +',' ',
            re.sub(r'\-|/|\&','',query.lower())
        ).strip()
        
        
        n = len(query.split(' '))
        threshold = (n-1)/n
        
        total = 0
        
        def score(str1,str2):
            tokens = str1.split(' ')
            total = 0
            for token in tokens:
                total += fuzz.partial_ratio(token,str1)
            
            return total/len(tokens)
            
        
        return search_space.loc[
            search_space['sku'].apply(
                lambda sku: score(query,sku)>= threshold
            )
        ].index
        
        
        
    
    def state_space_search(self,query:str,mini_fetch:int)->pd.DataFrame:
        import heapq
        
        brands = self.brand_matcher(query)
        brand_cat_q = []
            
        for brand,brand_score in brands:
            cats = self.cat_matcher(query,{brand,})
            for cat,cat_score in cats:
                heapq.heappush(brand_cat_q,(-brand_score*cat_score,(brand,cat)))
        
        res_idx = []
    
        while(brand_cat_q and len(res_idx)<mini_fetch):
            score,(brand,cat) = heapq.heappop(brand_cat_q)
            res_idx.extend(self.sku_search(query,{brand,},{cat,}))
         
        return self.df.iloc[np.array(res_idx)-1]
            
            