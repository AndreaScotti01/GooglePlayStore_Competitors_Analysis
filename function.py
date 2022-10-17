import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import pickle
from function import *
import plotly.express as px
import plotly.figure_factory as ff
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from google_play_scraper import app, Sort, reviews_all,search
from deep_translator import GoogleTranslator
from textblob import TextBlob

def remove_similar(df):
    data = df.copy()
    ls = []
    for idx in range(df.shape[0]):
        A = data["Category"][idx].split()
        B = data["Genres"][idx].split()
        condition = (A == B)
        if condition == False:
            for elemA in set(A):
                for elemB in set(B):
                    if elemA == elemB:
                        B.remove(elemA)
            ls.append(B)
        else:
            ls.append("None")
    for idx, elem in enumerate(ls):
        if ls[idx] != "None":
            ls[idx] = " ".join(ls[idx])

    data["Genres"] = pd.DataFrame(ls).copy()
    return data, ls

def find_competitor(df, param1: str, param1_cond: str, param2: str,
                    param2_cond: float, operator2: str, sort_by:str, param3:str,csv_file_name:str):
    condition1 = (df[param1] == param1_cond)

    if operator2 == ">=":
        condition2 = (df[param2] >= param2_cond)
    elif operator2 == ">":
        condition2 = (df[param2] > param2_cond)
    elif operator2 == "<=":
        condition2 = (df[param2] <= param2_cond)
    elif operator2 == "<":
        condition2 = (df[param2] < param2_cond)
    elif operator2 == "==":
        condition2 = (df[param2] == param2_cond)
    elif operator2 == "!=":
        condition2 = (df[param2] != param2_cond)
    
    competitor = df.loc[condition1 & condition2].sort_values(sort_by,ascending=True)
    if competitor.shape[0] == 0:
        print("There are no aps that match those conditions.")
        return competitor, None
    else:
        competitor_name = competitor[param3].values[0]
        comments = pd.read_csv(csv_file_name)
        comments = comments.loc[comments[param3] == competitor_name]
        if comments.shape[0] == 0:
            print("Comments about the app are not in the file.")
            return competitor, competitor_name
        return competitor, comments

def find_reviews(palystore_app_id:str,
                 score:int,
                 score_operator:str,
                 above_thumbsUpCount_quantile:float,below_year:int
                ):
    
    reviews = reviews_all(palystore_app_id,sort=Sort.MOST_RELEVANT)
    df_reviews = pd.DataFrame(data=np.array(reviews).tolist()).drop(columns=[
    "reviewId", "userName", "replyContent", "userImage",
    "reviewCreatedVersion", "repliedAt"])
    s = score
    so = score_operator
    q = above_thumbsUpCount_quantile
    ye = below_year
    
    if so == ">=":
        df_reviews = df_reviews.query(
    'score >= @s and thumbsUpCount >= thumbsUpCount.quantile(@q) and at <= @ye').sort_values("thumbsUpCount", ascending=False).reset_index(drop=True)
    elif so == ">":
        df_reviews = df_reviews.query(
    'score > @s and thumbsUpCount >= thumbsUpCount.quantile(@q) and at <= @ye').sort_values("thumbsUpCount", ascending=False).reset_index(drop=True)
    elif so == "<=":
        df_reviews = df_reviews.query(
    'score <= @s and thumbsUpCount >= thumbsUpCount.quantile(@q) and at <= @ye').sort_values("thumbsUpCount", ascending=False).reset_index(drop=True)
    elif so == "<":
        df_reviews = df_reviews.query(
    'score < @s and thumbsUpCount >= thumbsUpCount.quantile(@q) and at <= @ye').sort_values("thumbsUpCount", ascending=False).reset_index(drop=True)
    elif so == "==":
        df_reviews = df_reviews.query(
    'score == @s and thumbsUpCount >= thumbsUpCount.quantile(@q) and at <= @ye').sort_values("thumbsUpCount", ascending=False).reset_index(drop=True)
    elif so == "!=":
        df_reviews = df_reviews.query(
    'score != @s and thumbsUpCount >= thumbsUpCount.quantile(@q) and at <= @ye').sort_values("thumbsUpCount", ascending=False).reset_index(drop=True)
    
    return df_reviews

def translate(df,col_to_tanslate):
    df_translated = df.copy()
    df_translated[col_to_tanslate] = df_translated[col_to_tanslate].apply(lambda x: GoogleTranslator(
    source='auto', target='en').translate(str(x).lower()))
    return df_translated

def polarity_score(df):
    df_reviews = df.copy()
    scaler = MinMaxScaler()
    df_reviews["content"] = df_reviews["content"].astype(str)
    df_reviews["polarity"] = df_reviews["content"].apply(lambda x: TextBlob(x).sentiment.polarity)
    df_reviews["len_text"] = df_reviews["content"].apply(lambda x: len(x.split()))
    df_reviews["polarity_score"] = ((df_reviews["polarity"] * df_reviews["thumbsUpCount"]) *df_reviews["len_text"]) / df_reviews["score"]
    df_reviews = df_reviews.query("polarity_score < 0")
    df_reviews["polarity_score"] = (1 - scaler.fit_transform(df_reviews[["polarity_score"]])) * 100
    df_reviews = df_reviews.sort_values("polarity_score",ascending=False).reset_index(drop=True)
    return df_reviews

def find_app_id(app_name):
    result = search(app_name,lang="en",country="us",n_hits=1)
    app_id = list(result[0].values())[0]
    return app_id

def find_app_comments(df,col_of_app_names,max_iter,stars,operator,quantile,year):
    for idx,names in enumerate(df[col_of_app_names].unique()):
        if idx == max_iter:
            print("Reache max numbers of iterations allowed.")
            break
        else:
            app_id = find_app_id(names)
            print(f"\nApp name:{names}",f"\nApp playstore id:{app_id}",f"\nIteration:{idx}")
            reviews = find_reviews(app_id,stars,operator,quantile,year)
            if reviews.shape[0] == 0:
                print("No comments match the requirements")
            else:
                print("Match found.")
                break 
    return reviews

def evaluate_comments(df_base,query1:str,result_quesry1,param1,quantile1,operator1,sort_col,apps_name_col,file_name,quantile2,year,max_iter,stars,operator2,need_translation):
    #Create a local variable
    df = df_base.loc[df_base[query1] == result_quesry1].copy()
    
    #Find all aps that do match certian condition and sort them by specific parameters, then based on the sorted list it returns the whole dataframe and the app that is at the top
    print("Finding all apps that mach the specified critiria ...")
    competitors, name = find_competitor(df, query1, result_quesry1, param1, df[param1].quantile(quantile1), operator1,
    sort_col, apps_name_col,file_name)
    print("No comments found on the given file.")
    
    if competitors.shape[0] != 0:
        # Get the app id and looks for comments on thw playstore
        print("Looking online for comments ...")
        reviews = find_app_comments(competitors,apps_name_col,max_iter,stars,operator2,quantile2,year)
        
        if need_translation == True:
            # Translate the comments
            print("Traslating comments ...")
            reviews = translate(reviews,"content")

        # Evaluate the comments
        print("Evaluating comments ...")
        reviews = polarity_score(reviews)
    
        # Create a list of all comments
        full_text = list(reviews["content"].values)
        print("Done!")
        
        return reviews,full_text
        
    else:
        print("There are no apps to show.")
        return competitors
    
    