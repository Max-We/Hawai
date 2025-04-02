import pandas as pd

datei_pfad_pers='/Users/felixipfling/Documents/GitHub/Hawai/personas_Simp_uuid.csv'
datei_pfad_ques='/Users/felixipfling/Documents/GitHub/Hawai/data/questionnaires.csv'

df_pers=pd.read_csv(datei_pfad_pers)
df_ques=pd.read_csv(datei_pfad_ques)

df=pd.merge(df_pers, df_ques, on='uuid', how='inner') 

def replace_values(df, column_list, condition_column, condition_value):
    """
    Ersetzt Werte in bestimmten Spalten durch 1, wenn sie 10 enthalten und
    eine bestimmte Bedingung in einer anderen Spalte erfüllt ist.

    :param df: Pandas DataFrame
    :param column_list: Liste der Spaltennamen, die überprüft werden sollen
    :param condition_column: Name der Spalte mit der Bedingung
    :param condition_value: Der Wert in der Bedingungsspalte, der erfüllt sein muss
    :return: DataFrame mit ersetzten Werten
    """
    mask = df[condition_column] == condition_value
    df.loc[mask, column_list] = df.loc[mask, column_list].replace(10, 1)
    return df


nutColls=["marzipan"]
veganPLUSColls=["mayonnaise","eggs","honey"]
milkyColls=[  "blue_cheese",
    "butter_bread",
    "cheesecake",
    "cream",
    "goats_cheese",
    "hard_cheese",
    "ice_cream",
    "milk_chocolate",
    "plain_yogurt",
    "skimmed_milk",
    "soft_cheese",
    "whole_milk"]
meatColls=["bacon", "baked_steamed_fish", "barbequed_grilled_meat", "beef_steak", 
    "bolognese_sauce", "burgers_meat", "chicken", "cod", "fried_chicken", "fried_battered_fish", "herring", "lamb", "liver", "pollock", "pork_chop", "red_meat", 
    "roast_chicken", "salami", "salmon", "sardines", "sausages_meat", "smoked_fish", "tinned_tuna"]

veganColls=meatColls+veganPLUSColls+milkyColls


df=replace_values(df,veganColls,"diet","Vegan")
df=replace_values(df,meatColls,"diet","Vegetarian")
df=replace_values(df,nutColls,"diet","Nut Allergy")
df=replace_values(df,milkyColls,"diet","Lactose Intolerant")

def drop_columns(df, columns_to_drop):
    
    return df.drop(columns=columns_to_drop, errors='ignore')


collsToDrop = ["name","age","gender","country","diet","description"]
df=drop_columns(df,collsToDrop)
df.to_csv("questionairsMOD.csv", index=False)
