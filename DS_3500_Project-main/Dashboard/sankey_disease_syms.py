import pandas as pd
import numpy as np
import sankey as sk

def clean_cols(df, col1, col2, col3):

    # filter to keep only necessary columns
    df = df.filter([col1, col2, col3], axis=1)

    return df

def filter_input(df, col, diseaseinput):

    disease_df = df[df[col].isin(diseaseinput)]
    return disease_df

def drop_zeros(df, col):

    # drop any rows where the input is 0
    cleaned_df = df.drop(df[df[col] == 0].index)

    cleaned_df2 = cleaned_df[cleaned_df[col].notna()]

    return cleaned_df2

def threshold(df, col):

    # drop rows in column with value below threshold of 20
    thres_df = df.drop(df[df[col] < 20].index)

    return thres_df

def round(df, col):

    # round all values in column to next whole number

    df[col] = df[col].round(0)

    df[col] = df[col].astype('int')

    return df

def make_df(data):
    dic = {}
    columns = list(data.columns.copy())
    columns.pop(0)

    for index, row in data.iterrows():
        row1 = row.dropna()
        if row1[0] not in dic:
            dic[row1[0]] = {}

        for index, element in enumerate(row1):
            if index == 0:
                continue
            if row1[index] not in dic[row1[0]].keys():
                dic[row1[0]][row1[index]] = 1
            else:
                dic[row1[0]][row1[index]] += 1

    source_list = []
    target_list = []
    val_list = []
    for k,v in dic.items():
        for key, val in v.items():
            source_list.append(k)
            target_list.append(key)
            val_list.append(val)

    df = pd.DataFrame(list(zip(source_list, target_list, val_list)), columns = ['Disease', 'Symptom', 'Val'])

    return df

def make_sankey_diagrams(df, left, right, values):

    # make sankey diagram
    fig = sk.make_sankey(df, left, right, values,
                         title='Sankey Diagram Mapping Your Chosen Disease(s) to Correlated Symptoms', pad=50)
    return fig
