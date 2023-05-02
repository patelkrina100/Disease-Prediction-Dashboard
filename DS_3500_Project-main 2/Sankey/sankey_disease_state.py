"""
Produces a sankey diagram that links diseases to states, with the link sizes being determined by the disease's prevalence
in the state.

Unused in our dashboard, but still functional
"""
import pandas as pd
import numpy as np
import sankey as sk

def clean_cols(df, col1, col2, col3):

    # filter to keep only necessary columns
    df = df.filter([col1, col2, col3], axis=1)

    return df

def filter_input(df, col, diseaseinput):

    disease_df = df[df[col] == diseaseinput]

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

def make_sankey_diagrams(df, left, right, values):

    # make sankey diagram
    fig = sk.make_sankey(df, left, right, values, pad=50)
    return fig