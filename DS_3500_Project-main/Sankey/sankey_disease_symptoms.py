import pandas as pd
import sankey as sk


def combine_cols(df):

    # put all symptoms together in one column
    df['All_Symptoms'] = df[df.columns[1:]].apply(
        lambda x: ','.join(x.dropna().astype(str)),
        axis=1)

    df_combined = df.groupby(['Disease'])['All_Symptoms'].apply(lambda x: ','.join(x)).reset_index()

    df_combined['All_Symptoms'] = df_combined['All_Symptoms'].str.split(', ')
    df_combined = df_combined.explode('All_Symptoms')
    df_combined = df_combined.drop_duplicates()

    # add values column to prep for sankey
    df_combined['Value'] = ['1' if m != 'nan' else '0' for m in df_combined['All_Symptoms']]

    return df_combined

def unique(df):

    # make sure there are no repeat symptoms in each disease row
    for i in range(len(df)):
        list_res = (list(df.loc[i, "All_Symptoms"]))
        print(list_res)
    df.to_csv('groupby_test.csv', sep='\t', encoding='utf-8')
    return df

def filter_input(df, col, input):

    # use isin function to filter rows for only those that have given symptom
    df = df[df[col].isin(input)]

    return df

def filter_input1(df, col, input):

    # use isin function to filter rows for only those that have given symptom
    df = df[df[col].isin(input)]

    return df


def make_sankey_diagrams(df, left, right, values):

    # make sankey diagram
    sk.make_sankey(df, left, right, values, pad=50)

def main():
    # read in csv file
    df = pd.read_csv("disease_data.csv")

    # put all symptoms together in one "All_Symptoms" columns
    df_combined = combine_cols(df)

    df_combined = df_combined.reset_index()
    df_combined = df_combined.drop(['index'], axis=1)

    print(df_combined)

    # ask user for symptoms
    symptom_list = input("Enter all symptoms separated by space  ")

    # Split string into words
    symptoms = symptom_list.split(" ")
    print(symptoms)
    print(df_combined)

    # filter dataframe for only user inputted symptoms
    df_input = filter_input(df_combined, "All_Symptoms", symptoms)

    print(df_input)

    # add headers
    headers = ["Disease", "All_Symptoms", "Value"]
    df_input.columns = headers
    print(df_input)

    make_sankey_diagrams(df_input, 'All_Symptoms', 'Disease', 'Value')

main()