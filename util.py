import pandas as pd
import unicodedata

# df = pd.read_csv("data/heart.csv")
# train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])


def case_and_accent_insensitive_compare(str1, str2):
    return remove_accents(str1.casefold()) == remove_accents(str2.casefold())


def get_df_name(df):
    name = [x for x in globals() if globals()[x] is df][0]
    return name


def separate_x_y(dataframe_names, independent_vars):
    """
    a function to separate test, train sets' X and y variables

    Input:
        dataframe_names: the dataframe that used for extraction X and y variables
        independent_vars: independent variables used during extraction
    Return:
        X_train: Independent variables of train data.
        y_train: Dependent variables of train data.
        X_test: Independent variables of test data.
        y_test: Dependent variables of test data.

    """
    datasets = {}

    for i in tuple(dataframe_names):
        # get dataframe name
        X = "X_" + get_df_name(i).replace("_data", "")
        y = "y_" + get_df_name(i).replace("_data", "")

        datasets[X] = i.loc[:, independent_vars]
        datasets[y] = i["output"]

    return datasets.get('X_train'), datasets.get('y_train'), datasets.get('X_test'), datasets.get("y_test")
