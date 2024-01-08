from data_pre_processor import *
from data_presenter import *

# from models.decision_tree import Decision_tree
from models.knn import Knn
from models.nn import Nn
from util import case_and_accent_insensitive_compare

df = pd.read_csv("data/heart.csv")
df_o2_saturation = pd.read_csv("data/o2Saturation.csv")

print('Checking data frame uniqueness... ')
count_cardinality(df)

# remove missing values if found
print('Checking missing values... ')
if any(value != 0 for key, value in count_empty_values(df).items()):
    print('Missing values found!, removing... ')
    df = remove_rows_with_missing_values(df)

# remove if duplicates are found
print('Checking duplicate values... ')
if count_duplicated_values(df) > 0:
    print('Duplicate values found!, removing... ')
    df.drop_duplicates(inplace=True)

# Use the split_data function
train_data, test_data = split_data(df, test_size=0.2, random_state=42)


while True:
    user_input = input('Plot data histogram? [y/n]: ')
    if case_and_accent_insensitive_compare(user_input, 'y'):
        plot_histograms(df, train_data)
    else:
        user_input = input('Chose a model to run. [knn/nn/dt]: ')
        if case_and_accent_insensitive_compare(user_input, 'knn'):

            print('Starting to train data using k-n-neighbours...')
            user_input = input(
                'Use self-search for the best parameters? (if you choose not to, default parameters will be '
                'used). [y/n]')

            if case_and_accent_insensitive_compare(user_input, 'y'):
                classifier = Knn(train_data, test_data, find_best_params=True)
                classifier.classify()
                print('Best parameters have been chosen: ', classifier.find_best_params())
                classifier.plot()
            else:
                classifier = Knn(train_data, test_data)
                classifier.classify()
                print('Default parameters have been chosen. 5-neighbours.')
                classifier.plot()

        elif case_and_accent_insensitive_compare(user_input, 'dt'):

            print('Starting to train data using decision tree...')
            # user_input = input(
            #     'Use self-search for the best parameters? (if you choose not to, default parameters will be '
            #     'used). [y/n]')
            #
            # if case_and_accent_insensitive_compare(user_input, 'y'):
            #     classifier = Decision_tree(train_data, test_data, find_best_params=True)
            #     classifier.classify()
            #     print('Best parameters have been chosen: ', classifier.find_best_params())
            #     classifier.plot_decision_tree()
            #     classifier.plot()
            # else:
            #     classifier = Decision_tree(train_data, test_data)
            #     classifier.classify()
            #     classifier.plot_decision_tree()
            #     classifier.plot()
            #     print('Default parameters have been chosen.')

        else:

            print('Starting to train data using neural network...')
            user_input = input('Use batch gradient descent? (if you choose not to, stochastic gradient decent will be '
                               'used). [y/n]')

            if case_and_accent_insensitive_compare(user_input, 'y'):
                classifier = Nn(train_data, test_data)
                classifier.bgd()
                print('After training using batch gradient descent, plotting the losses and accuracy...: ')
                classifier.plot_loss_accuracy()
                print('Classifying... ')
                classifier.classify()
                classifier.plot()
            else:
                classifier = Nn(train_data, test_data)
                classifier.sgd()
                print('After training using batch gradient descent, plotting the losses and accuracy...: ')
                classifier.plot_loss_accuracy()
                print('Classifying... ')
                classifier.classify()
                classifier.plot()
