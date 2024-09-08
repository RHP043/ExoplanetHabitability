import pandas as pd
import matplotlib as plt
import math
import numpy as np
from preprocessing import planets, col_names


def entropy(prob_vector):
    entropy_val = 0
    for prob in prob_vector:
        if prob != 0:
            entropy_val += prob * math.log(prob, 2)
    return -entropy_val


def information_gain(data, column_name, target_name):
    column_values = data[column_name].unique()
    target_values = data['P_HABITABLE'].unique()
    total_count = len(data)

    # Calculate the entropy of the target column
    target_probabilities = []
    for target_value in target_values:
        target_count = len(data[data['P_HABITABLE'] == target_value])
        target_probability = target_count / total_count
        target_probabilities.append(target_probability)
    target_entropy = entropy(target_probabilities)

    # Calculate the conditional entropy for each unique value of the column
    conditional_entropies = []
    for column_value in column_values:
        subset = data[data[column_name] == column_value]
        subset_count = len(subset)
        subset_probabilities = []
        for target_value in target_values:
            target_count = len(subset[subset['P_HABITABLE'] == target_value])
            if subset_count != 0:
                target_probability = target_count / subset_count
            subset_probabilities.append(target_probability)
        subset_entropy = entropy(subset_probabilities)
        conditional_entropies.append(subset_entropy * (subset_count / total_count))

    # Calculate the information gain
    information_gains = target_entropy - sum(conditional_entropies)
    return information_gains


def build_decision_tree(data, target_name, header_list):
    global bc
    goat_column = None
    max_information_gain = -1

    # Find the column with the highest information gain
    for column_name in header_list:
        if column_name != target_name:
            information_gain_value = information_gain(data, column_name, target_name)
            if information_gain_value > max_information_gain:
                max_information_gain = information_gain_value
                goat_column = column_name
                if information_gain_value == 0:
                    break
                bc = column_name

    decision_tree = {}
    decision_tree['split_on'] = goat_column
    decision_tree['split_components'] = {}
    # print(decision_tree)

    # Split the data based on the best column

    # Gives unique values in column
    column_values = data[goat_column].unique()
    for column_value in column_values:
        subset = data[data[goat_column] == column_value]
        subset_target_values = subset[target_name].unique()

        # Not recursive, just adds final values to decision tree and moves on
        if len(subset_target_values) == 1:
            decision_tree['split_components'][column_value] = subset_target_values[0]

        # Recursively build the decision tree
        else:
            subset_data = subset.drop(columns=[goat_column])
            subset_header_list = list(header_list.copy())
            subset_header_list.remove(goat_column)
            decision_tree['split_components'][column_value] = build_decision_tree(subset_data, target_name,
                                                                                  subset_header_list)
    return decision_tree


# Run method
def column():
    df_headers = col_names

    df = planets
    target_name = 'P_HABITABLE'

    decision_tree = build_decision_tree(df, target_name, df_headers)
    np.array(decision_tree)
    # a = ''
    # for i in decision_tree:
    #    if decision_tree[i] == 'P_FLUX':
    #       a += 'P_FLUX'
    # print('Best column = ', bc)
    return bc

    # print_decision_tree(decision_tree)

    # 48300 27600
