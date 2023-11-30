from typing import Tuple, Union
import numpy as np
import pandas as pd
import itertools


def make_attributes_binary(dataset: pd.DataFrame, attributes: list[str]) -> pd.DataFrame:
    """
    This method converts the attributes passed as argument into binary ones according to a specific policy. More
    specifically if the attribute is already a binary one then the greater value is replaced with 1 and the lower one
    with 0.
    If the attribute is not yet a binary one then a threshold is computed and according to this threshold the values are
    replaced with either 1 or 0
    :param dataset: the dataset from which retrieve the value
    :param attributes: the attributes to convert into binary ones
    :return: the dataset with the attributes specified as argument converted into binary ones
    """
    for attribute in attributes:
        if is_attribute_binary(dataset[attribute]):
            max_value: dataset[attribute].dtype = dataset[attribute].max()
            dataset[attribute]: pd.Series = (dataset[attribute] == max_value).astype(int)
        else:
            threshold: float = return_threshold_for_series(dataset[attribute])
            dataset[attribute]: pd.Series = (dataset[attribute] > threshold).astype(int)
    return dataset


def is_attribute_binary(attribute: pd.Series) -> bool:
    """
    This method checks if the series passed as argument is a binary one
    :param attribute: the attribute to assess if it's either binary or not
    :return: the results of the check
    """
    return len(attribute.unique()) == 2


def return_threshold_for_series(attribute: pd.Series) -> float:
    """
    This method returns the threshold for the attribute specified as argument used into the binary conversion of the attribute
    :param attribute: the series on which compute the threshold
    :return: return the computed threshold
    """
    return (attribute.max() + attribute.min()) / 2


def return_combination_list(unique_values_for_combination_attributes: list[Union[int, float]]) \
        -> list[list[Union[int, float]]]:
    """
    This method returns the combination list for the list of lists specified as argument. More specifically each list
    belonging to the return list has in its i-th position an element of the i-th list in the list of lists
    :param unique_values_for_combination_attributes: the list of lists from which to build the combination list
    :return: the combination list
    """
    return list(itertools.product(*unique_values_for_combination_attributes))


def return_query_for_dataframe(combination: list[Union[int, float]], combination_attributes: list[str]) -> str:
    """
    This method returns the query string to make a search on the original dataset
    :param combination: the combination representing the values the combination attributes must have in the sub-dataset
    :param combination_attributes: the combination attributes
    :return: a string to be passed to the .query method of DataFrame object
    """
    conditions = [f"{attribute} == {value}" for value, attribute in zip(combination, combination_attributes)]
    return " and ".join(conditions)


def return_combination_list_for_combination_attributes(dataset: pd.DataFrame, protected_attributes: list[str],
                                                       output_column: str) -> list[list[Union[int, float]]]:
    """
    This method returns the combination list for the combination attributes intended as combination of
    protected attributes and output column
    :param dataset: the dataset on which start to retrieve the unique values
    :param protected_attributes: the protected attribute
    :param output_column: the output column
    :return: the combination list for the combination attribute list
    """
    combination_attributes: list[str] = protected_attributes + [output_column]
    unique_values_for_combination_attributes: list = [dataset[x].unique() for x in combination_attributes]

    combination_list: list[list[Union[int, float]]] = return_combination_list(unique_values_for_combination_attributes)
    return combination_list


def return_combinations_frequency(dataset: pd.DataFrame, combination_attributes: list[str],
                                  combination_list: list[list[Union[int, float]]]) -> list[int | float]:
    """
    This method returns the combination frequency on the dataset for each element in combination list
    :param dataset: the dataset on which compute the frequencies
    :param combination_attributes: the combination attributes to be marched with the element of the combination list
    :param combination_list: the combination list
    :return: the frequency into the dataset for each element of the combination list
    """
    return [len(dataset.query(return_query_for_dataframe(combination, combination_attributes))) for combination in
            combination_list]


def return_combination_frequency_target(combination_frequency_list: list) -> Union[int, float]:
    return max(combination_frequency_list)


def is_attribute_continuous(dataset: pd.DataFrame, attribute: str) -> bool:
    return all(isinstance(attribute, float) for attribute in dataset[attribute])


def is_variable_discrete(dataset: pd.DataFrame, attribute: str) -> bool:
    return all(isinstance(attribute, int) for attribute in dataset[attribute])


def return_attribute_frequency(attribute: str, dataset: pd.DataFrame) -> Tuple[Union[int, float], Union[int, float]]:
    unique_values, counts = np.unique(dataset[attribute], return_counts=True)

    if len(unique_values) == 0:
        return 1, 0

    most_frequent_value = unique_values[np.argmax(counts)]
    least_frequent_value = unique_values[np.argmin(counts)]

    return most_frequent_value, least_frequent_value

