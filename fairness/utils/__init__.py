import numbers

import pandas as pd


def make_attributes_binary(dataset: pd.DataFrame, attributes: list[str]) -> pd.DataFrame:
    for attribute in attributes:
        if _is_attribute_binary(dataset[attribute]):
            max_value: dataset[attribute].dtype = dataset[attribute].max()
            dataset[attribute]: pd.Series = (dataset[attribute] == max_value).astype(int)
        else:
            threshold: float = _return_threshold_for_series(dataset[attribute])
            dataset[attribute]: pd.Series = (dataset[attribute] > threshold).astype(int)
    return dataset


def _is_attribute_binary(attribute: pd.Series) -> bool:
    return len(attribute.unique()) == 2


def _return_threshold_for_series(attribute: pd.Series) -> float:
    return (attribute.max() + attribute.min()) / 2


def _return_combination_list(unique_values_for_combination_attributes: list[numbers.Number]) \
        -> list[list[numbers.Number]]:
    pass


def return_combination_list(dataset: pd.DataFrame, protected_attributes: list[str], output_column: str) -> list:
    combination_attributes: list[str] = protected_attributes + [output_column]
    unique_values_for_combination_attributes: list = [dataset[x].unique() for x in combination_attributes]

    combination_list: list[list[numbers.Number]] = _return_combination_list(unique_values_for_combination_attributes)
    pass
