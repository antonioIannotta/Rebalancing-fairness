import pandas as pd


def make_attributes_binary(dataset: pd.DataFrame, attributes: list[str]) -> pd.DataFrame:
    for attribute in attributes:
        if is_attribute_binary(dataset[attribute]):
            max_value = dataset[attribute].max()
            pass
    pass


def is_attribute_binary(attribute: pd.Series) -> bool:
    return len(attribute.unique()) == 2

