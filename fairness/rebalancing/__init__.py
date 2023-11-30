import random
from fairness.utils import *


def augment_data(dataset: pd.DataFrame, protected_attributes: list[str], output_column: str) -> pd.DataFrame:
    combination_list: list[Union[list, float]] = (
        return_combination_list_for_combination_attributes(dataset=dataset, protected_attributes=protected_attributes,
                                                           output_column=output_column))

    combination_attributes: list[str] = protected_attributes + [output_column]
    combination_frequency: list[int] = return_combinations_frequency(dataset=dataset,
                                                                     combination_attributes=combination_attributes,
                                                                     combination_list=combination_list)
    combination_frequency_target: int = return_combination_frequency_target(combination_frequency_list=combination_frequency)

    for index, combination in enumerate(combination_list):
        temp_dataset: pd.DataFrame = dataset.query(return_query_for_dataframe(combination, combination_attributes))
        if len(temp_dataset) == 0:
            temp_dataset = dataset
        if combination_frequency[index] == combination_frequency_target:
            continue

        new_rows = []
        for _ in range(combination_frequency_target - combination_frequency[index]):
            new_row: dict = {attribute: value for (attribute, value) in zip(combination_attributes, combination)}
            for attribute in dataset.columns.difference(combination_attributes):
                first_value, second_value = return_attribute_frequency(attribute, temp_dataset)
                min_value, max_value = min(first_value, second_value), max(first_value, second_value)
                if is_variable_discrete(temp_dataset, attribute):
                    new_row[attribute] = random.randint(min_value, max_value)
                else:
                    new_row[attribute] = random.uniform(min_value, max_value)
            new_rows.append(new_row)

        dataset = pd.concat([dataset, pd.DataFrame(new_rows)], ignore_index=True)

    return dataset
