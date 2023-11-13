import pandas as pd
import numpy as np
import re

def standardize_strings(df, search_list, replace_list, column, replacement):
    # Check if search and replace lists have equal length
    if len(search_list) != len(replace_list):
        print("The length of search and replace lists are not equal!!!")
        return df

    # Create a copy of the dataframe
    df = df.copy()
    
    # Convert the column to string
    df[column] = df[column].astype(str)

    # Perform the replacements
    for search_string, replace_string in zip(search_list, replace_list):
        # Using str.contains for vectorized operation
        mask = df[column].str.contains(search_string, case=False, na=False)
        df.loc[mask, column] = replace_string

    # Find values not in replace_list and replace them
    not_found_mask = ~df[column].isin(replace_list)
    if not_found_mask.any():
        not_found_list = df.loc[not_found_mask, column].unique()
        df.loc[not_found_mask, column] = replacement
        print(f"The following values were not found in the 'replace_list': {not_found_list}")
    else:
        print(f"All values in {column} are standardized.")

    return df

### Creating a Search_and_Replace function
from fuzzywuzzy import fuzz
def find_ranks(query, my_df, search_col_name, result_col_name):
    
    # Define the threshold for the fuzzy score minimum score
    threshold = 60

    # Create an empty dictionary to store the results
    results_dict = {}

    # Loop through each string in the list
    for string in my_df[search_col_name]:
        # Calculate the Levenshtein distance between the search term and the current string
        score = fuzz.token_sort_ratio(query, string)

        # If the score is above the threshold, add the string and score to the dictionary
        if score >= threshold:
            results_dict[string] = score

    # Sort the dictionary by score, in descending order
    sorted_results = sorted(results_dict.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_results) == 0:
        ## In case searc result is not found, specficy the output here
        ### print("Nothing found")
        result = 1000
        query_final = "N/A"
        
    else:
        # Search the index first element of sorted results through search col of my_df
        query_final = sorted_results[0][0]
        query_final_escaped = re.escape(query_final)
        index = my_df[my_df[search_col_name].str.contains(query_final_escaped)].index

        # Check if the index exists before returning the result
        if not index.empty:
            result = my_df.loc[index[0], result_col_name]
        else:
            result = "N/A"

    return sorted_results, result, query_final

### function to encode the scholarship column
def encode_scholarship(df):
    df['Scholarship'] = np.where(df['Scholarship'] == 'No', 0, df['Scholarship'])
    df['Scholarship'] = np.where(df['Scholarship'] == 'Yes', 1, df['Scholarship'])
    df['Scholarship'] = np.where(df['Scholarship'] == 'Assistantship', 2, df['Scholarship'])
    return df