import gradio as gr
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from utils import standardize_strings, find_ranks

### Define the function for processing inputs ###
def process_inputs(university_name, sponsor, relatives, program, scholarship, degree_level, visa_attempt):
    # Load the university ranking database
    uni_data_dw = pd.read_csv(r"data\National Universities Rankings.csv", encoding='ANSI')
    uni_ranks = pd.DataFrame({'University': uni_data_dw['Name'], 'Rank': uni_data_dw['Rank']})

    # Find the rank of the university
    uni_rank, rank, query = find_ranks(university_name, uni_ranks, 'University', 'Rank')

    # Create a pandas dataframe and store the inputs
    input_df = pd.DataFrame({
        "Visa Attempt": visa_attempt,
        "Program": program,
        "Degree Level": degree_level,
        "University": university_name,
        "Sponsor": sponsor,
        "Scholarship": scholarship,
        "Relatives": relatives,
        "Rank": rank
    }, index=[0])

    # Standardize the inputs using the standardize_strings function
    # Standardize Visa Attempt
    column = 'Visa Attempt'
    attempt_search_list = ['st', 'nd', 'rd', 'th']
    attempt_replace_list = [1, 2, 3, 4]
    input_df = standardize_strings(input_df, attempt_search_list, attempt_replace_list, column, '2')

    # Standardize Degree Level
    column = 'Degree Level'
    degree_search_list = ['Masters', 'PhD', 'Bachelors', 'Community College']
    degree_replace_list = [3, 4, 2, 1]
    input_df = standardize_strings(input_df, degree_search_list, degree_replace_list, column, 3)

    # Load the Red List
    with open('inference_files/red_listed_unis.pkl', 'rb') as file:
        red_list = pickle.load(file)

    # Add Redlist column
    input_df['Red List'] = np.where(input_df['University'].isin(red_list), 1, 0)
    input_df.drop(['University'], axis=1, inplace=True)

    # Define the categorical columns
    cat_columns = ['Program', 'Scholarship', 'Sponsor','Relatives']

    with open('inference_files/onehot_encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)

    # Encode the categorical columns
    input_df = pd.concat([input_df, pd.DataFrame(encoder.transform(input_df[cat_columns]), columns=encoder.get_feature_names_out(cat_columns))], axis=1)
    cat_columns.append('Sponsor')
    input_df.drop(cat_columns, axis=1, inplace=True)

    # Normalize the data using the saved standard_scaler
    with open("inference_files/scaler.pkl", 'rb') as file:
        scaler = pickle.load(file)

    input_df = scaler.transform(input_df)
    input_df = tf.convert_to_tensor(input_df)

    return input_df

### Load model
model = tf.saved_model.load('inference_files/model')

### Inferencing
def predict_prob(university_name, sponsor, relatives, program, scholarship, degree_level, visa_attempt):
    input_df = process_inputs(university_name, sponsor, relatives, program, scholarship, degree_level, visa_attempt)
    return model.predict(input_df)

### Gradio Interface
# University name
university_name = gr.Text(type="text", label="University Name")

# Sponsor
sponsor = gr.Dropdown(["Self", "Family", "Loan", "University", "Other"], label="Sponsor")

# Relatives
relatives = gr.Dropdown(["No", "Siblings", "Relatives", "Parents"], label="Relatives")

# Program
program = gr.Dropdown(["Business Studies", "CS", "IT", "Engineering", "Natural Sciences", "Other"], label="Program")

# Scholarship
scholarship = gr.Dropdown(["Full/Assistantship", "Yes/Partial", "No"], label="Scholarship")

# Degree level
degree_level = gr.Dropdown(["Masters", "PhD", "Bachelors", "Community College"], label="Degree Level")

# Visa attempt
visa_attempt = gr.Dropdown(["First", "Second", "Third", "Fourth or more"], label="Visa Attempt")

inputs = [university_name, sponsor, relatives, program, scholarship, degree_level, visa_attempt]

# Gradio Interface
gr.Interface(fn=predict_prob, inputs=inputs, outputs='text').launch()
