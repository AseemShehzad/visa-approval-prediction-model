import gradio as gr
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from utils import standardize_strings, find_ranks

def load_model_files():
    """ Load the necessary model files such as encoder, red_list, and model """
    with open('inference_files/onehot_encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)
    with open('inference_files/red_listed_unis.pkl', 'rb') as file:
        red_list = pickle.load(file)
    model = tf.saved_model.load('inference_files/model')
    return encoder, red_list, model

### Define the function for processing inputs ###
def process_inputs(university_name, sponsor, relatives, program, scholarship, degree_level, visa_attempt, encoder, red_list):
    """ Process the inputs and return a dataframe
    Parameters:
    university_name (str): Name of the university
    sponsor (str): Sponsor of the student
    relatives (str): Relatives of the student
    program (str): Program of the student
    scholarship (str): Scholarship of the student
    degree_level (str): Degree level of the student
    visa_attempt (str): Visa attempt of the student
    encoder (OneHotEncoder): OneHotEncoder object
    red_list (list): List of red listed universities
    Returns:
    input_df (pd.DataFrame): Processed dataframe
    """
    # Load the university ranking database
    uni_data_dw = pd.read_csv(r"data\National Universities Rankings.csv", encoding='ANSI')
    uni_ranks = pd.DataFrame({'University': uni_data_dw['Name'], 'Rank': uni_data_dw['Rank']})

    # Find the rank of the university
    _, rank, _ = find_ranks(university_name, uni_ranks, 'University', 'Rank')

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

    # Add Redlist column
    input_df['Red List'] = np.where(input_df['University'].isin(red_list), 1, 0)
    input_df.drop(['University'], axis=1, inplace=True)

    # Bin the ranks
    input_df['Rank'] = pd.cut(input_df['Rank'], bins=[0, 100, 200, 300, 400, 500, 2000], labels=[1, 2, 3, 4, 5, 6])

    # Define the categorical columns
    cat_columns = ['Program', 'Scholarship', 'Sponsor', 'Relatives']

    # Encode the categorical columns
    encoded_cols = encoder.get_feature_names_out(cat_columns)
    print(encoded_cols)
    encoded_df = pd.DataFrame(encoder.transform(input_df[cat_columns]), columns=encoded_cols)
    input_df = pd.concat([input_df, encoded_df], axis=1)
    print(input_df.columns)
    #cat_columns.append('Sponsor')
    input_df.drop(cat_columns, axis=1, inplace=True)

    input_df = (input_df)
    input_df = tf.convert_to_tensor(input_df,  dtype=tf.float32)

    return input_df

### Inferencing
def predict_prob(university_name, sponsor, relatives, program, scholarship, degree_level, visa_attempt, loader=load_model_files):
    """ Predict the probability of getting a visa """
    encoder, red_list, model = loader()
    input_df = process_inputs(university_name, sponsor, relatives, program, scholarship, degree_level, visa_attempt, encoder, red_list)
    predictions = model.signatures['serving_default'](input_df)
    print(predictions)
    prediction_bin = np.round(predictions['dense_19'].numpy()[0][0])
    return np.round(predictions['dense_19'].numpy()[0][0] * 100, 1)

## Define the Gradio interface

with gr.Blocks(theme=gr.themes.Soft()) as interface:
    with gr.Row(equal_widths=True):
        gr.Markdown("## US Visa Prediction", align="center")
    with gr.Row(equal_widths=True):
        gr.Markdown("### This app predicts the probability of getting a US visa", align="center")
    with gr.Row(equal_widths=True):
        with gr.Column(width=1):
            # University name
            university_name = gr.Text(type="text", label="University Name", placeholder="Enter University Name", scale=0)
        with gr.Column(width=1):
            # Sponsor
            sponsor = gr.Dropdown(choices=['Other', 'Family', 'Loan', 'Self', 'University', 'Employer'], label="Sponsor", placeholder="Who's sponsoring you?", scale=0)

    with gr.Row(equal_widths=True):
        # Relatives
        relatives = gr.Dropdown(choices=["No", "Sibling(s)", "Relative(s)", "Parents"], label="Relatives", placeholder="Do you have amy relatives in US?")

        # Program
        program = gr.Dropdown(choices=["Business Studies", "CS", "IT", "Engineering", "Natural Sciences", "Other"], label="Program", placeholder="What is program of stufy in the US?")

    with gr.Row(equal_widths=True):
    # Scholarship
        scholarship = gr.Dropdown(choices=["Full/Assistantship", "Yes/Partial", "No"], label="Scholarship", placeholder="Do you have any scholarship?")

        # Degree level
        degree_level = gr.Dropdown(choices=["Masters", "PhD", "Bachelors", "Community College"], label="Degree Level", placeholder="What is your degree level?")
    with gr.Row():
        # Visa attempt
        visa_attempt = gr.Dropdown(choices=["First", "Second", "Third", "Fourth or more"], label="Visa Attempt", placeholder="How many times have you applied for visa before?")

    with gr.Row(align="center"):
        with gr.Column(width=1):
            click_button = gr.Button("Predict")        

    # Output
    pred_prob = gr.Textbox(label="Probability of getting a visa")
    
    click_button.click(fn=predict_prob, inputs=[university_name, sponsor, relatives, program, scholarship, degree_level, visa_attempt], outputs=pred_prob)

interface.launch(debug=True, width="70%", show_api=False)
