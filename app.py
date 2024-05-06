import gradio as gr
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os
from utils import standardize_strings, find_ranks
import socket
from pathlib import Path

hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)
static_dir = Path('./static')
static_dir.mkdir(parents=True, exist_ok=True)

detail_html_path = static_dir / "detail.html"

def load_model_files():
    """ Load the necessary model files such as encoder, red_list, and model """
    with open(os.path.join('inference_files', 'onehot_encoder.pkl'), 'rb') as file:
        encoder = pickle.load(file)
    with open(os.path.join('inference_files', 'red_listed_unis.pkl'), 'rb') as file:
        red_list = pickle.load(file)
    model = tf.saved_model.load(os.path.join('inference_files', 'model'))
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
    uni_data_dw = pd.read_csv(os.path.join("data", "National Universities Rankings.csv"), encoding='windows-1252')
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

    input_df = standardize_strings(input_df, attempt_search_list, attempt_replace_list, column, 2)

    # Standardize Degree Level
    column = 'Degree Level'
    degree_search_list = ['Masters', 'PhD', 'Bachelors', 'Community College']
    degree_replace_list = [3, 4, 2, 1]
    input_df = standardize_strings(input_df, degree_search_list, degree_replace_list, column, 3)

    # Add Redlist column
    input_df['Red List'] = np.where(input_df['University'].isin(red_list), 1, 0)
    input_df.drop(['University'], axis=1, inplace=True)

    # Convert the columns to numeric
    input_df['Visa Attempt'] = pd.to_numeric(input_df['Visa Attempt'])
    input_df['Degree Level'] = pd.to_numeric(input_df['Degree Level'])
    input_df['Rank'] = pd.to_numeric(input_df['Rank'])

    # Bin the ranks
    input_df['Rank'] = pd.cut(input_df['Rank'], bins=[0, 100, 200, 300, 400, 500, 2000], labels=[1, 2, 3, 4, 5, 6])

    # Define the categorical columns
    cat_columns = ['Program', 'Scholarship', 'Sponsor', 'Relatives']

    # Encode the categorical columns
    encoded_cols = encoder.get_feature_names_out(cat_columns)
    encoded_df = pd.DataFrame(encoder.transform(input_df[cat_columns]), columns=encoded_cols)
    input_df = pd.concat([input_df, encoded_df], axis=1)
    input_df.drop(cat_columns, axis=1, inplace=True)

    input_df = tf.convert_to_tensor(input_df,  dtype=tf.float32)

    return input_df

### Inferencing
def predict_prob(university_name, sponsor, relatives, program, scholarship, degree_level, visa_attempt, loader=load_model_files):
    """ Predict the probability of getting a visa """
    encoder, red_list, model = loader()
    input_df = process_inputs(university_name, sponsor, relatives, program, scholarship, degree_level, visa_attempt, encoder, red_list)
    predictions = model.signatures['serving_default'](input_df)
    output_tensor_name = list(predictions.keys())[0]  
    output_tensor = predictions[output_tensor_name]
    prediction_value = np.round(output_tensor.numpy()[0][0] * 100, 1)
    return prediction_value

def generate_random_input():
    """ Generate random input values for testing """
    university_name = "University of Cincinnati"
    sponsor = np.random.choice(['Other', 'Family', 'Loan', 'Self', 'University', 'Employer'])
    relatives = np.random.choice(["No", "Sibling(s)", "Relative(s)", "Parents"])
    program = np.random.choice(["Business Studies", "CS", "IT", "Engineering", "Natural Sciences", "Other"])
    scholarship = np.random.choice(["Full/Assistantship", "Yes/Partial", "No"])
    degree_level = np.random.choice(["Masters", "PhD", "Bachelors", "Community College"])
    visa_attempt = np.random.choice(["First", "Second", "Third", "Fourth or more"])
    
    return university_name, sponsor, relatives, program, scholarship, degree_level, visa_attempt

# Test the function
university_name, sponsor, relatives, program, scholarship, degree_level, visa_attempt = generate_random_input()
probability = predict_prob(university_name, sponsor, relatives, program, scholarship, degree_level, visa_attempt)

# Load the universities database
uni_data_dw = pd.read_csv(os.path.join("data", "National Universities Rankings.csv"), encoding='windows-1252')
uni_ranks = pd.DataFrame({'University': uni_data_dw['Name'], 'Rank': uni_data_dw['Rank']})

## Define the Gradio interface
with gr.Blocks(title="US Visa Prediction App") as interface:
    gr.Markdown("<div style='text-align: center; font-size: 2rem;'>US Student Visa Prediction</div>")
    gr.Markdown("<div style='text-align: center;'>This application is created solely for the purpose of helping others. Everything about this is transparent. The creator of the app seeks no financial benefit in any shape or form, including visa consulting.")
    gr.Markdown("<div style='text-align: center;'><a href = ./detail.html Click here </a> for learning about the prediction process.</div>")

# University name
    university_name = gr.Dropdown(choices=list(uni_ranks['University'].unique()), label="University Name", info="Which university are you applying visa for?")    
    with gr.Row():
        
        # Sponsor
        sponsor = gr.Dropdown(choices=['Other', 'Family', 'Loan', 'Self', 'University', 'Employer'], label="Sponsor", info="Who is paying for your studies?")
        # Relative
        relatives = gr.Dropdown(choices=["No", "Sibling(s)", "Relative(s)", "Parents"], label="Relatives", info="Do you have any relatives in the US?")
    with gr.Row():
        # Program
        program = gr.Dropdown(choices=["Business Studies", "CS", "IT", "Engineering", "Natural Sciences", "Other"], label="Program", info = "What is your field of study?")

        # Scholarship
        scholarship = gr.Dropdown(choices=["Full/Assistantship", "Yes/Partial", "No"], label="Scholarship", info="Did you get any scholarship from the university?")
    with gr.Row():
        # Degree level
        degree_level = gr.Dropdown(choices=["Masters", "PhD", "Bachelors", "Community College"], label="Degree Level", info="Which degree are you applying for?")

        # Visa attempt
        visa_attempt = gr.Dropdown(choices=["First", "Second", "Third", "Fourth or more"], label="Visa Attempt", info="Which visa attempt is this?")
    
    click_button = gr.Button("Predict")
    pred_prob = gr.Textbox(label="Probability of Visa Approval")
    gr.Markdown("<div style='text-align: center;'>Contact at <a href='mailto:aseemshehzad10@gmail.com'>aseemshehzad10@gmail.com</a> for any ideas/questions about the application.</div>")
    
    click_button.click(fn=predict_prob, inputs=[university_name, sponsor, relatives, program, scholarship, degree_level, visa_attempt], outputs=pred_prob)

interface.launch(debug=True, show_api=False)