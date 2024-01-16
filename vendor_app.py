import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.subheader("📗 Refresh FB Creatives From Your Plutus Spreadsheet")

url = "https://docs.google.com/spreadsheets/d/1tqm7G0yzckwSCKXdPcGcWNH6y5nMj68rhpMQZlcO2wU/edit#gid=1913725997"

st.write("#### 1. Retrieved Plutus Creatives")

# Create a connection using Streamlit's experimental connection feature
conn = st.experimental_connection("gsheets", type=GSheetsConnection)

# Read data from the Google Sheet
df = conn.read(worksheet="PlutusDataImport", ttl=10)
desired_range = df.iloc[99:124, 0:2]  # Rows 100-124 and columns A-B (0-indexed)

# Hardcoding specific values in column A
desired_range.iloc[0:5, 0] = 'Headlines'      # Rows 100-104
desired_range.iloc[6:11, 0] = 'Primary Text'  # Rows 106-110
desired_range.iloc[12:17, 0] = 'Description'  # Rows 112-116
desired_range.iloc[19:24, 0] = 'Forcekeys'    # Rows 119-123

# Replace NaN values with an empty string
desired_range.fillna('', inplace=True)

# Rename the columns
desired_range.columns = ['Asset Type', 'Creative Text']

# Store values, omitting nulls
headlines = [text for text in desired_range[desired_range['Asset Type'] == 'Headlines']['Creative Text'] if text]
primary_text = [text for text in desired_range[desired_range['Asset Type'] == 'Primary Text']['Creative Text'] if text]
descriptions = [text for text in desired_range[desired_range['Asset Type'] == 'Description']['Creative Text'] if text]
forcekeys = [text for text in desired_range[desired_range['Asset Type'] == 'Forcekeys']['Creative Text'] if text]

# Use st.expander to create a toggle for showing the full table
with st.expander("Show Full Table", expanded=True):
    # Use st.markdown with HTML and CSS to enable text wrapping inside the expander
    st.markdown("""
    <style>
    .dataframe th, .dataframe td {
        white-space: nowrap;
        text-align: left;
        border: 1px solid black;
        padding: 5px;
    }
    .dataframe th {
        background-color: #f0f0f0;
    }
    .dataframe td {
        min-width: 50px;
        max-width: 700px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: normal;
    }
    </style>
    """, unsafe_allow_html=True)

    # Display the DataFrame with text wrapping inside the expander
    st.markdown(desired_range.to_html(escape=False, index=False), unsafe_allow_html=True)

from openai import OpenAI

# User inputs their OpenAI API key in the sidebar
openai_api_key = st.secrets["openai_secret"]

# Initialize or update the session state for form count and responses
if 'form_count' not in st.session_state:
    st.session_state['form_count'] = 1
if 'responses' not in st.session_state:
    st.session_state['responses'] = []

# Function to replace dynamic keys in the prompt with actual values
def replace_dynamic_keys(prompt):
    prompt = prompt.replace('[headlines]', ', '.join(headlines))
    prompt = prompt.replace('[primary_text]', ', '.join(primary_text))
    prompt = prompt.replace('[descriptions]', ', '.join(descriptions))
    prompt = prompt.replace('[forcekeys]', ', '.join(forcekeys))
    return prompt

# Function to generate response using OpenAI API
def generate_response(system_prompt, user_prompt, model="gpt-4", temperature=0.00):
    client = OpenAI(api_key=openai_api_key)

    # Replace dynamic keys with actual values
    system_prompt = replace_dynamic_keys(system_prompt)
    user_prompt = replace_dynamic_keys(user_prompt)

    # Map the friendly model name to the actual model ID
    model_id = {
        'gpt-3.5-turbo': 'gpt-3.5-turbo',
        'gpt-4': 'gpt-4',
        # Add other models as needed
    }.get(model, model)  # Default to the provided model name if it's not in the dictionary

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content.strip('"')

# Function to add a new prompt
def add_prompt():
    st.session_state['form_count'] += 1

# Function to remove the latest prompt
def remove_prompt():
    if st.session_state['form_count'] > 1:
        st.session_state['form_count'] -= 1

# Buttons to add or remove a prompt
col1, col2 = st.columns(2)
with col1:
    st.button('Add Prompt', on_click=add_prompt)
with col2:
    st.button('Remove Prompt', on_click=remove_prompt)

# Create expanders for each set of inputs
for i in range(st.session_state['form_count']):
    with st.expander(f"Chain Link {i+1}", expanded=True):
        model = st.selectbox('OpenAI Model', 
                            ('gpt-3.5-turbo', 'gpt-4'), 
                            key=f'model_{i}')
        temperature = st.number_input('Temperature', min_value=0.00, max_value=1.00, value=0.00, key=f'temp_{i}')
        system_prompt = st.text_area('System Prompt:', key=f'system_{i}')
        user_prompt = st.text_area('User Prompt', key=f'user_{i}')

# Single submit button for all inputs
if st.button('Submit All'):
    if not openai_api_key:
        st.warning('Please enter your OpenAI API key!', icon='⚠️')
    else:
        st.session_state['responses'] = []
        for i in range(st.session_state['form_count']):
            # Retrieve the model and temperature specific to each form
            current_model = st.session_state[f'model_{i}']
            current_temperature = st.session_state[f'temp_{i}']

            # Get the current system and user prompts
            current_system_prompt = st.session_state[f'system_{i}']
            current_user_prompt = st.session_state[f'user_{i}']

            # Apply dynamic replacements to both system and user prompts
            for j in range(i):
                replacement_text = st.session_state['responses'][j]
                current_system_prompt = current_system_prompt.replace(f'[output {j+1}]', replacement_text)
                current_user_prompt = current_user_prompt.replace(f'[output {j+1}]', replacement_text)

            # Pass the specific model and temperature for each form
            response = generate_response(current_system_prompt, current_user_prompt, current_model, current_temperature)
            st.session_state['responses'].append(response)
            st.text(f"**Generated Response {i+1}:** \n\n{response}")
