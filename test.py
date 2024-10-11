import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Define a function to process the data (OLS model fitting)
def fit_model(df, formula):
    model = smf.ols(formula, data=df).fit()
    return model

# Define a function to dynamically build formulas from the target and predictors
def build_formula(target, predictors):
    predictors_formula = '+'.join([f'C({col})' if df[col].dtype == 'object' else col for col in predictors])
    return f'{target}~{predictors_formula}'

st.title('Testing For Final Project')

col1, col2 = st.columns(2)

with col1:
    # File upload on the left
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read CSV into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Display a limited preview of the DataFrame (first 5 rows)
        st.write("Data Preview (first 5 rows):")
        st.dataframe(df.head(5))

with col2:
    if uploaded_file is not None:
        # Dropdown menu on the right
        response = st.selectbox('Select a response variable: ', df.columns)
        predictors = st.multiselect('Select predictor variables: ', df.columns)

        #saving each models
        if 'models' not in st.session_state:
            st.session_state.models = []

        if st.button('Add Model'):
            if predictors and response:
                # Build the formula and fit the model
                formula = build_formula(response, predictors)
                model = fit_model(df, formula)
                
                # Store the model and its formula
                st.session_state.models.append({'formula': formula, 'model': model})
                st.success(f'Model added with formula: {formula}')
            else:
                st.error('Please select both target and predictors.')

        # If models are created, show them in a selectbox to choose which model to evaluate
        if len(st.session_state.models) > 0:
            st.write("### Models Created:")
        
        # Display the formulas of the created models
        for i, m in enumerate(st.session_state.models):
            st.write(f"Model {i+1}: {m['formula']}")
        
        # Dropdown to select a model to evaluate
        model_index = st.selectbox('Select a model to evaluate', range(len(st.session_state.models)),
                                   format_func=lambda x: st.session_state.models[x]['formula'])

        # Show the summary of the selected model
        if st.button('Evaluate Model'):
            selected_model = st.session_state.models[model_index]['model']
            st.text(selected_model.summary().as_text())

        # Perform calculation on the selected column
        # if st.button('Calculate'):
        #     if response and predictors:
        #         if response in predictors:
        #             st.write("You cannot include a response variable as a predictor variable.")
        #         else:
        #             try:
        #                 model_summary = process_data(df, response, predictors)
        #                 st.text(model_summary)
        #             except Exception as e:
        #                 st.error(f"Error in fitting the model: {str(e)}")
        #     else:
        #         st.write("Please select at least one column.")


