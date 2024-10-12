import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.set_page_config(page_title="MLR: The Model Selection", page_icon=":✅:", layout="wide")

if 'show_step3' not in st.session_state:
    st.session_state.show_step3 = False
if 'chosen_metrics' not in st.session_state:
    st.session_state.chosen_metrics = []
if 'show_step4' not in st.session_state:
    st.session_state.show_step4 = False

page = st.sidebar.selectbox("Pages", ["Blog Post", "Model Selection App"])

if page == "Blog Post":
    st.title("✅ MLR: The Model Selection")


    st.header("Introduction")
    st.markdown("""<hr style='border: 2px solid #4CAF50; margin: 0; padding: 0;'>""", unsafe_allow_html=True)

    st.markdown("""
    Selecting the right statistical model for a dataset can be a daunting task, particularly when faced with a wide range of options and evaluation criteria. 
    Many factors come into play, such as:
    - How well a model fits the data.
    - Whether its assumptions hold.
    - Its performance across different metrics.

    **Data scientists often struggle** to make informed decisions efficiently. Our solution to this problem is a user-friendly platform that **simplifies model selection** by comparing the user's selected model with the statistically optimal model based on metrics like:
    - Akaike Information Criterion (AIC)
    - Bayesian Information Criterion (BIC)
    - Adjusted R²
    - Mallows’ Cp
    - Hannan-Quinn Criterion (HQC)

    In addition, the platform provides insights into model performance through **error plots** and **diagnostic checks** for issues like **multicollinearity**, helping users understand and have confidence in their final choices.
    """)
    st.markdown("""<hr style='border: 2px solid #E0E0E0; margin: 20px 0;'>""", unsafe_allow_html=True)

    st.subheader("🔍 Motivation")
    st.markdown("""
    The primary motivation behind this tool stems from the common frustrations practitioners face when navigating the model selection process:
    - Ensuring **accuracy**, **interpretability**, and **reliability** can be challenging, especially for those new to data science or machine learning.
    - Many struggle to navigate the maze of **statistical evaluations** and **theoretical assumptions**.

    This challenge is particularly apparent when comparing models based on criteria like AIC or BIC and interpreting results without over-reliance on p-values.

    Our goal is to:
    - **Automate model comparison** to simplify decision-making.
    - Provide clear explanations and visualizations to help users understand key metrics.
    - Offer insights into model diagnostics to check whether assumptions like **multicollinearity** and **homoskedasticity** hold.

    By lowering the barrier to effective model selection, we hope to make the process more transparent and accessible to both beginners and experienced data scientists alike.
    """)


    st.header("MLR And The Model Selection Process")
    st.markdown("""<hr style='border: 2px solid #4CAF50; margin: 0; padding: 0;'>""", unsafe_allow_html=True)

    st.markdown("""
    With a given dataset, a **response variable** (our target outcome) and a set of **predictor variables** (features), our goal is to construct a model that can accurately predict the response variable. Specifically, we aim to determine which predictor variables are **significant** in predicting the response variable, \( y \).
    For our model, we’ll focus on **Multiple Linear Regression (MLR)**.
    """)

    # Section Header: What is MLR?
    st.subheader("📈 What is MLR?")
    st.markdown("""
        **Multiple Linear Regression (MLR)** is a statistical technique used to model the relationship between one dependent variable (response) and two or more independent variables (predictors). 
        """)

    st.latex(r"""
    y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \epsilon
    """)

    # Explanation of variables
    st.write("""
    Where:
    - (y) is the response variable.
    - (x1, x2, ... , xn) are the predictor variables.
    - (beta_0) is the intercept.
    - (beta_1, beta_2, ... , beta_n) are the coefficients that represent the impact of each predictor.
    - Epsilon(E) is the error term (representing unexplained variance).
    """)

    st.write("""
    The goal of MLR is to estimate the coefficients (beta_1, beta_2, ... , beta_n), 
    in such a way that the sum of the squared residuals (the differences between the observed and predicted \( y \)) is minimized.
    """)
    # Divider
    st.markdown("""<hr style='border: 1px solid #E0E0E0;'>""", unsafe_allow_html=True)

    col1, col2 = st.columns([5, 2])
    with col1:
        # Section Header: Before Diving Into the Model Selection Process...
        st.subheader("🛠️ Before Diving Into The Model Selection Process...")

        st.markdown("""
        In order to determine the best MLR model to use to accurately predict our response variable, we will use the **Best Subsets Regression** process.

        #### What is Best Subsets Regression?

        **Best Subsets Regression** is a method used to select the best combination of predictor variables for a given model. It explores all possible combinations of the predictor variables to identify which subset gives the best predictive performance, based on criteria like:
        - **Adjusted R-squared**
        - **Mallows Cp**
        - **AIC** (Akaike Information Criterion)
        - **BIC** (Bayesian Information Criterion)
        - **PRESS** (Prediction Sum Of Squared Error)
        - **HQC** (Hannan-Quinn Criterion)

        This process is especially useful when there are many potential predictors, allowing us to find the combination that provides the best fit without overfitting.
        """)

        st.markdown("""
        Next, we will go through the **model selection process** with examples to illustrate how to find the best MLR model for our dataset.
        """)
    with col2:
        st.image("https://quantifyinghealth.com/wp-content/uploads/2020/09/best-subset-selection-example-with-3-variables.png", caption="Best Subset Regression Illustration", use_column_width=True)
    # Divider
    st.markdown("""<hr style='border: 1px solid #E0E0E0;'>""", unsafe_allow_html=True)

    st.subheader("📈 The Model Selection Process Using Best Subsets Regression")
    st.markdown("""
        Before proceeding with best subsets regression, we must decide our response variable and our predictor.
        """)
        
    st.info(
        """
        To help out with the understanding, IceCreamConsumption.csv data will be used for illustrations.
        """,
        icon="🔴",
    )
    
    col1, col2 = st.columns([1, 1])
    icecream = pd.read_csv('./data/IceCreamConsumption.csv')
    with col1:
        st.markdown("""Let's take a look at the Ice Cream Consumption Data: """)
        st.dataframe(icecream.head(5))



elif page == "Model Selection App":
    # Cool introductory section
    st.title("📊 Best Subset Regression App")
    st.markdown("""
    Welcome to the **Best Subset Regression App**! This interactive app allows you to:

    1. **Upload your own dataset**.
    2. **Select a response variable** and ***predictor variables*** from your dataset.
    3. **Automatically run best subsets regression** to find the optimal combinations of predictor variables.

    ---

    The app evaluates all possible subsets of predictors and presents the best models according to multiple metrics, including:
    - **Adjusted R-squared**
    - **Mallows Cp**
    - **AIC** (Akaike Information Criterion)
    - **BIC** (Bayesian Information Criterion)
    - **PRESS** (Prediction Sum Of Squared Error)
    - **HQC** (Hannan-Quinn Criterion)

    You will receive the **best model suggestions**, but it's **up to you** to determine which model is the best fit for your analysis based on the metrics provided.

    Explore different models and make an informed decision about which subset of predictors works best for your data!
    """)

    st.header("🖍Step 1: Upload your data")
    st.markdown("""<hr style='border: 2px solid #92939c; margin: 0; padding: 0;'>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.replace('-', '_')  # Replace hyphens with underscores

        st.write("Data Preview (first 5 rows):")
        st.dataframe(df.head(5))

        st.header("🖍Step 2: Choose your response and predictor variables:")
        st.markdown("""<hr style='border: 2px solid #92939c; margin: 0; padding: 0;'>""", unsafe_allow_html=True)

        # Dropdown for the user to select the response variable
        response = st.selectbox('Select the response variable:', df.columns)
        predictors = st.multiselect('Select predictor variables: ', df.columns)
        
        st.write(f"The following variable is the response variable (y): {response}")
        st.write(f"The following variables are the predictor variables: {predictors}")

        if len(predictors) > 0:
            if st.button("Done"):
                st.session_state.show_step3 = True
        else:
            st.warning("Please select at least one predictor variable.")

    if st.session_state.show_step3:
        st.header("🖍Step 3: Best Subsets Regression")
        st.markdown("""<hr style='border: 2px solid #92939c; margin: 0; padding: 0;'>""", unsafe_allow_html=True)
        
        def best_subset_selection(X, y):
            n, p = X.shape
            models = []
    
            for k in range(1, p + 1):  # Iterate over subset sizes
                for combo in combinations(range(1, p), k):  # Generate combinations of predictors
                    combo = (0,) + combo  # Include the intercept
                    X_subset = X[:, combo]
                    model = sm.OLS(y, X_subset).fit()
                    models.append((model, combo))
            
            return models

        def get_cp(sse_k, mse_p, num_params, n):
            return (sse_k / mse_p) + 2 * num_params - n
        
        def calculate_metrics(model, X, y):
            n = len(y)
            k = model.df_model  # Number of predictors, excluding intercept

            # AIC
            aic = model.aic
            # BIC
            bic = model.bic
            # PRESS (Prediction Sum of Squares)
            hat_matrix = X @ np.linalg.inv(X.T @ X) @ X.T
            residuals = model.resid
            press = np.sum((residuals / (1 - np.diag(hat_matrix))) ** 2)
            # Adjusted R-squared
            r2 = model.rsquared
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
            log_likelihood = model.llf
            hqc = -2 * log_likelihood + 2 * (k+1) * np.log(np.log(n))   
            
            return aic, bic, press, adj_r2, hqc, int(k) #dont consider intercept as a predictor


        X = sm.add_constant(np.array(df[predictors]))
        y = np.array(df[response])
        models = best_subset_selection(X, y)
        full_model = sm.OLS(y, X).fit()
        sse_full = full_model.ssr
        n = len(y)
        p_full = X.shape[1]
        mse_full = sse_full / (n - p_full) 

        results = []
        for model, combo in models:
            predictors = []
            aic, bic, press, adj_r2, hqc, num_predictors = calculate_metrics(model, X[:, combo], y)
            for c in combo:
                if c != 0:
                    predictors.append(df.columns[c])
            sse_k = model.ssr
            num_params = len(combo)
            cp = get_cp(sse_k, mse_full, num_params, n)
            
            results.append({
                'Predictors': predictors,
                'n_Predictors': num_predictors,
                'Adjusted R^2': adj_r2,
                'Mallows Cp': cp,
                'AIC': aic,
                'BIC': bic,
                'PRESS': press,
                'HQC': hqc
            })
        
        # Convert results to pd DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='n_Predictors').reset_index(drop=True)

        # Create two columns: Make them equal-sized by setting each to occupy half the page
        col1, col2 = st.columns([1, 1])

        # Column 1: Display the results dataframe
        with col1:
            st.subheader("Values of each metric are calculated for each model below:")
            st.dataframe(results_df)

        # Column 2: Allow selection of multiple metrics for model comparison
        with col2:
            st.subheader("Select on the metric to check which model was selected as the best model:")
            # Available metrics
            available_metrics = ['Adjusted R^2', 'AIC', 'BIC', 'Mallows Cp', 'PRESS', 'HQC']

            # Initialize session state for selected metrics and models if not already done
            if 'selected_models' not in st.session_state:
                st.session_state.selected_models = {}

            # Multiselect for metrics
            chosen_metrics = st.multiselect('Select metrics to find the best models:', available_metrics)

            # Handle the display and removal of models based on selected metrics
            for metric in chosen_metrics:
                if metric not in st.session_state.selected_models:
                    # Logic to select the best model based on the metric
                    if metric == 'Adjusted R^2':
                        best_model_row = results_df.loc[results_df[metric].idxmax()]
                    elif metric == "Mallows Cp":
                        diff_ser = results_df['Mallows Cp'] - (results_df['n_Predictors'] + 1)[:-1]  # Exclude full model
                        best_model_row = results_df.loc[diff_ser.idxmin()]
                    else:
                        best_model_row = results_df.loc[results_df[metric].idxmin()]
                    
                    best_model_predictors = best_model_row['Predictors']
                    
                    # Store the selected model for the metric in session state
                    st.session_state.selected_models[metric] = f"The best model selected by {metric} is: y ~ {best_model_predictors}"
            
            # Remove the model from session state if the metric is deselected
            for metric in list(st.session_state.selected_models.keys()):
                if metric not in chosen_metrics:
                    del st.session_state.selected_models[metric]

            # Display the selected models
            for model in st.session_state.selected_models.values():
                st.write(model)
            
            if len(st.session_state.selected_models) > 0:
                if st.button("Done with comparison"):
                    st.session_state.show_step4 = True
    
    if st.session_state.show_step4:
        st.header("🖍Step 4: Checking Multicollinearity, Heteroscedasticity, and Influential Points")
        st.markdown("""<hr style='border: 2px solid #92939c; margin: 0; padding: 0;'>""", unsafe_allow_html=True)

        def plot_qq(model):
            # Plots the QQ-Plot
            qq_plot = ProbPlot(model.resid)
            
            # Create figure for the plot with smaller size (1/4 of the original size)
            fig, ax = plt.subplots(figsize=(6, 4))  # Adjust the size here

            fig.set_size_inches(5, 4)

            # Plot the Q-Q plot with customized markers and quartile line for proper visualization
            qq_plot.qqplot(ax=ax, line='q', markerfacecolor='blue', markeredgewidth=1, marker='o', alpha=0.7)

            # Set plot title and axis labels
            ax.set_title('Q-Q Plot of Residuals', fontsize=12)
            ax.set_xlabel('Theoretical Quantiles', fontsize=10)
            ax.set_ylabel('Sample Quantiles', fontsize=10)

            # Display the plot in Streamlit using st.pyplot
            st.pyplot(fig)

        def plot_residual_histogram(residuals):
            # Plots a histogram of residuals
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(residuals, kde=True, bins=20, ax=ax)
            ax.set_title('Histogram of Residuals')
            ax.set_xlabel('Residuals')
            ax.set_ylabel('Frequency')
            
            # Display the plot in Streamlit
            st.pyplot(fig)

        def plot_correlation_matrix(X):
            # Plots the correlation matrix
            fig, ax = plt.subplots(figsize=(10.8, 6))
            corr = X.corr()
            sns.heatmap(corr, annot=True, cmap='Blues', vmin=-1, vmax=1, ax=ax)
            ax.set_title('Correlation Heatmap')
            
            # Display the plot in Streamlit
            st.pyplot(fig)

        def plot_vif(X):
            # Calculates VIF for each feature
            vif_data = pd.DataFrame()
            vif_data['Feature'] = X.columns
            vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

            # Display the VIF data frame in Streamlit
            st.dataframe(vif_data)

        def check_homoskedasticity(ols_model):
            # Extract fitted values and residuals
            fitted_values = ols_model.fittedvalues
            residuals = ols_model.resid
            sqrt_abs_residuals = np.sqrt(np.abs(residuals))  # Square root of absolute residuals

            # Set up the figure
            fig, ax = plt.subplots(1, 2, figsize=(16, 6))

            # Plot: Residuals vs. Fitted Values
            sns.scatterplot(x=fitted_values, y=residuals, ax=ax[0])
            ax[0].axhline(0, color='red', linestyle='--')
            ax[0].set_title('Residuals vs Fitted Values')
            ax[0].set_xlabel('Fitted Values')
            ax[0].set_ylabel('Residuals')

            # Adjust layout and show plot in Streamlit
            plt.tight_layout()
            st.pyplot(fig)

        def check_influential_points(ols_model):
            # Extract leverage and Cook's distance
            influence = ols_model.get_influence()
            leverage = influence.hat_matrix_diag  # Leverage values
            cooks_d = influence.cooks_distance[0]  # Cook's distance
            
            # Create a DataFrame to hold leverage and Cook's distance
            influence_df = pd.DataFrame({
                'Leverage': leverage,
                'Cooks_Distance': cooks_d
            })

            # Identify influential points based on Cook's Distance
            n = len(ols_model.model.endog)  # Number of observations
            threshold_cooks = 4 / n
            influential_points = influence_df[influence_df['Cooks_Distance'] > threshold_cooks]

            # Plot: Leverage vs. Cook's Distance
            fig, ax = plt.subplots(figsize=(10.75, 6))
            ax.scatter(leverage, cooks_d)
            ax.axhline(threshold_cooks, color='red', linestyle='--', label='Cook\'s Distance Threshold')
            ax.set_xlabel('Leverage')
            ax.set_ylabel('Cook\'s Distance')
            ax.set_title('Leverage vs. Cook\'s Distance')
            ax.legend()

            # Display the plot in Streamlit
            st.pyplot(fig)

            # Display the influential points DataFrame in Streamlit
            if not influential_points.empty:
                st.markdown("### Influential Points (based on Cook's Distance):")
                st.dataframe(influential_points)
            else:
                st.markdown("No influential points detected based on Cook's Distance threshold.")

        st.subheader("Please select a model that you would like to check for Regression Assumption, Multicollinearity, Heteroscedasticity, and Influential Points:")
        # Create a list of models from the results
        model_names = [f"y ~ {', '.join(model['Predictors'])}" for model in results_df.to_dict(orient='records')]
        # User selects a model
        model_selected = st.selectbox('Select a model:', model_names)
        # Extract the selected model's predictors
        selected_model_row = results_df.loc[results_df['Predictors'].apply(lambda x: f"y ~ {', '.join(x)}") == model_selected].iloc[0]
        predictors_for_model = selected_model_row['Predictors']

        # Prepare the X matrix and y vector for the selected model
        X_model = sm.add_constant(df[predictors_for_model])  # Add intercept (constant term)
        y_model = df[response]  # Assuming 'response' is the target variable (adjust this as needed)

        # Fit the model
        fitted_model = sm.OLS(y_model, X_model).fit()

        col1, col2 = st.columns([1, 1])
        with col1:
            # Q-Q Plot of Residuals
            plot_qq(fitted_model)
            st.markdown("""
                <div style="display: flex; align-items: center; height: 100%;">
                    <div style="margin: auto;">
                        <b>Things To Check On QQ Plots:</b><br><br>
                        - If the points closely follow the diagonal line, the residuals are approximately normally distributed, which is a key assumption of linear regression.<br><br>
                        - Deviations from this line, especially in the tails, may indicate issues such as non-normality of residuals, which can affect hypothesis testing and confidence intervals.
                    </div>
                </div>
            """, unsafe_allow_html=True)
            check_homoskedasticity(fitted_model)
            st.markdown("""
                <div style="display: flex; align-items: center; height: 100%;">
                    <div style="margin: auto;">
                        <b>Checking for homoskedasticity:</b><br><br>
                        - If the residuals are randomly scattered around zero with no discernible pattern, homoskedasticity is satisfied, supporting the linear regression assumptions.<br><br>
                        - Patterns in the residuals, such as funnels or curves, suggest heteroskedasticity (non-constant variance), which can lead to inefficient estimates and affect statistical tests.
                    </div>
                </div>
            """, unsafe_allow_html=True)
            vif = plot_vif(X_model)
            st.write(vif)
            
        with col2:
            plot_residual_histogram(fitted_model.resid)
            st.markdown("""
                <div style="display: flex; align-items: center; height: 100%;">
                    <div style="margin: auto;">
                        <b>Things To Check On Histogram of Residuals:</b><br><br>
                        - A bell-shaped histogram suggests that the residuals are normally distributed, supporting the assumption of linear regression. <br><br>
                        - If the histogram is skewed or has multiple peaks, it may indicate non-normality, which could violate regression assumptions.
                    </div>
                </div>
            """, unsafe_allow_html=True)
            check_influential_points(fitted_model)
            st.markdown("""
                <div style="display: flex; align-items: center; height: 100%;">
                    <div style="margin: auto;">
                        <b>Checking for influential points</b><br><br>
                        - Points identified as influential may indicate outliers or leverage points that could skew the model results.<br><br>
                        - It's important to investigate these points further to understand their impact on the model and determine if they should be retained or excluded from analysis.
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            

       