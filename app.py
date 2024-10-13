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
from PIL import Image

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
    - **Akaike Information Criterion (AIC)**
    - **Bayesian Information Criterion (BIC)**
    - **Adjusted R²**
    - **Mallows Cp**
    - **Hannan-Quinn Criterion (HQC)**

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

    st.subheader("📈 What is MLR?")
    st.markdown("""
    **Multiple Linear Regression (MLR)** is a statistical technique used to model the relationship between one dependent variable (response) and two or more independent variables (predictors). 
    """)

    st.latex(r"""
    y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \epsilon
    """)

    st.write("""
    Where:
    - $y$ is the response variable.
    - $x_1, x_2, \ldots, x_n$ are the predictor variables.
    - $\\beta_0$ is the intercept.
    - $\\beta_1, \\beta_2, \ldots, \\beta_n$ are the coefficients that represent the impact of each predictor.
    - $\\epsilon$ is the error term (representing unexplained variance).
    """)

    st.write("""
    The goal of MLR is to estimate the coefficients $\\beta_1, \\beta_2, \ldots, \\beta_n$, 
    in such a way that the sum of the squared residuals (the differences between the observed and predicted $y$) is minimized.
    """)


    st.markdown("""<hr style='border: 1px solid #E0E0E0;'>""", unsafe_allow_html=True)

    col1, col2 = st.columns([5, 2])
    with col1:
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

    st.markdown("""<hr style='border: 1px solid #E0E0E0;'>""", unsafe_allow_html=True)

    st.subheader("📈 The Model Selection Process Using Best Subsets Regression")
    st.markdown("""
    Before proceeding with best subsets regression, we must decide our response variable and our predictors.
    """)
            
    st.info(
        """
        To help out with the understanding, **IceCreamConsumption.csv** data will be used for illustrations.
        """,
        icon="🔴",
    )

    st.markdown("Let's take a look at the Ice Cream Consumption Data: ")

    col1, col2, col3 = st.columns([1, 1, 1])
    icecream = pd.read_csv('./data/IceCreamConsumption.csv')
    with col1:
        image = Image.open("./data/icecream.head().png")
        st.image(image, caption="IceCreamConsumption.csv Preview", use_column_width=True)

    with col2:
        markdown_text = """ 
        <div style='background-color: #D9EDDB; padding: 6px; border-radius: 5px;'>
            There are a total of 5 variables:
            <ul>
                <li><strong>cons</strong>: consumption of ice cream per head (in pints);</li>
                <li><strong>income</strong>: average family income per week (in US Dollars);</li>
                <li><strong>price</strong>: price of ice cream (per pint);</li>
                <li><strong>temp</strong>: average temperature (in Fahrenheit);</li>
                <li><strong>time</strong>: index from 1 to 30</li>
            </ul>
        </div>
        """
        st.markdown(markdown_text, unsafe_allow_html=True)

    st.markdown("We chose **cons** as our response variable (y) and **income**, **price**, and **temp** as our predictor variables (X). All of them are numerical value.")

    st.markdown("""
    #### # Setting Up The Subset Models
    
    Next steps will be:
    1. ***Find all possible combinations of predictors***
    2. ***Create a model with the subset predictors and the response variables***
                
    And applying this steps to the **IceCreamConsumption.csv**:
    """)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        ***1. Find all possible combinations of predictors*** (counting only the subsets with at least one predictor)
        - [income]
        - [price]
        - [temp]
        - [income, price]
        - [income, temp]
        - [price, temp]
        - [income, price, temp]
        """)
    with col2:
        st.markdown("""
        ***2. Create a model with the subset predictors and the response variables***
        - y ~ income
        - y ~ price
        - y ~ temp
        - y ~ income + price
        - y ~ income + temp
        - y ~ price + temp
        - y ~ income + price + temp
        """)
    
    st.markdown("""
    #### # Calculating Each Model Selection Criteria
    
    Now we are going to calculate the **Adjusted R-squared**, **Mallows Cp**, **AIC** (Akaike Information Criterion),
                **BIC** (Bayesian Information Criterion), **PRESS** (Prediction Sum Of Squared Error) for each subset models.
    """)

    # col1, col2 = st.columns([1, 1])
    # with col1:
    st.markdown("""
    **Wait, what does each criteria measure?**

    - **Adjusted R-squared** measures how much of the variability in the response variable is 
    explained by the model, adjusted for the number of predictors. We want it to be **as large as possible** 
    to show better explanatory power while controlling for overfitting. The formula is:
    """)
    st.latex(r"R^2_{\text{adj}} = 1 - \frac{\text{MSE}}{\text{MST}}")

    st.markdown("""
    - **Mallows Cp** helps balance bias and variance. We want Cp to be close to the parameters (P), 
    indicating a well-fitted model without unnecessary complexity. The formula is:
    """)
    st.latex(r"C_p = \frac{\text{SSE}_p}{\text{MSE}_{\text{full}}} + 2p - n")

    st.markdown("""
    - **AIC (Akaike Information Criterion)** balances model fit and complexity. AIC discourages 
    overfitting and encourages simpler models that still provide a good fit to the data. In model comparisons, 
    **lower AIC values** are preferred. The formula is:
    """)
    st.latex(r"\text{AIC} = n \cdot \log(2\pi) + n \cdot \log(\text{SSE}_p) - n \cdot \log(n) + 2p")

    st.markdown("""
    - **BIC (Bayesian Information Criterion)** imposes a stricter penalty for additional parameters, 
    making it less prone to overfitting. **Lower BIC values** are preferred in model comparisons. The formula is:
    """)
    st.latex(r"\text{BIC} = n \cdot \log(2\pi) + n \cdot \log(\text{SSE}_p) - n \cdot \log(n) + (\log(n)) \cdot p")

    st.markdown("""
    - **PRESS (Prediction Sum Of Squared Error)** measures how well the model predicts new data by summing 
    squared errors from leave-one-out predictions. Lower PRESS values are preferred. The formula is:
    """)
    st.latex(r"\text{PRESS} = \sum_{i=1}^{n} \left( \frac{e_i}{1 - h_{ii}} \right)^2")

# with col2:
    st.markdown("""
    Using the formula on the left, we have calculated the Adjusted R squared, Mallows Cp,
                AIC, BIC, and PRESS for each of the subset models:
    """)
    st.write("")
    image = Image.open("./data/model_selection_criteria_output.png")
    st.image(image, caption="Model Selection Criteria Values", use_column_width=False)

    st.markdown("""
    ##### Before we go over to evaluating these values, we are going to add an extra criteria.
    """)
    
    st.markdown("""<hr style='border: 1px solid #E0E0E0;'>""", unsafe_allow_html=True)
    st.subheader("📌 Extending Our Thoughts #1: Hannan-Quinn Criterion (HQC)")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        #### What is the Hannan-Quinn Criterion (HQC)?
        """)
        st.markdown("""
        ##### :: As per the textbook, HQC is defined as:
        > The Hannan-Quinn Criterion (HQC) is a statistical criterion used for model selection in regression 
        analysis. It is designed to assess the goodness-of-fit of a model while penalizing for model 
        complexity, helping to prevent overfitting.
                    
        ##### :: As per our understanding, HQC is defined as
        > HQC, much like AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion), is 
        used to evaluate the model and handle the complexity of a model by applying a penalty to the likelihood
        function based on the number of parameters, helping to prevent overfitting by discouraging overly 
        complex models.
        """)
        st.markdown("""
        #### Key Points About HQC:
        """)
        st.markdown("""
        - **Logarithmic Transformation:** The criterion uses natural logarithms to transform ratio of RSS to number of observations, which helps in normalizing the measure.
        - **Penalty for Complexity:** The term `2×𝑘×ln(ln(𝑛))` acts as a penalty for model complexity, discouraging the use of too many parameters. The logarithmic nature of the penalty allows it to grow at a rate that is less severe than that of BIC, making HQC a moderate choice.
        """)
    with col2:
        st.markdown("""
            #### Applying HQC to the Ice Cream Consumption Dataset
        """)
        image = Image.open("./data/HQC_included.png")
        st.image(image, caption="Model Selection Criteria Values", use_column_width=True)

        st.markdown("""
        #### Difference from AIC and BIC:
        - AIC uses a penalty of `2×k`, meaning it penalizes complexity less than HQC and BIC.
        - BIC uses `k×ln(n)`, which gives a stronger penalty for model complexity than HQC.
        - HQC’s penalty term `2×k×ln(ln(n))` grows more slowly than BIC but faster than AIC, making it a middle ground between the two.
        """)
    st.markdown("""
    ##### Note:

    - **Comparison with AIC/BIC:** HQC lies between AIC and BIC in terms of model selection. For example, if your HQC value is lower than the AIC or BIC values, it suggests that HQC finds a better balance between fit and complexity.
    - **Model Selection:** If you are comparing multiple models, you should select the one with the lowest HQC value, as it balances goodness of fit and model complexity.
    - **Large Sample Sizes:** For large datasets, BIC may over-penalize complexity, leading to an overly simplistic model. In such cases, HQC provides a more reasonable penalty for complexity without selecting too simple a model.
    """)
    st.markdown("""
    #### What Makes HQC Interesting
    1. **Asymptotic Properties:**
        - HQC has favorable asymptotic properties, meaning that as the sample size increases, it tends to select the correct model more reliably than AIC, making it appealing in larger datasets.
    2. **Flexibility in Application:**
        - HQC can be used for various types of models beyond linear regression, such as nonlinear and time series models, enhancing its versatility in different statistical contexts.
    3. **Focus on Predictive Accuracy:**
        - The primary goal of HQC is to improve predictive accuracy. By balancing model complexity and goodness of fit, it aids in developing models that generalize well to new data.
    """)
    st.markdown("""
    #### When to Use HQC:
    - **Large Datasets:** HQC is better suited for large datasets where the number of parameters grows slowly relative to the number of observations. In such cases, the slower-growing penalty term in HQC prevents over-penalizing the model, which could happen with BIC.
    - **Balancing Simplicity and Complexity:** HQC finds a middle ground between AIC (which tends to overfit) and BIC (which tends to underfit). If you find that AIC suggests overly complex models and BIC suggests overly simple ones, HQC might strike a better balance.
    
    #### Drawbacks of HQC
    1. **Complexity in Interpretation:**
        - The criteria can be complex to interpret, especially for those new to statistical modeling. It may require a solid understanding of the underlying theory to apply effectively.
    2. **Sensitivity to Sample Size:**
        - HQC is sensitive to the sample size; smaller datasets might not provide reliable estimates, leading to misleading conclusions. As sample size increases, the criterion becomes more reliable, but small samples can skew results.
    3. **Overfitting Risk:**
        - Like other information criteria (AIC, BIC), HQC can still select overly complex models that overfit the data, particularly if the penalty for additional parameters isn't sufficient to discourage unnecessary complexity.
    4. **Comparative Use:**
        - HQC is primarily useful for comparing models rather than providing an absolute measure of model quality. It can indicate which model is better among a set, but it does not inherently indicate how good any individual model is.
    5. **Assumption of Normality:**
        - The criterion assumes that the errors are normally distributed, which may not always hold in practice. If this assumption is violated, it could lead to inaccurate model selection.
                        
    > HQC is an excellent alternative to AIC and BIC, particularly useful when AIC tends to overfit, and BIC tends to underfit. By applying HQC in combination with AIC and BIC, you can make better-informed decisions regarding model selection.
    """)

    st.markdown("""<hr style='border: 1px solid #E0E0E0;'>""", unsafe_allow_html=True)

    st.markdown("""#### # Evaluating Best Model For Each Criterion""")

    col1, col2 = st.columns([1, 1])
    with col1:
        markdown_text = """ 
        <div style='background-color: #D9EDDB; padding: 6px; border-radius: 5px;'>
            <h5> ** Recap of Model Selection Rule for Each Criteria ** </h5>
            <ul>
                <li><strong>Adjusted R-squared</strong>: Look for the model that has the <strong>largest adjusted R squared</strong> value;</li>
                <li><strong>Mallows' Cp</strong>: Look for the model that has <strong>Cp close to the number of parameters p.</strong>;</li>
                <li><strong>AIC (Akaike Information Criterion)</strong>: Look for the model that has the <strong>smallest AIC</strong> value;</li>
                <li><strong>BIC (Bayesian Information Criterion)</strong>: Look for the model that has the <strong>smallest BIC</strong> value;</li>
                <li><strong>PRESS (Prediction Sum of Squares)</strong>: Look for the model that has the <strong>smallest PRESS</strong> value;</li>
                <li><strong>HQC (Hannan-Quinn Criterion)</strong>: Look for the model that has the <strong>smallest HQC</strong> value;</li>
            </ul>
        </div>
        """
        st.markdown(markdown_text, unsafe_allow_html=True)
    with col2:
        image = Image.open("./data/best_models.png")
        st.image(image, caption="Best Model Selected For Each Criteria", use_column_width=True)


    st.header("Model Diagnostics")
    st.markdown("""<hr style='border: 2px solid #4CAF50; margin: 0; padding: 0;'>""", unsafe_allow_html=True)

    st.markdown("""
    Now, we can assess the assumptions of linear regression by evaluating various diagnostic plots, including the Q-Q Plot of Residuals, 
    Histogram of Residuals, Correlation Matrix, Variance Inflation Factor (VIF), Homoskedasticity, and Influential Points, to ensure model validity and 
    identify potential issues such as multicollinearity, heteroskedasticity, and outliers.
    """)

    st.markdown("""
        ### Q-Q Plot of Residuals:

        **Purpose**: The Q-Q (quantile-quantile) plot compares the distribution of the model's residuals to a normal distribution.

        **Interpretation**:
        - If the points closely follow the diagonal line, the residuals are approximately normally distributed, which is a key assumption of linear regression.
        - Deviations from this line, especially in the tails, may indicate issues such as non-normality of residuals, which can affect hypothesis testing and confidence intervals.
        """)
    qqplot = Image.open("./data/qqplot.png")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.image(qqplot, width=700)

    st.markdown("""
        ### Histogram of Residuals:

        **Purpose**: This plot shows the frequency distribution of the residuals.

        **Interpretation**:
        - A bell-shaped histogram suggests that the residuals are normally distributed, supporting the assumption of linear regression.
        - If the histogram is skewed or has multiple peaks, it may indicate non-normality, which could violate regression assumptions.
        """)
    histogram = Image.open("./data/hist.png")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.image(histogram, width=700)

    st.markdown("""
        ### Correlation Matrix:

        **Purpose**: The correlation matrix provides insights into the relationships between independent variables.

        **Interpretation**:
        - Values close to +1 or -1 indicate strong correlations, while values near 0 suggest weak correlations.
        - High correlation between independent variables may suggest multicollinearity, which can inflate the variance of coefficient estimates and make the model unstable.
        """)
    heatmap = Image.open("./data/heatmap.png")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.image(heatmap, width=700)

    st.markdown("""
        ### Variance Inflation Factor (VIF):

        **Purpose**: VIF measures how much the variance of a regression coefficient is increased due to multicollinearity.

        **Interpretation**:
        - A VIF value above 5 or 10 is often considered problematic, suggesting significant multicollinearity.
        - High VIF values indicate that the predictor variable is highly correlated with other variables, which can distort the model’s performance.
        """)
    vif = Image.open("./data/vif.png")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.image(vif, width=700)

    st.markdown("""
        ### Homoskedasticity:

        **Purpose**: This analysis checks whether the residuals exhibit constant variance across all levels of the independent variables.

        **Interpretation**:
        - If the residuals are randomly scattered around zero with no discernible pattern, homoskedasticity is satisfied, supporting the linear regression assumptions.
        - Patterns in the residuals, such as funnels or curves, suggest heteroskedasticity (non-constant variance), which can lead to inefficient estimates and affect statistical tests.
        """)
    homo = Image.open("./data/homo.png")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.image(homo, width=700)

    st.markdown("""
        ### Influential Points:

        **Purpose**: This analysis identifies observations that have a disproportionate impact on the regression model’s coefficients and predictions.

        **Interpretation**:
        - Points identified as influential may indicate outliers or leverage points that could skew the model results.
        - It’s important to investigate these points further to understand their impact on the model and determine if they should be retained or excluded from analysis.
        """)
    influ = Image.open("./data/influ.png")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.image(influ, width=700)

    st.header("📌 Extending Our Thoughts #2: Polynomial Models")
    st.markdown("""<hr style='border: 2px solid #4CAF50; margin: 0; padding: 0;'>""", unsafe_allow_html=True)
    st.subheader("📊 What is Polynomial Models?")
    st.markdown("""
    ##### :: As per the textbook, Polynomial Models are defined as:
    > statistical models where the relationship between the dependent variable and the 
    independent variable(s) is expressed as a polynomial equation. For example, a 
    polynomial regression model of degree \(n\) can be represented as:
    """)
    st.latex(r"""
    y = \beta_0 + \beta_1 x + \beta_2 x^2 + \cdots + \beta_n x^n + \epsilon
    """)
    st.markdown("""
    > where \(y\) is the dependent variable, \(x\) is the independent variable, \(beta_0, beta_1, ..., beta_n\) are the coefficients, and \(epsilon\) is the error term.
    """)
    
    st.markdown("""
    ##### :: As per our understanding,
    > Polynomial models extend from linear regression by including curvilinear relationships among the variables. Squares and cubes of the independent variables are included, among higher-order terms, to facilitate even more complex learning of the patterns in the data.
    """)

    st.subheader("📍 Difference from Single Power Models")
    st.markdown("""
    - **Form**: Single power models use a straight-line equation \(y = beta_0 + beta_1*x\), while polynomial models can include squared or higher-order terms (y = beta_0 + beta_1 * x + beta_2 * x^2).
    - **Fit**: Linear models assume a constant rate of change, whereas polynomial models can adapt to varying rates of change, allowing for curves in the data.
    - **Complexity**: Linear models are generally simpler and easier to interpret, while polynomial models can become complex as the degree increases.
    """)

    st.subheader("👀 What Makes Polynomial Models Interesting")
    st.markdown("""
    ##### 1. **Local Behavior Control**
    - The degree of the polynomial allows analysts to control the curvature of the fit. For example, a quadratic model may be sufficient for data with a single curve, while a cubic model can accommodate more intricate patterns.
    - Polynomials can exhibit local maxima and minima, which can be valuable in understanding the dynamics of the data, such as identifying peaks in marketing response curves or other phenomena.
    ##### 2. **Modeling Seasonal and Cyclical Patterns**
    - In time series data, polynomial models can effectively capture cyclical patterns and trends over time, such as seasonal fluctuations, by including time as a predictor variable raised to various powers.
    ##### 3. **Data Smoothing**
    - Polynomial regression can be used to smooth out noisy data points, providing a clearer view of underlying trends without overfitting, especially when choosing an appropriate degree.
    """)

    st.subheader("🌐 Practical Examples Using Actual Dataset")
    st.markdown("""
    To explain the concept of Polynomial Models with an actual example, we have created a random data and have fitted to
    - Model 1: Simple Linear
    - Model 2: Linear with Polynomial Term (X^2)
    - Model 3: Linear with Two Polynomial Terms (X^2, X^3)
                
    We then calculated the AIC, BIC, HQC, Cp, and Adjusted R Squared for each models:
    """)

    image = Image.open("./data/extension2.png")
    st.image(image, caption="Criteria Metrics Calculated For Each Model", use_column_width=False)

    st.subheader("📏Comparison by Metrics")
    st.markdown("""
    1. **X + X² + X³** has the lowest **AIC** with **726.473523**.
    2. **X + X² + X³** has the lowest **BIC** with **736.89420**.
    3. **X + X² + X³** has **Cp** closest to parameters with **4.0**.
    4. **X + X² + X³** has the highest **Adjusted R²** with **0.983682**.
    5. **X + X² + X³** has the lowest **PRESS** with **8297.851715**.
    6. **X + X² + X³** has the lowest **HQC** with **730.690960**.

    **Note**
    - X + X² is very close in performance to X + X² + X³ across most metrics, especially BIC, PRESS, and HQC. 
    - The very slight increase in Adjusted R² and small decreases in AIC and PRESS for X + X² + X³ suggest that while the cubic term provides a marginally better fit, it doesn't add much value in explaining the data.

    > If you prioritize fit while also considering model complexity, the **Linear (X, X²)** model is the best choice overall due to its lower AIC, BIC, HQC, and Cp values. However, if the goal is to maximize explained variance, then the **Linear (X, X², X³)** model could be preferred due to its highest Adjusted R². 
    """)

    st.subheader("💦 Why the Single Power Model(X) Performs Poorly:")
    st.markdown("""
    - The model with just X shows *crazy* values for Cp and PRESS because a simple linear term cannot capture the true non-linear relationship in the data. As a result, metrics like Cp and PRESS explode, indicating that the model is underfitting. The model can't adequately explain the variability, which causes inflated errors and poor predictive power.
    - Real-world data often has non-linear patterns, and restricting the model to just X fails to account for these complexities. This is why introducing higher-order terms (like X² and X³) dramatically improves the model fit and predictive performance. A single power model (just X) is insufficient for such data because it oversimplifies the relationship between the predictor and the response.
    """)

    st.subheader("❓When To Use Polynomials")
    st.markdown("""
    - **Non-linear Relationships**: When data suggests that the relationship between the independent and dependent variables is not linear, polynomial models can be a good fit.
    - **Curvilinear Trends**: In cases where trends appear to have curvature (e.g., U-shaped or inverted U-shaped), polynomial regression can capture these shapes effectively.
    - **Data Patterns**: When exploratory data analysis reveals non-linear patterns, applying polynomial terms can enhance model accuracy.
    - **Feature Expansion**: Polynomial models can be beneficial in feature engineering, transforming existing features into polynomial terms to improve model performance.
    """)

    st.subheader("⚠️ Drawbacks of Polynomial Models")
    st.markdown("""
    1. **Overfitting:**
        - Polynomial models can easily overfit the data, especially with higher-degree polynomials. They may fit the training data perfectly but perform poorly on unseen data, as they capture noise instead of the underlying trend.
    2. **Extrapolation Issues:**
        - Polynomial models can behave erratically outside the range of the training data. Extrapolating predictions based on polynomial functions can lead to unrealistic and highly variable results.
    3. **Multicollinearity:**
        - Polynomial terms (e.g., \(x^2\), \(x^3\)) can introduce multicollinearity, leading to inflated standard errors for the coefficients and making it difficult to determine the individual effect of each predictor.
    
    > Polynomial models are powerful tools for modeling complex relationships between variables. They provide greater flexibility than linear models, but careful consideration must be taken regarding the degree of the polynomial used to avoid overfitting and multicollinearity issues.
    """)

    st.header("📌 Extending Our Thoughts #3: Other Metrics")
    st.markdown("""<hr style='border: 2px solid #4CAF50; margin: 0; padding: 0;'>""", unsafe_allow_html=True)
    
    st.subheader("# What Other Metrics Can We Use to Evaluate a Model?")
    st.markdown("""
    When it comes to evaluating the performance of regression models, relying solely on discussed metrics can sometimes lead to a limited understanding of how well your model is performing. Different metrics capture different aspects of model accuracy and robustness, allowing us to better assess how well our models predict outcomes. 
    In this exploration, we also explored a little bit of two other significant metrics: Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).
    """)

    st.markdown(""" 
    #### Mean Absolute Error (MAE)

    **What is it**: Mean Absolute Error (MAE) is the average of the absolute differences between the predicted and actual values. It quantifies the magnitude of errors in a set of predictions, providing a straightforward measure of model accuracy.
    """)
    st.markdown(
        r"""
        **Formula**:
        $$
        \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
        $$
        """
    )
    st.markdown("""
    Where:

    - y_i is the actual value.
    - y_hatis the predicted value.
    - n is the total number of observations.

    **Interpretation**:
    - MAE is expressed in the same units as the response variable, making it intuitive and easy to interpret. For example, if you are predicting house prices, an MAE of 5,000 dollars means, on average, your predictions are off by 5,000 dollars.
    - This metric is particularly useful when you want to provide a simple summary of prediction errors without overly penalizing larger discrepancies. Thus, MAE is robust to outliers compared to other metrics, such as RMSE, which emphasizes larger errors due to squaring.
    """)

    st.markdown(""" 
    **When to Use**:
    - MAE is especially beneficial in applications where precision is important, such as budgeting or inventory management, where stakeholders need to understand the average error in straightforward terms.
    - Additionally, it is valuable in settings where the data might contain outliers or extreme values that could disproportionately influence other metrics.
    """)

    st.markdown(""" 
    #### Mean Absolute Percentage Error (MAPE)

    **What is it?**: Mean Absolute Percentage Error (MAPE) measures the average absolute percentage error between predicted and actual values. This metric expresses the error in relative terms, making it easier to compare performance across different datasets or scales.
    """)

    st.markdown(
    r"""
    **Formula**:
    $$
    \text{MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
    $$
    """)

    st.markdown("""
    Where:

    - y_i is the actual value.
    - y_hatis the predicted value.
    - n is the total number of observations.

    **Interpretation**:
    - MAPE is expressed as a percentage, providing an intuitive understanding of the accuracy of predictions relative to actual values. For instance, a MAPE of 10% indicates that, on average, the predictions deviate from the actual values by 10%.
    - It is particularly useful when you want to convey errors in a way that stakeholders can easily comprehend, as it normalizes the error against the magnitude of the actual values.

    **When to Use**:
    - MAPE is commonly used in business settings, such as sales forecasting or demand prediction, where stakeholders are interested in understanding performance relative to the scale of sales or demand.
    - It is important to note that MAPE is best suited for datasets where the actual values are always positive, as the percentage calculation can yield undefined results when actual values are zero.
    """)

    st.markdown("""#### Practical Example: We have created a sample random data to illustrate MAE""")

    image = Image.open("./data/other_metrics.png")
    st.image(image, caption="MAE Illustration", use_column_width=False)

    st.markdown(""" 
    ##### Interpretation:
    1. **Mean Absolute Error (MAE): 1.55**
        The MAE value of **1.55** indicates that, on average, the predicted values deviate from the actual values by approximately **1.55 units**. This means that, in terms of the scale of the response variable `y`, our predictions are off by about 1.55 units on average. Lower MAE values are preferred, as they indicate better model performance in absolute terms. 
        
        - **Example interpretation**: If `y` represents some quantity like sales in units, then the predicted sales values are off by about 1.55 units from the actual sales on average. Depending on the context and scale of `y`, this could be considered a small or large error.

    2. **Mean Absolute Percentage Error (MAPE): 9.48%**
        The MAPE value of **9.48%** means that, on average, the predictions are about **9.48% off** from the actual values. MAPE provides a relative measure of error, which is particularly useful when the range of the response variable is wide or when you need to compare the model’s performance across datasets with different scales.
        
        - **Example interpretation**: If `y` represents some sales figures (e.g., dollars, units sold), then the model's predictions are approximately **9.48% less accurate** compared to the actual values. A MAPE of less than 10% is generally considered a reasonably accurate model, so this result suggests that the model is doing fairly well in predicting `y`.
    """)

    st.markdown(""" 
    > Both Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE) provide valuable insights into model performance, each with its strengths and weaknesses. MAE offers a clear understanding of average errors in the same units as the predictions, making it easy to interpret. MAPE, on the other hand, provides a relative perspective on prediction accuracy, facilitating comparisons across different scales.
    """)



elif page == "Model Selection App":
    # Cool introductory section
    st.title("📊 Best Subset Regression App")
    st.markdown("""
    Welcome to the **Best Subset Regression App**! This interactive app allows you to:

    1. **Upload your own dataset**.
    2. **Select a response variable** and ***predictor variables*** from your dataset.
    3. **Automatically run best subsets regression** to find the optimal combinations of predictor variables.

    ***DISCLAIMER: This app can only take in numerical variables!!!
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
        df.columns = df.columns.str.replace('-', '_')

        st.write("Data Preview (first 5 rows):")
        st.dataframe(df.head(5))

        st.header("🖍Step 2: Choose your response and predictor variables:")
        st.markdown("""<hr style='border: 2px solid #92939c; margin: 0; padding: 0;'>""", unsafe_allow_html=True)

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
        

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='n_Predictors').reset_index(drop=True)

        col1, col2 = st.columns([1, 1])


        with col1:
            st.subheader("Values of each metric are calculated for each model below:")
            st.dataframe(results_df)

        with col2:
            st.subheader("Select on the metric to check which model was selected as the best model:")

            available_metrics = ['Adjusted R^2', 'AIC', 'BIC', 'Mallows Cp', 'PRESS', 'HQC']


            if 'selected_models' not in st.session_state:
                st.session_state.selected_models = {}

            chosen_metrics = st.multiselect('Select metrics to find the best models:', available_metrics)

      
            for metric in chosen_metrics:
                if metric not in st.session_state.selected_models:
                    if metric == 'Adjusted R^2':
                        best_model_row = results_df.loc[results_df[metric].idxmax()]
                    elif metric == "Mallows Cp":
                        diff_ser = results_df['Mallows Cp'] - (results_df['n_Predictors'] + 1)[:-1]  # Exclude full model
                        best_model_row = results_df.loc[diff_ser.idxmin()]
                    else:
                        best_model_row = results_df.loc[results_df[metric].idxmin()]
                    
                    best_model_predictors = best_model_row['Predictors']
                    
                    st.session_state.selected_models[metric] = f"The best model selected by {metric} is: y ~ {best_model_predictors}"

            for metric in list(st.session_state.selected_models.keys()):
                if metric not in chosen_metrics:
                    del st.session_state.selected_models[metric]

            for model in st.session_state.selected_models.values():
                st.write(model)
            
            if len(st.session_state.selected_models) > 0:
                if st.button("Done with comparison"):
                    st.session_state.show_step4 = True
    if st.session_state.show_step4:
        st.header("🖍Step 4: Checking Multicollinearity, Heteroscedasticity, and Influential Points")
        st.markdown("""<hr style='border: 2px solid #92939c; margin: 0; padding: 0;'>""", unsafe_allow_html=True)

        def plot_qq(model):
            qq_plot = ProbPlot(model.resid)
            fig, ax = plt.subplots(figsize=(6, 4))
            qq_plot.qqplot(ax=ax, line='q', markerfacecolor='blue', markeredgewidth=1, marker='o', alpha=0.7)
            ax.set_title('Q-Q Plot of Residuals', fontsize=12)
            ax.set_xlabel('Theoretical Quantiles')
            ax.set_ylabel('Sample Quantiles')
            st.pyplot(fig)

        def plot_residual_histogram(residuals):
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(residuals, kde=True, bins=20, ax=ax)
            ax.set_title('Histogram of Residuals', fontsize=12)
            ax.set_xlabel('Residuals')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

        def plot_vif(X):
            vif_data = pd.DataFrame()
            vif_data['Feature'] = X.columns
            vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            st.write("Variance Inflation Factors (VIF):")
            st.dataframe(vif_data)

        def check_homoskedasticity(ols_model):
            fitted_values = ols_model.fittedvalues
            residuals = ols_model.resid
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            sns.scatterplot(x=fitted_values, y=residuals, ax=ax)
            ax.axhline(0, color='red', linestyle='--')
            ax.set_title('Residuals vs Fitted Values', fontsize=12)
            ax.set_xlabel('Fitted Values')
            ax.set_ylabel('Residuals')
            st.pyplot(fig)

        def check_influential_points(ols_model):
            influence = ols_model.get_influence()
            leverage = influence.hat_matrix_diag
            cooks_d = influence.cooks_distance[0]
            influence_df = pd.DataFrame({
                'Leverage': leverage,
                'Cook\'s Distance': cooks_d
            })
            n = len(ols_model.model.endog)
            threshold_cooks = 4 / n
            influential_points = influence_df[influence_df['Cook\'s Distance'] > threshold_cooks]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(leverage, cooks_d)
            ax.axhline(threshold_cooks, color='red', linestyle='--', label='Cook\'s Distance Threshold')
            ax.set_xlabel('Leverage')
            ax.set_ylabel('Cook\'s Distance')
            ax.set_title('Leverage vs. Cook\'s Distance', fontsize=12)
            ax.legend()
            st.pyplot(fig)

            if not influential_points.empty:
                st.markdown("### Influential Points Detected:")
                st.dataframe(influential_points)
            else:
                st.markdown("No influential points detected.")

        st.subheader("Select a Model to Check for Assumptions, Multicollinearity, and Influential Points:")
        model_names = [f"y ~ {', '.join(model['Predictors'])}" for model in results_df.to_dict(orient='records')]
        model_selected = st.selectbox('Select a model:', model_names)

        selected_model_row = results_df.loc[results_df['Predictors'].apply(lambda x: f"y ~ {', '.join(x)}") == model_selected].iloc[0]
        predictors_for_model = selected_model_row['Predictors']
        X_model = sm.add_constant(df[predictors_for_model])
        y_model = df[response]
        fitted_model = sm.OLS(y_model, X_model).fit()


        col1, col2 = st.columns([1, 1])
        with col1:
            plot_qq(fitted_model)
            st.markdown("""
            ##### Q-Q Plot of Residuals:
            - **Purpose**: The Q-Q (quantile-quantile) plot compares the distribution of the model's residuals to a normal distribution.
            - **Interpretation**:
                - If the points closely follow the diagonal line, the residuals are approximately normally distributed, which is a key assumption of linear regression.
                - Deviations from this line, especially in the tails, may indicate issues such as non-normality of residuals, which can affect hypothesis testing and confidence intervals.
            """, unsafe_allow_html=True)

        with col2:
            plot_residual_histogram(fitted_model.resid)
            st.markdown("""
            ##### Histogram of Residuals:
            - **Purpose**: The histogram visualizes the distribution of the residuals from the model.
            - **Interpretation**:
                - A bell-shaped histogram suggests that the residuals are normally distributed, which supports the assumption of linear regression.
                - Skewness or multiple peaks in the histogram may indicate non-normality, suggesting potential issues with the regression model's assumptions.
            """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            check_homoskedasticity(fitted_model)
            st.markdown("""
            ##### Homoskedasticity:
            - **Purpose**: This analysis checks whether the residuals exhibit constant variance across all levels of the independent variables.
            - **Interpretation**:
                - If the residuals are randomly scattered around zero with no discernible pattern, homoskedasticity is satisfied, supporting the linear regression assumptions.
                - Patterns in the residuals, such as funnels or curves, suggest heteroskedasticity (non-constant variance), which can lead to inefficient estimates and affect statistical tests.
            """, unsafe_allow_html=True)

            plot_vif(X_model)
            st.markdown("""
            ##### Variance Inflation Factor (VIF):
            - **Purpose**: VIF measures how much the variance of a regression coefficient is increased due to multicollinearity.
            - **Interpretation**:
                - A VIF value above 5 or 10 is often considered problematic, suggesting significant multicollinearity.
                - High VIF values indicate that the predictor variable is highly correlated with other variables, which can distort the model’s performance.
            """, unsafe_allow_html=True)
            
        with col2:
            check_influential_points(fitted_model)
            st.markdown("""
            ##### Influential Points:
            - **Purpose**: This analysis identifies observations that have a disproportionate impact on the regression model’s coefficients and predictions.
            - **Interpretation**:
                - Points identified as influential may indicate outliers or leverage points that could skew the model results.
                - It's important to investigate these points further to understand their impact on the model and determine if they should be retained or excluded from analysis.
            """, unsafe_allow_html=True)