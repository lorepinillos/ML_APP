import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, r2_score, confusion_matrix, roc_curve, auc, classification_report, accuracy_score)
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------------------------
# 1. SESSION STATE INITIALIZATION
# -------------------------------------------------------------------
def init_session_state():
    """
    Initialize all variables stored in st.session_state
    (if they do not already exist).

    This allows us to access these variables throughout the app without
    passing them around through function arguments.
    """
    if "df" not in st.session_state:
        st.session_state.df = None  # Will hold the loaded DataFrame

    if "model" not in st.session_state:
        st.session_state.model = None  # Will hold the trained model object

    if "model_type" not in st.session_state:
        st.session_state.model_type = None  # "regression" or "classification"

    if "feature_cols" not in st.session_state:
        st.session_state.feature_cols = []

    if "target_col" not in st.session_state:
        st.session_state.target_col = None

    if "df_cleaned" not in st.session_state:
        st.session_state.df_cleaned = None  # Holds DataFrame after missing data handling or encoding

# -------------------------------------------------------------------
# 2. DATA LOADING
# -------------------------------------------------------------------

def load_sns_dataset(selected_dataset_name):
    """
    Load one of Seaborn's built-in datasets.

    We only demonstrate 'iris' and 'penguins' here.
    You can add more if you want (e.g. 'tips', 'titanic', etc.).
    """
    if selected_dataset_name == "iris":
        df = sns.load_dataset("iris")
    else:
        df = sns.load_dataset("penguins")  # dropna is not done here; we'll handle missing data separately
    return df

def handle_missing_data(df, strategy):
    """
    Basic function to handle missing data based on user's selection.
    - If strategy == "Drop Rows": drop any row with missing data.
    - If strategy == "Fill Mean": fill numeric columns with mean.
    - Otherwise, return df as-is.
    """
    if strategy == "Drop Rows":
        return df.dropna()
    elif strategy == "Fill Mean":
        # Only fill numeric columns with mean, keep others as-is
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        return df
    else:
        # "None" or unrecognized -> do nothing
        return df
# -------------------------------------------------------------------
# 3. SIDEBAR: DATASET SELECTION
# -------------------------------------------------------------------
def sidebar_dataset_selection():
    """
    This function places UI elements in the Streamlit sidebar
    that let the user pick either a built-in Seaborn dataset
    or upload a CSV file.
    """
    st.sidebar.header("1. Choose a Dataset")
    data_source = st.sidebar.radio("Dataset Source", ("Seaborn", "Upload CSV"))

    if data_source == "Seaborn":
        dataset_name = st.sidebar.selectbox(
            "Select a built-in Seaborn dataset:",
            ["iris", "penguins"]
        )
        # When user clicks "Load", load the seaborn dataset
        if st.sidebar.button("Load Selected Seaborn Dataset"):
            st.session_state.df = load_sns_dataset(dataset_name)

    else:  # "Upload CSV"
        uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
        if uploaded_file is not None:
            st.session_state.df = pd.read_csv(uploaded_file)
# -------------------------------------------------------------------
# 4. MAIN PAGE: MISSING DATA HANDLING + DATA PREVIEW
# -------------------------------------------------------------------
def data_preview_and_missing_data():
    """
    If a dataset is loaded, let the user handle missing data
    and show a preview of the resulting DataFrame.
    """
    if st.session_state.df is not None:
        st.write("### Original Data Preview")
        st.write("DataFrame shape:", st.session_state.df.shape)

        # Show first 5 rows
        st.dataframe(st.session_state.df.head())

        # Missing data handling
        st.write("### Handle Missing Data")
        missing_strategy = st.selectbox(
            "Choose a strategy to handle missing values:",
            ["None", "Drop Rows", "Fill Mean"],
            help=(
                "'None': do nothing.\n"
                "'Drop Rows': drop any row with NaN.\n"
                "'Fill Mean': fill numeric columns with their mean."
            )
        )
        apply_missing = st.button("Apply Missing Data Strategy")

        if apply_missing:
            # Clean the data according to user choice
            cleaned_df = handle_missing_data(st.session_state.df.copy(), missing_strategy)
            st.session_state.df_cleaned = cleaned_df
            st.success(f"Missing data strategy '{missing_strategy}' applied!")
        else:
            # If user didn't click yet, just use the original df as default
            if st.session_state.df_cleaned is None:
                st.session_state.df_cleaned = st.session_state.df.copy()

        # Show preview of the "cleaned" DataFrame
        st.write("### Data Preview (After Missing Data Handling)")
        st.write("DataFrame shape:", st.session_state.df_cleaned.shape)
        st.dataframe(st.session_state.df_cleaned.head())

    else:
        st.info("No dataset loaded yet. Please choose a dataset in the sidebar.")
# -------------------------------------------------------------------
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# -------------------------------------------------------------------
def perform_eda():
    """
    Let the user explore the numeric columns with:
      - Correlation Heatmap
      - Distribution (Histogram) of each numeric column
    """
    if st.session_state.df_cleaned is None:
        return

    st.write("## Exploratory Data Analysis")

    # 5A. Correlation Heatmap
    st.write("### Correlation Heatmap (Numeric Columns)")
    numeric_df = st.session_state.df_cleaned.select_dtypes(include=np.number)

    if not numeric_df.empty:
        corr_matrix = numeric_df.corr(numeric_only=True)
        fig, ax = plt.subplots()
        # Using matshow or imshow for correlation
        cax = ax.matshow(corr_matrix)
        fig.colorbar(cax)
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="left")
        ax.set_yticklabels(corr_matrix.columns)
        st.pyplot(fig)
    else:
        st.write("No numeric columns for correlation heatmap.")

    # 5B. Distribution plots of numeric columns
    st.write("### Distribution of Numeric Features")
    for col in numeric_df.columns:
        st.write(f"#### Distribution of '{col}'")
        fig, ax = plt.subplots()
        ax.hist(numeric_df[col].dropna(), bins=30)  # Avoid subplots, so 1 plot each loop
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
# -------------------------------------------------------------------
# 6. FEATURE & TARGET SELECTION
# -------------------------------------------------------------------
def feature_target_selection():
    """
    Let the user pick whether they want a 'Regression' or 'Classification' task,
    which columns will be used as features, and which column is the target.
    """
    if st.session_state.df_cleaned is None:
        return

    st.write("## 1) Select Features and Target")

    # For clarity, let's identify numeric vs. non-numeric columns
    df = st.session_state.df_cleaned
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    # Let user choose model type
    model_type = st.radio("Model Type", ("Regression", "Classification"))
    st.session_state.model_type = model_type.lower()

    # Choose feature columns
    st.session_state.feature_cols = st.multiselect(
        "Select Feature Columns:",
        all_cols,
        default=numeric_cols  # By default, just choose numeric columns
    )

    # Choose target column
    possible_targets = [col for col in all_cols if col not in st.session_state.feature_cols]
    if len(possible_targets) == 0:
        st.error("No columns left to select as target! Please unselect at least one feature column.")
        return

    st.session_state.target_col = st.selectbox(
        "Select Target Column:",
        possible_targets,
    )

    # Check if user selected Regression but the target is non-numeric
    if st.session_state.model_type == "regression":
        target_dtype = df[st.session_state.target_col].dtype
        if not np.issubdtype(target_dtype, np.number):
            st.warning("⚠️ You selected a Regression model but the target column is categorical. Consider choosing a Classification model instead.")

# -------------------------------------------------------------------
# 7. MODEL DOWNLOAD
# -------------------------------------------------------------------
def download_trained_model_button():
    """
    Provide a button for downloading the trained model in pickle format.
    """
    if st.session_state.model is not None:
        model_bytes = pickle.dumps(st.session_state.model)
        st.download_button(
            label="Download Trained Model",
            data=model_bytes,
            file_name="trained_model.pkl",
            mime="application/octet-stream"
        )
# -------------------------------------------------------------------
# 8. MODEL TRAINING FORM
# -------------------------------------------------------------------
def model_training_form():
    """
    Let the user choose the train/test split size and pick a model type.
    Then train the model upon submission.
    """
    if st.session_state.df_cleaned is None:
        return

    # If no features or target, do nothing
    if not st.session_state.feature_cols or not st.session_state.target_col:
        return

    st.write("## 2) Model Training & Configuration")
    with st.form("model_training_form"):
        test_size = st.slider("Test size (fraction for test set)", 0.1, 0.9, 0.2, 0.05)

        # Choose which model
        if st.session_state.model_type == "regression":
            model_name = st.selectbox(
                "Choose Regression Model:",
                ["Linear Regression", "Random Forest Regressor"]
            )
        else:
            model_name = st.selectbox(
                "Choose Classification Model:",
                ["Logistic Regression", "Random Forest Classifier"]
            )

        # Model-specific parameters
        if "Random Forest" in model_name:
            n_estimators = st.number_input(
                "Number of trees (n_estimators)", min_value=10, max_value=500, value=100, step=10
            )
        else:
            n_estimators = None

        # Submit button
        submitted = st.form_submit_button("Fit Model!")
        if submitted:
            train_model(model_name, test_size, n_estimators)
# -------------------------------------------------------------------
# 9. TRAIN MODEL LOGIC
# -------------------------------------------------------------------
def train_model(model_name, test_size, n_estimators=None):
    """
    Perform the actual training of the selected model using
    the user-selected features & target.
    """
    df = st.session_state.df_cleaned
    feature_cols = st.session_state.feature_cols
    target_col = st.session_state.target_col

    if (not feature_cols) or (target_col is None):
        st.error("Please select features and target before training!")
        return

    X = df[feature_cols]
    y = df[target_col]

    # If classification, let's ensure the target is numeric (for certain ops like ROC)
    # We'll label-encode it if it's object or string-based:
    if st.session_state.model_type == "classification":
        if y.dtype == object or isinstance(y.iloc[0], str):
            le = LabelEncoder()
            y = le.fit_transform(y)
            st.info("Target column was label-encoded for classification.")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Instantiate the chosen model
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=200)
    elif model_name == "Random Forest Regressor":
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    elif model_name == "Random Forest Classifier":
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    else:
        st.error("Unknown model type selected.")
        return

    # Fit (train) the model on the training data
    model.fit(X_train, y_train)

    # Save the model to session state (so we can reuse or download it)
    st.session_state.model = model

    # Let the user know training finished
    st.success(f"Model '{model_name}' trained successfully!")

    # Display results
    display_model_results(model, X_train, X_test, y_train, y_test)
# -------------------------------------------------------------------
# 10. DISPLAY MODEL RESULTS
# -------------------------------------------------------------------
def display_model_results(model, X_train, X_test, y_train, y_test):
    """
    Display different metrics and plots depending on whether
    we have a regression or classification model.
    """
    model_type = st.session_state.model_type
    # Predictions on X_test
    y_pred = model.predict(X_test)

    if model_type == "regression":
        st.write("## Regression Results")

        # 9A. Basic regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        st.write(f"**MSE**: {mse:.3f}")
        st.write(f"**RMSE**: {rmse:.3f}")
        st.write(f"**R²**: {r2:.3f}")

        # 9B. Residual distribution
        st.write("### Residual Distribution")
        residuals = y_test - y_pred
        fig, ax = plt.subplots()
        ax.hist(residuals, bins=30)
        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # 9C. Predicted vs Actual plot
        st.write("### Predicted vs. Actual")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

        # 9D. Feature importances (if RandomForestRegressor)
        if isinstance(model, RandomForestRegressor):
            show_feature_importances(model, X_train.columns)

    else:
        st.write("## Classification Results")

        # 9A. Accuracy
        acc = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy**: {acc:.3f}")

        # 9B. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        im = ax.imshow(cm, aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # 9C. Classification Report
        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))

        # 9D. ROC Curve (binary only)
        unique_labels = np.unique(y_test)
        if len(unique_labels) == 2:
            # Get predicted probabilities for the positive class
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            st.write("### ROC Curve & AUC (Binary Classification)")
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", linewidth=2)
            ax.plot([0, 1], [0, 1], 'r--', label="Random Guess")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Receiver Operating Characteristic (ROC)")
            ax.legend(loc="lower right")
            st.pyplot(fig)
        else:
            # If multi-class, just display a note
            st.info("ROC curve is only available for binary classification (2 classes).")

        # 9E. Feature importances (if RandomForestClassifier)
        if isinstance(model, RandomForestClassifier):
            show_feature_importances(model, X_train.columns)
# -------------------------------------------------------------------
# 11. FEATURE IMPORTANCE HELPER
# -------------------------------------------------------------------
def show_feature_importances(model, columns):
    """
    Show a simple bar chart of feature importances for RandomForest models.
    """
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    st.write("### Feature Importances")
    fig, ax = plt.subplots()
    ax.bar(range(len(importances)), importances[sorted_idx])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(columns[sorted_idx], rotation=45, ha="right")
    st.pyplot(fig)
# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------
def main():
    st.title("Interactive ML Trainer App")

    # 1. Initialize session state
    init_session_state()

    # 2. Sidebar for data selection
    sidebar_dataset_selection()

    if st.session_state.df is not None:
        # 3. Missing data handling + data preview
        data_preview_and_missing_data()

        # 4. EDA
        perform_eda()

        # 5. Feature/target selection
        feature_target_selection()

        # 6. Model training form
        model_training_form()

        # --------------------------------------
        # 7. Put the download button HERE, after the form is closed
        # --------------------------------------
        if st.session_state.model is not None:
            # Use your existing helper function if you like,
            # or just call st.download_button() directly:
            download_trained_model_button()
            # or:
            # model_bytes = pickle.dumps(st.session_state.model)
            # st.download_button("Download Trained Model", model_bytes, "trained_model.pkl")
    else:
        st.info("Please select or upload a dataset in the sidebar to get started.")


if __name__ == "__main__":
    main()
