import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. TRAIN AND CACHE MODEL & METRICS
# ============================================================

@st.cache_resource
def get_trained_model():

    # Load dataset
    df = pd.read_csv("Attendance_Prediction_2000.csv")

    # Separate features and target
    X = df.drop(
        columns=[
            "student_id",
            "attendance",
            "absence_reason"
        ]
    )

    y = df["attendance"]

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # ========================================================
    # PREPROCESSING
    # ========================================================

    preprocessor = ColumnTransformer(
        transformers=[

            (
                "num",
                "passthrough",
                numerical_cols
            ),

            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore"
                ),
                categorical_cols
            )
        ]
    )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    pipeline = Pipeline(
        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            )
        ]
    )

    # ========================================================
    # TRAIN TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Train
    pipeline.fit(
        X_train,
        y_train
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    # Get trained Random Forest
    rf_model = pipeline.named_steps[
        "classifier"
    ]

    # Get encoded feature names
    encoded_features = pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    # Get importance values
    importance = rf_model.feature_importances_

    simple_names = []

    # Convert encoded names into simple names
    for feature in encoded_features:

        # Remove preprocessing prefixes
        feature = feature.replace(
            "num__",
            ""
        )

        feature = feature.replace(
            "cat__",
            ""
        )

        # Determine original feature
        if feature.startswith("age"):
            simple_name = "Age"

        elif feature.startswith("gender"):
            simple_name = "Gender"

        elif feature.startswith("course"):
            simple_name = "Course"

        elif feature.startswith("year"):
            simple_name = "Year"

        elif feature.startswith("parent_education"):
            simple_name = "Parent Education"

        elif feature.startswith("internet_access"):
            simple_name = "Internet Access"

        elif feature.startswith("hostel_resident"):
            simple_name = "Hostel Resident"

        elif feature.startswith("class_type"):
            simple_name = "Class Type"

        elif feature.startswith("weather"):
            simple_name = "Weather"

        elif feature.startswith("study_hours"):
            simple_name = "Study Hours"

        elif feature.startswith("sleep_hours"):
            simple_name = "Sleep Hours"

        elif feature.startswith("travel_time_minutes"):
            simple_name = "Travel Time"

        else:
            simple_name = feature

        simple_names.append(simple_name)

    # Create feature importance DataFrame
    feature_importance_df = pd.DataFrame({

        "Feature": simple_names,

        "Importance": importance
    })

    # Combine encoded categories
    feature_importance_df = (
        feature_importance_df
        .groupby(
            "Feature",
            as_index=False
        )["Importance"]
        .sum()
    )

    # Sort
    feature_importance_df = feature_importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    # Dataset information
    dataset_info = {

        "total_records": len(df),

        "train_records": len(X_train),

        "test_records": len(X_test),

        "features_count": X.shape[1],

        "accuracy": acc,

        "report": report,

        "confusion_matrix": cm,

        "feature_importance": feature_importance_df
    }

    return pipeline, dataset_info


# ============================================================
# GET TRAINED MODEL
# ============================================================

model, info = get_trained_model()


# ============================================================
# 2. UI LAYOUT
# ============================================================

st.set_page_config(
    page_title="Student Attendance Prediction",
    layout="centered"
)

st.title(
    "🎓 Student Attendance Predictor"
)

st.write(
    "Enter student details below to predict class attendance probability."
)


# ============================================================
# INPUT COLUMNS
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    age = st.number_input(
        "Age",
        min_value=16,
        max_value=40,
        value=20
    )

    gender = st.selectbox(
        "Gender",
        ["male", "female"]
    )

    course = st.selectbox(
        "Course",
        ["bca", "bsc", "bcom", "ba", "bba"]
    )

    year = st.selectbox(
        "Year",
        [
            "1st year",
            "2nd year",
            "3rd year"
        ]
    )

    parent_education = st.selectbox(
        "Parent Education",
        [
            "school",
            "graduate",
            "postgraduate"
        ]
    )

    internet_access = st.selectbox(
        "Internet Access",
        ["yes", "no"]
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    hostel_resident = st.selectbox(
        "Hostel Resident",
        ["yes", "no"]
    )

    class_type = st.selectbox(
        "Class Type",
        ["offline", "online"]
    )

    weather = st.selectbox(
        "Weather",
        [
            "sunny",
            "cloudy",
            "rainy"
        ]
    )

    study_hours = st.slider(
        "Study Hours / Day",
        0.0,
        12.0,
        4.0,
        0.1
    )

    sleep_hours = st.slider(
        "Sleep Hours / Day",
        3.0,
        12.0,
        7.0,
        0.1
    )

    travel_time_minutes = st.number_input(
        "Travel Time (minutes)",
        min_value=0,
        max_value=180,
        value=30
    )


# ============================================================
# 3. PREDICTION BUTTON
# ============================================================

if st.button(
    "Predict Attendance",
    type="primary"
):

    # Create input DataFrame
    input_data = pd.DataFrame(
        [
            {

                "age": age,

                "gender": gender,

                "course": course,

                "year": year,

                "parent_education": parent_education,

                "internet_access": internet_access,

                "hostel_resident": hostel_resident,

                "class_type": class_type,

                "weather": weather,

                "study_hours": study_hours,

                "sleep_hours": sleep_hours,

                "travel_time_minutes": travel_time_minutes

            }
        ]
    )

    # Prediction
    pred = model.predict(
        input_data
    )[0]

    # Probability
    proba = model.predict_proba(
        input_data
    )[0]


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown("---")

    if pred == 1:

        st.success(
            f"**Status: STUDENT IS LIKELY TO BE PRESENT** "
            f"(Confidence: {proba[1] * 100:.1f}%)"
        )

    else:

        st.error(
            f"**Status: STUDENT IS LIKELY TO BE ABSENT** "
            f"(Confidence: {proba[0] * 100:.1f}%)"
        )


    st.progress(
        float(proba[1])
    )

    st.caption(
        f"Absent Probability: {proba[0] * 100:.1f}% | "
        f"Present Probability: {proba[1] * 100:.1f}%"
    )


# ============================================================
# 4. MODEL INFORMATION
# ============================================================

st.markdown("---")


with st.expander(
    "📊 View Model & Training Details",
    expanded=False
):

    st.subheader(
        "Model Overview"
    )


    m_col1, m_col2, m_col3 = st.columns(3)


    m_col1.metric(
        "Model Architecture",
        "Random Forest"
    )


    m_col2.metric(
        "Test Accuracy",
        f"{info['accuracy'] * 100:.2f}%"
    )


    m_col3.metric(
        "Total Dataset Size",
        f"{info['total_records']} rows"
    )


    # ========================================================
    # HYPERPARAMETERS
    # ========================================================

    st.markdown(
        "#### Hyperparameters & Configuration"
    )


    st.write(
        "- **Estimators:** 100 Trees\n"
        "- **Max Depth:** 10\n"
        "- **Train / Test Split:** 80% / 20%\n"
        "- **Features Count:** 12 predictors\n"
        "- **Encoding:** One-Hot Encoding"
    )


    # ========================================================
    # CLASSIFICATION METRICS
    # ========================================================

    st.markdown(
        "#### Detailed Classification Metrics"
    )


    df_report = pd.DataFrame(
        info["report"]
    ).transpose().round(2)


    st.dataframe(
        df_report,
        use_container_width=True
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.markdown(
        "#### Confusion Matrix"
    )

    cm = info["confusion_matrix"]

    fig_cm, ax_cm = plt.subplots(
        figsize=(6, 5)
    )

    ax_cm.imshow(
        cm,
        interpolation="nearest",
        cmap=plt.cm.Blues
    )

    ax_cm.set_title(
        "Confusion Matrix"
    )

    ax_cm.set_xlabel(
        "Predicted Label"
    )

    ax_cm.set_ylabel(
        "Actual Label"
    )

    ax_cm.set_xticks(
        [0, 1]
    )

    ax_cm.set_yticks(
        [0, 1]
    )

    ax_cm.set_xticklabels(
        ["Absent", "Present"]
    )

    ax_cm.set_yticklabels(
        ["Absent", "Present"]
    )

    # Display values inside matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            ax_cm.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    st.pyplot(
        fig_cm
    )

    # Confusion matrix table
    st.markdown(
        "#### Confusion Matrix Values"
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Absent", "Actual Present"],
        columns=["Predicted Absent", "Predicted Present"]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.markdown(
        "#### 🌳 Random Forest Feature Importance"
    )

    st.write(
        "This graph shows how important each student-related "
        "feature is for predicting attendance."
    )


    feature_df = info[
        "feature_importance"
    ].copy()


    # Create graph
    fig, ax = plt.subplots(
        figsize=(9, 6)
    )


    ax.barh(
        feature_df["Feature"][::-1],
        feature_df["Importance"][::-1]
    )


    ax.set_xlabel(
        "Feature Importance"
    )


    ax.set_ylabel(
        "Feature"
    )


    ax.set_title(
        "Random Forest Feature Importance"
    )


    plt.tight_layout()


    # Display graph in Streamlit
    st.pyplot(
        fig
    )


    # ========================================================
    # FEATURE IMPORTANCE TABLE
    # ========================================================

    st.markdown(
        "#### Feature Importance Values"
    )


    display_df = feature_df.copy()


    display_df["Importance"] = (
        display_df["Importance"]
        .round(4)
    )


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
