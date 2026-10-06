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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Attendance Prediction",
    page_icon="🎓",
    layout="centered"
)


# ============================================================
# 1. TRAIN AND CACHE MODEL
# ============================================================

@st.cache_resource
def get_trained_model():

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv("Attendance_Prediction_2000.csv")

    # --------------------------------------------------------
    # Separate features and target
    # --------------------------------------------------------

    X = df.drop(
        columns=[
            "student_id",
            "attendance",
            "absence_reason"
        ]
    )

    y = df["attendance"]

    # --------------------------------------------------------
    # Identify categorical and numerical columns
    # --------------------------------------------------------

    categorical_cols = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Random Forest Classifier
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Train / Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_pred = pipeline.predict(
        X_test
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    acc = accuracy_score(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    # Get Random Forest model
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

    # --------------------------------------------------------
    # Convert encoded names to original feature names
    # --------------------------------------------------------

    for feature in encoded_features:

        feature = feature.replace(
            "num__",
            ""
        )

        feature = feature.replace(
            "cat__",
            ""
        )

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

        simple_names.append(
            simple_name
        )

    # --------------------------------------------------------
    # Create Feature Importance DataFrame
    # --------------------------------------------------------

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
    feature_importance_df = (
        feature_importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
    )

    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    dataset_info = {

        "total_records": len(df),

        "train_records": len(X_train),

        "test_records": len(X_test),

        "features_count": X.shape[1],

        "accuracy": acc,

        "report": report,

        "confusion_matrix": cm,

        "feature_importance":
            feature_importance_df
    }

    return pipeline, dataset_info


# ============================================================
# GET TRAINED MODEL
# ============================================================

model, info = get_trained_model()


# ============================================================
# 2. STREAMLIT TITLE
# ============================================================

st.title(
    "🎓 Student Attendance Predictor"
)

st.write(
    "Enter student details below to predict class attendance probability."
)


# ============================================================
# 3. INPUT COLUMNS
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
        [
            "male",
            "female"
        ]
    )

    course = st.selectbox(
        "Course",
        [
            "bca",
            "bsc",
            "bcom",
            "ba",
            "bba"
        ]
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
        [
            "yes",
            "no"
        ]
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    hostel_resident = st.selectbox(
        "Hostel Resident",
        [
            "yes",
            "no"
        ]
    )

    class_type = st.selectbox(
        "Class Type",
        [
            "offline",
            "online"
        ]
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
# 4. PREDICTION BUTTON
# ============================================================

if st.button(
    "Predict Attendance",
    type="primary"
):

    # --------------------------------------------------------
    # Create input DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [
            {

                "age": age,

                "gender": gender,

                "course": course,

                "year": year,

                "parent_education":
                    parent_education,

                "internet_access":
                    internet_access,

                "hostel_resident":
                    hostel_resident,

                "class_type":
                    class_type,

                "weather":
                    weather,

                "study_hours":
                    study_hours,

                "sleep_hours":
                    sleep_hours,

                "travel_time_minutes":
                    travel_time_minutes
            }
        ]
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    pred = model.predict(
        input_data
    )[0]

    # --------------------------------------------------------
    # Prediction probability
    # --------------------------------------------------------

    proba = model.predict_proba(
        input_data
    )[0]

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Prediction Result"
    )

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

    # --------------------------------------------------------
    # Probability bar
    # --------------------------------------------------------

    st.progress(
        float(proba[1])
    )

    st.caption(
        f"Absent Probability: "
        f"{proba[0] * 100:.1f}% | "
        f"Present Probability: "
        f"{proba[1] * 100:.1f}%"
    )


# ============================================================
# 5. MODEL INFORMATION
# ============================================================

st.markdown("---")

with st.expander(
    "📊 View Model & Training Details",
    expanded=False
):

    st.subheader(
        "Model Overview"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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
        "Dataset Size",
        f"{info['total_records']} rows"
    )

    # --------------------------------------------------------
    # Training information
    # --------------------------------------------------------

    st.markdown(
        "#### Training Information"
    )

    st.write(
        f"- **Total Records:** "
        f"{info['total_records']}\n"
        f"- **Training Records:** "
        f"{info['train_records']}\n"
        f"- **Testing Records:** "
        f"{info['test_records']}\n"
        f"- **Features:** "
        f"{info['features_count']}"
    )

    # --------------------------------------------------------
    # Hyperparameters
    # --------------------------------------------------------

    st.markdown(
        "#### Hyperparameters & Configuration"
    )

    st.write(
        "- **Algorithm:** Random Forest Classifier\n"
        "- **Estimators:** 100 Trees\n"
        "- **Maximum Depth:** 10\n"
        "- **Train / Test Split:** 80% / 20%\n"
        "- **Random State:** 42\n"
        "- **Encoding:** One-Hot Encoding"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    st.markdown(
        "#### 📋 Classification Report"
    )

    df_report = pd.DataFrame(
        info["report"]
    ).transpose().round(2)

    st.dataframe(
        df_report,
        use_container_width=True
    )


# ============================================================
# 6. CONFUSION MATRIX
# ============================================================

st.markdown("---")

st.subheader(
    "📊 Confusion Matrix"
)

st.write(
    "The confusion matrix shows the number of correct "
    "and incorrect attendance predictions."
)

# Get confusion matrix
cm = info["confusion_matrix"]

# Create figure
fig_cm, ax_cm = plt.subplots(
    figsize=(7, 5)
)

# Display matrix
image = ax_cm.imshow(
    cm,
    interpolation="nearest",
    cmap=plt.cm.Blues
)

# Color bar
fig_cm.colorbar(
    image,
    ax=ax_cm
)

# Class labels
class_labels = [
    "Absent",
    "Present"
]

# Set labels
ax_cm.set(
    xticks=range(len(class_labels)),
    yticks=range(len(class_labels)),
    xticklabels=class_labels,
    yticklabels=class_labels,
    ylabel="Actual Attendance",
    xlabel="Predicted Attendance",
    title="Confusion Matrix"
)

# Add numbers inside matrix
threshold = cm.max() / 2

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        ax_cm.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            ),
            fontsize=14
        )

plt.tight_layout()

# Display graph
st.pyplot(
    fig_cm
)

# ------------------------------------------------------------
# Confusion Matrix Table
# ------------------------------------------------------------

st.markdown(
    "#### Confusion Matrix Values"
)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual Absent",
        "Actual Present"
    ],
    columns=[
        "Predicted Absent",
        "Predicted Present"
    ]
)

st.dataframe(
    cm_df,
    use_container_width=True
)


# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================

st.markdown("---")

st.subheader(
    "🌳 Random Forest Feature Importance"
)

st.write(
    "This graph shows how important each student-related "
    "feature is for predicting attendance."
)

# Get feature importance
feature_df = info[
    "feature_importance"
].copy()

# ------------------------------------------------------------
# Feature Importance Graph
# ------------------------------------------------------------

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

# Display graph
st.pyplot(
    fig
)


# ============================================================
# FEATURE IMPORTANCE TABLE
# ============================================================

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


# ============================================================
# 8. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Student Attendance Prediction using "
    "Random Forest Classifier"
)
