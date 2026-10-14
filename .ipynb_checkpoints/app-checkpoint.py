
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "Models"

PRESSURE_MODEL_PATH = MODEL_DIR / "parental_pressure_model.pkl"
TOXICITY_MODEL_PATH = MODEL_DIR / "parental_toxicity_model.pkl"

FEATURES = (
    "Age",
    "Gender",
    "Education_Level",
    "Background",
    "Pressure_Relevance",
    "Comparison_Relevance",
    "Career_Relevance",
    "Personal_Decison_Relevance",
    "Decision_Relevance",
    "Friendship_Relevance",
    "Guilt_Relevance",
    "Feeling_relevance",
    "Fear_Relevance",
)

GENDER_CODES = {
    "Male": 0,
    "Female": 1,
    "Prefer not to say": 2,
}

EDUCATION_CODES = {
    "School": 0,
    "Undergraduate": 1,
    "Postgraduate": 2,
}

BACKGROUND_CODES = {
    "Rural": 1,
    "Semi-rural": 2,
    "Urban": 3,
}

SURVEY_QUESTIONS = (
    (
        "Pressure_Relevance",
        "Pressure to achieve high academic results",
        "How often do your parents pressure you to achieve high academic results?",
    ),
    (
        "Comparison_Relevance",
        "Compared with others",
        "How often do your parents compare your academic performance with others?",
    ),
    (
        "Career_Relevance",
        "Influence over career choice",
        "How often do your parents influence your career choice?",
    ),
    (
        "Personal_Decison_Relevance",
        "Parents making personal decisions",
        "How often do your parents take important decisions for you?",
    ),
    (
        "Decision_Relevance",
        "Freedom to make your own decisions",
        "How much freedom do you have to make decisions about your own life?",
    ),
    (
        "Friendship_Relevance",
        "Interference with friends or social life",
        "How often do your parents intervene with your friends or social life?",
    ),
    (
        "Guilt_Relevance",
        "Feeling guilty when disagreeing",
        "How often do you feel guilty when you disagree with your parents?",
    ),
    (
        "Feeling_relevance",
        "Feelings dismissed",
        "How often do your parents dismiss your feelings when you disagree?",
    ),
    (
        "Fear_Relevance",
        "Fear of disappointing your parents",
        "How afraid are you of disappointing your parents?",
    ),
)


st.set_page_config(
    page_title="Parental Pressure & Toxicity Estimator",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_models():
    for path in (PRESSURE_MODEL_PATH, TOXICITY_MODEL_PATH):
        if not path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Check the Models folder and the exact filenames."
            )

    pressure_model = joblib.load(PRESSURE_MODEL_PATH)
    toxicity_model = joblib.load(TOXICITY_MODEL_PATH)

    return pressure_model, toxicity_model


st.title("Parental Pressure & Toxicity Estimator")

st.write(
    "Enter demographic details and survey responses to estimate parental "
    "pressure and the perceived toxicity of the parenting environment."
)

st.caption(
    "These are model-based estimates, not a diagnosis or a substitute "
    "for professional support."
)

try:
    pressure_model, toxicity_model = load_models()
except Exception as error:
    st.error("The saved models could not be loaded.")
    st.code(str(error))
    st.stop()


with st.form("prediction_form"):
    st.subheader("About you")
    demographic_columns = st.columns(4)

    with demographic_columns[0]:
        age = st.number_input(
            "Age",
            min_value=14,
            max_value=50,
            value=20,
            step=1,
        )

    with demographic_columns[1]:
        gender_label = st.selectbox("Gender", tuple(GENDER_CODES))

    with demographic_columns[2]:
        education_label = st.selectbox(
            "Education level", tuple(EDUCATION_CODES)
        )

    with demographic_columns[3]:
        background_label = st.selectbox(
            "Residential background", tuple(BACKGROUND_CODES)
        )

    st.subheader("Your experiences")
    st.caption(
        "For each item, choose a response from 1 (Never) to 5 (Very often)."
    )

    survey_values = {}
    question_columns = st.columns(3)

    for index, (feature, label, question) in enumerate(SURVEY_QUESTIONS):
        with question_columns[index % 3]:
            survey_values[feature] = st.slider(
                label,
                min_value=1,
                max_value=5,
                value=3,
                help=question,
                key=feature,
            )

    submitted = st.form_submit_button(
        "Estimate scores",
        type="primary",
    )


if submitted:
    input_values = {
        "Age": age,
        "Gender": GENDER_CODES[gender_label],
        "Education_Level": EDUCATION_CODES[education_label],
        "Background": BACKGROUND_CODES[background_label],
        **survey_values,
    }

    model_input = pd.DataFrame(
        [[input_values[feature] for feature in FEATURES]],
        columns=FEATURES,
    )

    try:
        pressure_score = float(pressure_model.predict(model_input)[0])
        toxicity_score = float(toxicity_model.predict(model_input)[0])

        st.subheader("Estimated results")
        pressure_column, toxicity_column = st.columns(2)

        with pressure_column:
            st.metric("Parental pressure", f"{pressure_score:.2f} / 10")
            st.caption("Model: Linear Regression")

        with toxicity_column:
            st.metric(
                "Perceived parental toxicity",
                f"{toxicity_score:.2f} / 10",
            )
            st.caption("Model: Random Forest Regressor")

        st.info(
            "Model outputs are continuous estimates and may not be whole "
            "numbers. Interpret them in the context of the training data."
        )

    except Exception as error:
        st.error("Prediction failed. The saved model may expect a different input schema.")
        st.code(f"{type(error).__name__}: {error}")
