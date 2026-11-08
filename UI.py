import streamlit as st
import pandas as pd
import joblib
import string

from pathlib import Path


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="EmotiSense",
    page_icon="🧠",
    layout="wide"
)

# ==========================================
# 2. CUSTOM DESIGN
# ==========================================
st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    max-width: 1150px;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #2563EB;
}

.subtitle {
    font-size: 17px;
    color: #808080;
}

.author-line {
    font-size: 16px;
    color: #64748b;
    margin-top: 10px;
    font-weight: 400;
}

.author-name {
    color: #F97316;
    font-weight: 800;
    letter-spacing: 0.5px;
}

div[data-testid="stMetric"] {
    background: rgba(120,120,120,0.07);
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    model_path = Path(__file__).parent / "emotion_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            "emotion_model.joblib not found!"
        )

    package = joblib.load(model_path)

    return package


try:

    # Load saved model package
    package = load_model()

    vectorizer = package["vectorizer"]

    model = package["model"]

    emotion_labels = {
        int(k): v
        for k, v in package["emotion_labels"].items()
    }

    stop_words = set(package["stop_words"])

    accuracy = package.get("accuracy")

    # Additional model information
    algorithm_name = package.get(
        "algorithm",
        "Unknown Model"
    )

    macro_f1 = package.get("macro_f1")

    weighted_f1 = package.get("weighted_f1")

    # ==========================================
    # VERIFY ACTUAL UNDERLYING CLASSIFIER
    # ==========================================

    base_classifiers = []

    if hasattr(model, "calibrated_classifiers_"):

        for calibrated in model.calibrated_classifiers_:

            estimator = getattr(
                calibrated,
                "estimator",
                None
            )

            if estimator is None:

                estimator = getattr(
                    calibrated,
                    "base_estimator",
                    None
                )

            if estimator is not None:

                base_classifiers.append(
                    type(estimator).__name__
                )

    else:

        base_classifiers.append(
            type(model).__name__
        )

    # Print verification in VS Code terminal
    print("\n========== MODEL VERIFICATION ==========")

    print("LOADED ALGORITHM:", algorithm_name)

    print("ACTUAL CLASSIFIERS:", base_classifiers)

    print("TEST ACCURACY:", accuracy)

    print("MACRO F1:", macro_f1)

    print("WEIGHTED F1:", weighted_f1)

    print("========================================\n")


except Exception as error:

    st.error(f"Model loading error: {error}")

    st.info(
        "Please save your trained model from the Jupyter notebook "
        "and put emotion_model.joblib beside UI.py."
    )

    st.stop()


# ==========================================
# 4. TEXT PREPROCESSING
# ==========================================

def clean_text(text):

    text = str(text).lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    text = "".join(
        char for char in text
        if not char.isdigit()
    )

    # Remove emojis and non-ASCII characters
    text = "".join(
        char for char in text
        if char.isascii()
    )

    # Remove stopwords
    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# ==========================================
# 5. HELPER FUNCTIONS
# ==========================================

def get_emotion(class_id):

    return emotion_labels.get(
        int(class_id),
        "Unknown"
    )


def get_icon(emotion):

    icons = {
        "joy": "😊",
        "love": "❤️",
        "anger": "😠",
        "sadness": "😢",
        "fear": "😨",
        "surprise": "😲"
    }

    return icons.get(
        emotion.lower(),
        "💬"
    )


def get_scores(vectorized_text):

    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(
        vectorized_text
    )[0]

    scores = pd.DataFrame({
        "Emotion": [
            get_emotion(class_id)
            for class_id in model.classes_
        ],
        "Score": probabilities
    })

    return scores.sort_values(
        by="Score",
        ascending=False
    ).reset_index(drop=True)


# ==========================================
# 6. HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🧠 EmotiSense</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'NLP-Based Text Emotion Classification System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="author-line">'
    'Developed by '
    '<span class="author-name">Okasha Amir</span>'
    '</div>',
    unsafe_allow_html=True
)

st.write("")
st.markdown("---")

# ==========================================
# 7. SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🧠 EmotiSense")

    st.caption(
        "Machine Learning Emotion Detector"
    )

    st.success("ML Model Loaded Successfully")

    st.markdown("### Supported Emotions")

    for class_id in sorted(emotion_labels):

        emotion = emotion_labels[class_id]

        st.write(
            f"{get_icon(emotion)} {emotion.title()}"
        )

    st.markdown("---")

    st.markdown(
        "**Developed using:**"
    )

    st.write("Python")
    st.write("NLTK")
    st.write("Scikit-learn")
    st.write("Streamlit")
    st.markdown("---")

    st.markdown("### Project Developer")

    st.write("**Okasha Amir**")

    st.caption("Computer Science Student")
    st.markdown("---")

    st.caption(
        "This application classifies emotions expressed in text. "
        "It does not diagnose mental health conditions."
    )


# ==========================================
# 8. APPLICATION TABS
# ==========================================

tab1, tab2, tab3 = st.tabs([
    "✍️ Analyze Text",
    "📁 Batch Analysis",
    "📊 Model Information"
])


# ==========================================
# TAB 1: SINGLE TEXT PREDICTION
# ==========================================

with tab1:

    st.header("Emotion Analysis")

    st.write(
        "Enter any sentence or paragraph "
        "to identify its expressed emotion."
    )

    with st.form("emotion_form"):

        user_text = st.text_area(
            "Enter your text here",

            placeholder=(
                "Example: I am extremely happy "
                "because I achieved my goals today!"
            ),

            height=160
        )

        submit = st.form_submit_button(
            "Analyze Emotion",
            type="primary",
            use_container_width=True
        )


    if submit:

        if not user_text.strip():

            st.warning(
                "Please enter some text first."
            )

        else:

            # Preprocess the input
            processed_text = clean_text(user_text)

            if not processed_text:

                st.warning(
                    "No usable words remained after cleaning. "
                    "Please enter a different sentence."
                )

            else:

                # Convert text into numerical features
                text_vector = vectorizer.transform([
                    processed_text
                ])

                # Predict emotion using trained model
                prediction = model.predict(
                    text_vector
                )[0]

                predicted_emotion = get_emotion(
                    prediction
                )

                st.markdown("---")

                st.subheader("Prediction Result")

                col1, col2 = st.columns(2)

                with col1:

                    st.success(
                        f"{get_icon(predicted_emotion)} "
                        f"Predicted Emotion: "
                        f"{predicted_emotion.upper()}"
                    )

                scores = get_scores(text_vector)

                with col2:

                    if scores is not None:

                        st.metric(
                            "Top Class Score",
                            f"{scores.iloc[0]['Score']:.2%}"
                        )

                # Display all emotion scores
                if scores is not None:

                    st.subheader(
                        "Emotion Distribution"
                    )

                    st.bar_chart(
                        scores.set_index("Emotion")
                    )

                    display_scores = scores.copy()

                    display_scores["Score"] = (
                        display_scores["Score"]
                        .map(lambda x: f"{x:.2%}")
                    )

                    st.dataframe(
                        display_scores,
                        hide_index=True,
                        use_container_width=True
                    )

                with st.expander(
                    "View Preprocessed Text"
                ):

                    st.write(processed_text)


# ==========================================
# TAB 2: BATCH TEXT PREDICTION
# ==========================================

with tab2:

    st.header("Bulk Emotion Analysis")

    st.write(
        "Upload a CSV file containing multiple text entries."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            if df.empty:

                st.warning("The CSV file is empty.")

            else:

                columns = list(df.columns)

                default_index = (
                    columns.index("text")
                    if "text" in columns
                    else 0
                )

                text_column = st.selectbox(
                    "Select the text column",
                    columns,
                    index=default_index
                )

                if st.button(
                    "Analyze CSV",
                    type="primary"
                ):

                    # Clean all text entries
                    cleaned = (
                        df[text_column]
                        .fillna("")
                        .astype(str)
                        .apply(clean_text)
                    )

                    valid = cleaned.str.strip().ne("")

                    results = df.copy()

                    results["Predicted Emotion"] = (
                        "No usable text"
                    )

                    results["Top Class Score (%)"] = None

                    if valid.any():

                        vectors = vectorizer.transform(
                            cleaned.loc[valid].tolist()
                        )

                        predictions = model.predict(
                            vectors
                        )

                        results.loc[
                            valid,
                            "Predicted Emotion"
                        ] = [
                            get_emotion(x)
                            for x in predictions
                        ]

                        if hasattr(model, "predict_proba"):

                            scores = (
                                model.predict_proba(vectors)
                                .max(axis=1) * 100
                            )

                            results.loc[
                                valid,
                                "Top Class Score (%)"
                            ] = scores.round(2)

                    st.success(
                        f"Successfully analyzed {int(valid.sum())} text entries."
                    )

                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                    # Download results
                    csv = results.to_csv(
                        index=False
                    ).encode("utf-8-sig")

                    st.download_button(
                        "Download Results",
                        data=csv,
                        file_name="emotisense_results.csv",
                        mime="text/csv"
                    )

        except Exception as error:

            st.error(
                f"CSV processing error: {error}"
            )


# ==========================================
# TAB 3: MODEL INFORMATION
# ==========================================

with tab3:

    st.header("Model Information")

    # --------------------------------------
    # 1. Algorithm name formatting
    # --------------------------------------

    if "svm" in algorithm_name.lower():

        algorithm_display = "Calibrated SVM"

    elif "logistic" in algorithm_name.lower():

        algorithm_display = "Calibrated Logistic"

    else:

        algorithm_display = algorithm_name

    # --------------------------------------
    # 2. First row of model information
    # --------------------------------------

    col1, col2, col3 = st.columns([1.5, 1, 1])

    with col1:

        st.metric(
            "Algorithm",
            algorithm_display
        )

        st.caption(f"Full name: {algorithm_name}")

    with col2:

        st.metric(
            "Feature Extraction",
            "TF-IDF"
        )

    with col3:

        st.metric(
            "Emotion Classes",
            len(emotion_labels)
        )

    st.markdown("---")

    # --------------------------------------
    # 3. Second row of model metrics
    # --------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Test Accuracy",
            f"{accuracy:.2%}" if accuracy is not None else "N/A"
        )

    with col2:

        st.metric(
            "Macro F1",
            f"{macro_f1:.2%}" if macro_f1 is not None else "N/A"
        )

    with col3:

        st.metric(
            "Weighted F1",
            f"{weighted_f1:.2%}" if weighted_f1 is not None else "N/A"
        )

    st.markdown("---")

    # --------------------------------------
    # 4. Supported emotion categories
    # --------------------------------------

    st.subheader("Supported Emotion Categories")

    for class_id in sorted(emotion_labels):

        emotion = emotion_labels[class_id]

        st.write(
            f"{get_icon(emotion)} {emotion.title()}"
        )

    st.markdown("---")

    # --------------------------------------
    # 5. Working methodology
    # --------------------------------------

    st.subheader("How EmotiSense Works")

    st.write(f"""
    1. User enters a text message.

    2. Text preprocessing removes unwanted elements.

    3. TF-IDF converts text into numerical features.

    4. {algorithm_name} predicts the emotion.

    5. Streamlit displays the prediction and model scores.
    """)

    st.info(
        "The model scores represent estimated classification "
        "probabilities, not a verified measurement of someone's "
        "actual emotional state."
    )
# ==========================================
# 9. FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; padding:15px;">
        <h4>🧠 EmotiSense</h4>
        <p>NLP-Based Text Emotion Classification System</p>
        <p>Developed with ❤️ by <b>Okasha Amir</b></p>
        <p style="font-size:12px; color:gray;">
            Machine Learning | Natural Language Processing | Python
        </p>
    </div>
    """,
    unsafe_allow_html=True
)





