
import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Load trained model
model = joblib.load("spam_model.pkl")

# Load dataset for visualization
data = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Count messages
message_counts = data["label"].value_counts()

# Prepare test data for confusion matrix
X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Generate predictions using saved model
test_predictions = model.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=["ham", "spam"]
)

# Page configuration
st.set_page_config(
    page_title="Spam Detection System",
    page_icon="📩",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    color: #1f3c88;
    font-size: 36px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #555555;
    font-size: 16px;
}

.result {
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Application heading
st.markdown(
    '<p class="title">📩 Spam Detection System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Machine Learning Based Message Classification</p>',
    unsafe_allow_html=True
)

st.divider()

# Dataset Overview
st.subheader("📊 Dataset Overview")

total_messages = len(data)
spam_messages = message_counts.get("spam", 0)
ham_messages = message_counts.get("ham", 0)

col1, col2, col3 = st.columns(3)

col1.metric("Total Messages", total_messages)
col2.metric("Spam Messages", spam_messages)
col3.metric("Ham Messages", ham_messages)

# Bar Chart
st.write("### Spam vs Ham Distribution")

chart_data = pd.DataFrame({
    "Category": ["Spam", "Ham"],
    "Messages": [spam_messages, ham_messages]
})

st.bar_chart(
    chart_data,
    x="Category",
    y="Messages"
)

st.divider()

# Confusion Matrix
st.write("### Confusion Matrix")

confusion_data = pd.DataFrame(
    cm,
    index=["Actual Ham", "Actual Spam"],
    columns=["Predicted Ham", "Predicted Spam"]
)

st.dataframe(confusion_data)

st.caption(
    "Rows represent actual categories and columns represent "
    "the model's predicted categories."
)

st.divider()

# Project information
st.write(
    "Enter a message below. Our machine learning model "
    "will predict whether it is Spam or Ham."
)

# Function to clear the message input
def clear_message():
    st.session_state["message_input"] = ""

# Message input
message = st.text_area(
    "Enter your message here:",
    height=150,
    placeholder="Type or paste your message...",
    key="message_input"
)

# Buttons
col1, col2 = st.columns(2)

with col1:
    check = st.button(
        "🔍 Check Message",
        use_container_width=True
    )

with col2:
    st.button(
        "🗑️ Clear",
        use_container_width=True,
        on_click=clear_message
    )




if check:

    if message.strip() == "":
        st.warning("Please enter a message first.")

    else:
        # Get prediction probabilities
        probabilities = model.predict_proba([message])[0]

        # Find the spam class position
        spam_index = list(model.classes_).index("spam")

        # Get spam probability
        spam_probability = probabilities[spam_index]

        # Apply threshold of 0.3
        if spam_probability >= 0.3:
            prediction = "spam"
        else:
            prediction = "ham"

        # Display result
        st.divider()
        st.subheader("Prediction Result")

        if prediction == "spam":
            st.error("⚠️ SPAM MESSAGE DETECTED")
            st.write("This message has been classified as Spam.")

        else:
            st.success("✅ HAM MESSAGE")
            st.write("This message has been classified as Ham (Not Spam).")

        # Display spam probability
        st.write(
            "Spam Probability:",
            f"{spam_probability * 100:.2f}%"
        )


# Footer
st.divider()

st.caption(
    "Developed using Python, Scikit-learn, TF-IDF, "
    "Naive Bayes and Streamlit."
)
