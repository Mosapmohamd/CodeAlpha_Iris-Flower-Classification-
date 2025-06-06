import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle as pkl
from io import BytesIO

# ==============================================
# CONSTANTS AND CONFIGURATION
# ==============================================

MODEL_NAMES = {
    "Logistic Regression": "Logistic_Regression_model.pkl",
    "Random Forest": "Random_Forest_model.pkl", 
    "SVM": "SVM_model.pkl",
    "XGBoost": "XGBoost_model.pkl"
}

FEATURES = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# ==============================================
# STYLING AND PAGE CONFIG
# ==============================================

st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #000000, #130c8b) !important;
        color: white !important;
    }
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        color: white;
    }
    .custom-card {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        padding: 1.5rem;
        background: white;
        margin-bottom: 1.5rem;
        border-left: 4px solid #4f46e5;
    }
    .card-title {
        color: #1e293b;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
    }
    .metric-box {
        flex: 1;
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4f46e5;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
    }
    /* Optimized plot styling */
    .stPlot {
        border-radius: 12px;
    }
    /* Better form styling */
    .stForm {
        background:"Black";
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================
# HELPER FUNCTIONS
# ==============================================

@st.cache_data
def load_data():
    """Load and preprocess the iris dataset."""
    df = pd.read_csv("data/iris.csv")
    df = df.iloc[:, 1:]  # Remove ID column
    df['Species'] = df['Species'].astype('category')
    return df

@st.cache_resource
def load_models():
    """Load all trained models with caching."""
    models = {}
    for name, path in MODEL_NAMES.items():
        try:
            with open(path, 'rb') as f:
                models[name] = pkl.load(f)
        except FileNotFoundError:
            st.error(f"Model file not found: {path}")
    return models

def plot_to_buffer(fig):
    """Convert matplotlib figure to buffer for better performance."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)  # Close the figure to free memory
    return buf

def plot_feature_distribution(data, feature):
    """Generate feature distribution plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='Species', y=feature, data=data, ax=ax)
    plt.title(f"{feature} Distribution by Species")
    plt.xticks(rotation=45)
    return fig

def plot_accuracy_comparison(metrics_df):
    """Generate accuracy comparison plot."""
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x='Model', y='Accuracy', data=metrics_df, palette="viridis", ax=ax)
    plt.ylim(0, 1)
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=45)
    return fig

def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    """Generate confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6, 4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", 
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(f"{model_name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig

def plot_prediction_probabilities(proba, classes):
    """Generate prediction probabilities plot."""
    proba_df = pd.DataFrame({
        "Species": classes,
        "Probability": proba[0]
    }).sort_values("Probability", ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Species', y='Probability', data=proba_df, palette="viridis", ax=ax)
    plt.ylim(0, 1)
    plt.title("Prediction Probabilities")
    return fig, proba_df

# ==============================================
# DATA AND MODEL LOADING
# ==============================================

df = load_data()
le = LabelEncoder()
df['Species_encoded'] = le.fit_transform(df['Species'])

# Split data
X = df[FEATURES]
y = df['Species_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = load_models()

# ==============================================
# SIDEBAR
# ==============================================

with st.sidebar:
    # Logo and Title with better spacing
    st.markdown("""
    <div style="text-align:center; margin-bottom:30px;">
        <h1 style="font-size:28px; margin-bottom:0; color:white;">🌸 Iris Classifier</h1>
        <p style="font-size:14px; color:#cbd5e1;">Interactive Flower Classification</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation Links with Icons and Hover Effects
    st.markdown("""
    <style>
        .nav-item {
            padding: 10px 15px;
            margin: 5px 0;
            border-radius: 8px;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
        }
        .nav-item:hover {
            background-color: rgba(255, 255, 255, 0.1);
            transform: translateX(5px);
        }
        .nav-item.active {
            background-color: #4f46e5;
        }
        .nav-icon {
            margin-right: 10px;
            font-size: 18px;
        }
        .nav-text {
            font-size: 16px;
        }
    </style>
    
    <div style="margin-bottom:30px;">
        <a href="#data-exploration" class="nav-item">
            <span class="nav-icon">📊</span>
            <span class="nav-text">Data Exploration</span>
        </a>
        <a href="#model-performance" class="nav-item">
            <span class="nav-icon">📈</span>
            <span class="nav-text">Model Performance</span>
        </a>
        <a href="#live-predictions" class="nav-item">
            <span class="nav-icon">🔮</span>
            <span class="nav-text">Live Predictions</span>
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Selection Dropdown (if needed elsewhere in app)
    selected_model = st.selectbox(
        "Select Model", 
        list(MODEL_NAMES.keys()),
        key="sidebar_model_select",
        help="Choose which model to analyze"
    )
    
    # Divider with better styling
    st.markdown("""
    <hr style="border:0.5px solid rgba(255,255,255,0.1); margin:25px 0;">
    """, unsafe_allow_html=True)
    
    # Footer with Version Info
    st.markdown("""
    <div style="font-size:12px; color:#94a3b8;">
        <p style="margin:5px 0;">Dataset: Iris Flowers</p>
        <p style="margin:5px 0;">Last updated: June 2025</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================
# MAIN CONTENT
# ==============================================

st.title("Iris Flower Classification Dashboard")
st.markdown("""
This interactive dashboard allows you to explore the Iris flower dataset, compare different classification models, 
and make live predictions of Iris species based on flower measurements.
""")
st.markdown("---")

# Section 1: Data Exploration
st.header("📊 Data Exploration", anchor="data-exploration")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Dataset Overview")
    st.write(f"Total samples: {len(df)}")
    st.write("Features:")
    for feature in FEATURES:
        st.write(f"- {feature} (cm)")
    st.write("Target: Species")
    
    st.subheader("Class Distribution")
    species_counts = df['Species'].value_counts().reset_index()
    species_counts.columns = ['Species', 'Count']
    st.dataframe(species_counts.style.background_gradient(cmap="Blues"))

with col2:
    st.subheader("Feature Distribution by Species")
    feature = st.selectbox("Select feature to visualize", FEATURES)
    
    fig = plot_feature_distribution(df, feature)
    st.pyplot(fig)

# Pairplot - Only show if user expands
st.subheader("Feature Relationships")
with st.expander("Show Pairplot", expanded=False):
    fig = sns.pairplot(df, hue='Species')
    st.pyplot(fig.fig)

st.markdown("---")

# Section 2: Model Performance
st.header("📈 Model Performance", anchor="model-performance")

# Calculate metrics
if models:
    metrics_data = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        metrics_data.append({
            "Model": name,
            "Accuracy": round(accuracy, 3),
            "Precision (weighted avg)": round(report['weighted avg']['precision'], 3),
            "Recall (weighted avg)": round(report['weighted avg']['recall'], 3),
            "F1 Score (weighted avg)": round(report['weighted avg']['f1-score'], 3)
        })

    metrics_df = pd.DataFrame(metrics_data)

    # Display metrics
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Performance Metrics")
        st.dataframe(metrics_df.style.background_gradient(cmap="Blues"), height=300)

    with col2:
        st.subheader("Accuracy Comparison")
        fig = plot_accuracy_comparison(metrics_df)
        st.pyplot(fig)

    # Confusion matrices
    st.subheader("Confusion Matrices")
    model_choice = st.selectbox("Select model to view confusion matrix", 
                              list(models.keys()), 
                              key="confusion_matrix_model")

    y_pred = models[model_choice].predict(X_test)
    fig = plot_confusion_matrix(y_test, y_pred, le.classes_, model_choice)
    st.pyplot(fig)

else:
    st.warning("No models were loaded. Please check the model files.")

st.markdown("---")

# Section 3: Live Predictions
st.header("🔮 Live Predictions", anchor="live-predictions")

with st.form("prediction_form"):
    st.subheader("Make New Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.5, 0.1)
    
    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)
    
    model_choice = st.selectbox("Select Model for Prediction", 
                               options=list(models.keys()) if models else [])
    
    submitted = st.form_submit_button("Predict Species")
    
    if submitted and models:
        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                columns=FEATURES)
        
        model = models[model_choice]
        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)
        
        species = le.inverse_transform(prediction)[0]
        
        st.success(f"Predicted Species: **{species}**")
        
        # Show prediction probabilities
        st.subheader("Prediction Probabilities")
        fig, proba_df = plot_prediction_probabilities(proba, le.classes_)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(proba_df.style.background_gradient(cmap="YlOrBr"))
        
        with col2:
            st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>© 2025 Iris Flower Classification Dashboard | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)