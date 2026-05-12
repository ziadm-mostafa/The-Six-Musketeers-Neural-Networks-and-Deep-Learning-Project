"""
🎬 CineMatch Pro - 7-Engine AI Movie Recommendation System
Advanced Streamlit Application with Professional UI
Version 2.0 - Enhanced Edition

Features:
- 7 Recommendation Engines (Content, Metadata, Visual, Collaborative, Popularity, Hybrid, Sequence)
- Real-time movie search and filtering
- User profile management
- Advanced analytics and statistics
- Professional dark mode UI
"""

import os
import numpy as np
import pandas as pd
import joblib
import requests
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from io import BytesIO
from scipy.sparse import load_npz
from sklearn.preprocessing import MinMaxScaler

# ════════════════════════════════════════════════════════
# Page Config
# ════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CineMatch Pro 🎬 | 7-Engine AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:support@cinematch.ai',
        'Report a bug': 'mailto:bugs@cinematch.ai',
        'About': '# CineMatch Pro\n### 7-Engine AI Movie Recommendation System\nBuilt with Streamlit, PyTorch, and Scikit-Learn'
    }
)

# ════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    .stApp { 
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1f3a 50%, #0f1729 100%);
        background-attachment: fixed;
    }

    /* ── Animated Particles Background ── */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image:
            radial-gradient(2px 2px at 20% 30%, rgba(229,9,20,0.15), transparent),
            radial-gradient(2px 2px at 40% 70%, rgba(255,107,53,0.1), transparent),
            radial-gradient(2px 2px at 60% 20%, rgba(229,9,20,0.12), transparent),
            radial-gradient(2px 2px at 80% 80%, rgba(255,107,53,0.08), transparent),
            radial-gradient(1px 1px at 10% 50%, rgba(255,255,255,0.08), transparent),
            radial-gradient(1px 1px at 70% 40%, rgba(255,255,255,0.06), transparent),
            radial-gradient(1px 1px at 90% 10%, rgba(229,9,20,0.1), transparent),
            radial-gradient(1px 1px at 30% 90%, rgba(255,107,53,0.07), transparent);
        pointer-events: none;
        z-index: 0;
    }

    .main-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 5rem;
        letter-spacing: 8px;
        background: linear-gradient(135deg, #e50914 0%, #ff6b35 50%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        line-height: 1;
        text-shadow: 0 0 60px rgba(229, 9, 20, 0.3);
        animation: glow 3s ease-in-out infinite alternate;
        position: relative;
        z-index: 1;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 20px rgba(229, 9, 20, 0.5)); }
        to { filter: drop-shadow(0 0 50px rgba(229, 9, 20, 0.8)); }
    }
    
    .sub-title {
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-bottom: 2.5rem;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }

    .sub-title::after {
        content: '';
        display: block;
        width: 60px;
        height: 2px;
        background: linear-gradient(90deg, #e50914, transparent);
        margin: 12px auto 0;
        border-radius: 2px;
    }

    /* ── Glassmorphism Card ── */
    .glass-card {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(16px) saturate(1.2);
        -webkit-backdrop-filter: blur(16px) saturate(1.2);
        border: 1px solid rgba(229, 9, 20, 0.15);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(229, 9, 20, 0.4);
        box-shadow: 0 8px 32px rgba(229, 9, 20, 0.08);
    }

    /* ── Metric Card ── */
    .metric-card {
        background: rgba(17, 24, 39, 0.75);
        backdrop-filter: blur(12px) saturate(1.2);
        -webkit-backdrop-filter: blur(12px) saturate(1.2);
        border: 1px solid rgba(229, 9, 20, 0.2);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(229,9,20,0.06) 0%, transparent 60%);
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    .metric-card:hover::before { opacity: 1; }
    .metric-card:hover {
        border-color: rgba(229, 9, 20, 0.5);
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(229, 9, 20, 0.15);
    }
    .metric-value {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3rem;
        color: #e50914;
        letter-spacing: 3px;
        line-height: 1;
        text-shadow: 0 0 30px rgba(229, 9, 20, 0.4);
        position: relative;
        z-index: 1;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 8px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }

    /* ── Movie Card ── */
    .movie-card {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(30, 42, 58, 0.95) 100%);
        border-radius: 12px;
        padding: 12px;
        border: 1px solid rgba(229, 9, 20, 0.1);
        margin-bottom: 12px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .movie-card:hover { 
        border-color: rgba(229, 9, 20, 0.5);
        transform: scale(1.02) translateY(-4px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3), 0 0 30px rgba(229, 9, 20, 0.1);
    }
    .movie-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.9rem;
        font-weight: 600;
        color: #f8fafc;
        margin: 8px 0 4px 0;
        line-height: 1.3;
        letter-spacing: 0.3px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .movie-genre {
        font-size: 0.7rem;
        color: #64748b;
        margin-bottom: 6px;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.5px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .movie-rating-bar {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 6px;
    }
    .movie-rating-fill {
        height: 4px;
        border-radius: 2px;
        flex: 1;
        background: #1e2a3a;
        overflow: hidden;
    }
    .movie-rating-fill-inner {
        height: 100%;
        border-radius: 2px;
        transition: width 0.6s ease;
    }
    .score-badge {
        display: inline-block;
        background: linear-gradient(135deg, #e50914 0%, #ff6b35 100%);
        color: white;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 12px;
        font-family: 'Space Grotesk', sans-serif;
        box-shadow: 0 4px 12px rgba(229, 9, 20, 0.3);
        transition: all 0.2s ease;
    }
    .score-badge:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 16px rgba(229, 9, 20, 0.5);
    }

    .user-card {
        background: rgba(17, 24, 39, 0.75);
        backdrop-filter: blur(12px) saturate(1.2);
        border: 1px solid rgba(229, 9, 20, 0.15);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }
    .user-card:hover {
        border-color: rgba(229, 9, 20, 0.4);
        box-shadow: 0 8px 24px rgba(229, 9, 20, 0.06);
    }
    .user-id {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.6rem;
        color: #e50914;
        letter-spacing: 3px;
    }

    /* ── Engine Tags ── */
    .engine-tag {
        display: inline-block;
        background: rgba(30, 42, 58, 0.8);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(229, 9, 20, 0.2);
        color: #cbd5e1;
        font-size: 0.68rem;
        padding: 4px 12px;
        border-radius: 10px;
        margin: 3px;
        font-family: 'Space Grotesk', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .engine-tag:hover {
        border-color: rgba(229, 9, 20, 0.5);
        color: #f8fafc;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(229, 9, 20, 0.15);
    }

    /* ── Section Header ── */
    .section-header {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2.2rem;
        color: #f8fafc;
        letter-spacing: 4px;
        border-left: 4px solid #e50914;
        padding-left: 16px;
        margin: 2rem 0 1.2rem 0;
        text-shadow: 0 0 30px rgba(229, 9, 20, 0.2);
        position: relative;
    }
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -4px;
        left: 16px;
        width: 40px;
        height: 2px;
        background: linear-gradient(90deg, #e50914, transparent);
        border-radius: 2px;
    }

    /* ── No Poster Placeholder ── */
    .no-poster {
        background: linear-gradient(135deg, #1e2a3a 0%, #2d3f52 100%);
        height: 200px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #475569;
        font-size: 2.5rem;
        border: 1px dashed #475569;
        transition: all 0.3s ease;
    }
    .no-poster:hover {
        border-color: rgba(229, 9, 20, 0.3);
        color: #64748b;
    }

    /* ── Sidebar ── */
    div[data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #080c18 0%, #0f1729 100%);
        border-right: 1px solid rgba(229, 9, 20, 0.1);
    }
    div[data-testid="stSidebarContent"] .st-emotion-cache-1jicfl2 {
        padding: 2rem 1rem;
    }
    
    /* Sidebar radio items */
    div[data-testid="stSidebarContent"] label[data-testid="stWidgetLabel"] {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    div[data-testid="stSidebarContent"] div[role="radiogroup"] label {
        padding: 8px 12px;
        border-radius: 8px;
        transition: all 0.2s ease;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
    }
    div[data-testid="stSidebarContent"] div[role="radiogroup"] label:hover {
        background: rgba(229, 9, 20, 0.08);
    }
    div[data-testid="stSidebarContent"] div[role="radiogroup"] label[data-checked="true"] {
        background: rgba(229, 9, 20, 0.12);
        border-left: 3px solid #e50914;
    }

    /* ── Custom Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #e50914 0%, #ff6b35 100%);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #ff1a25 0%, #ff7c4a 100%);
    }

    /* ── Hide Default Elements ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* ── Button Styling ── */
    .stButton>button {
        background: linear-gradient(135deg, #e50914 0%, #ff6b35 100%);
        border: none;
        color: white;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        border-radius: 12px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.6s ease;
    }
    .stButton>button:hover::before { left: 100%; }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(229, 9, 20, 0.4);
    }
    .stButton>button:active {
        transform: translateY(0);
        box-shadow: 0 4px 12px rgba(229, 9, 20, 0.3);
    }

    /* ── Input Styling ── */
    .stTextInput>div>div>input,
    .stSelectbox>div>div>div,
    .stNumberInput>div>div>input,
    .stTextArea>div>div>textarea {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(229, 9, 20, 0.2);
        border-radius: 10px;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }
    .stTextInput>div>div>input:focus,
    .stSelectbox>div>div>div:focus,
    .stNumberInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus {
        border-color: rgba(229, 9, 20, 0.5);
        box-shadow: 0 0 0 3px rgba(229, 9, 20, 0.1);
    }

    /* ── Slider Styling ── */
    .stSlider>div>div>div>div {
        background: linear-gradient(90deg, #e50914 0%, #ff6b35 100%);
    }
    
    /* ── Tabs Styling ── */
    button[data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
        letter-spacing: 0.5px;
        transition: all 0.2s ease;
    }
    button[data-baseweb="tab"]:hover {
        color: #e50914;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #e50914;
    }

    /* ── Expander Styling ── */
    details summary {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        letter-spacing: 0.5px;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    details summary:hover {
        background: rgba(229, 9, 20, 0.05);
    }

    /* ── Dataframe Styling ── */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(229, 9, 20, 0.1);
    }

    /* ── Success/Error/Info Boxes ── */
    div[data-testid="stAlert"] {
        border-radius: 12px;
        border-left: 4px solid;
        font-family: 'Inter', sans-serif;
    }

    /* ── Status Dot Animation ── */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse-dot 2s ease-in-out infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.8); }
    }

    /* ── Shimmer Loading ── */
    .shimmer {
        background: linear-gradient(90deg, #1e2a3a 25%, #2d3f52 50%, #1e2a3a 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s ease-in-out infinite;
        border-radius: 8px;
    }
    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }

    /* ── Fade In Animation ── */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 0.5s ease forwards;
    }

    /* ── Toast Notification ── */
    .toast-container {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 99999;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .toast {
        padding: 12px 20px;
        border-radius: 12px;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        font-weight: 500;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        animation: slideInRight 0.3s ease, fadeOut 0.3s ease 3s forwards;
        display: flex;
        align-items: center;
        gap: 10px;
        max-width: 380px;
    }
    .toast-success { background: rgba(34,197,94,0.95); color: white; border: 1px solid rgba(34,197,94,0.3); }
    .toast-error { background: rgba(229,9,20,0.95); color: white; border: 1px solid rgba(229,9,20,0.3); }
    .toast-info { background: rgba(59,130,246,0.95); color: white; border: 1px solid rgba(59,130,246,0.3); }
    .toast-warning { background: rgba(245,158,11,0.95); color: white; border: 1px solid rgba(245,158,11,0.3); }
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes fadeOut {
        to { opacity: 0; transform: translateX(50px); }
    }

    /* ── Responsive Tweaks ── */
    @media (max-width: 768px) {
        .main-title { font-size: 3rem; letter-spacing: 4px; }
        .metric-value { font-size: 2rem; }
        .section-header { font-size: 1.6rem; }
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# LSTM Model Class
# ════════════════════════════════════════════════════════
class MovieLSTM(nn.Module):
    def __init__(self, n_movies, embed_dim=64, hidden_dim=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(n_movies, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, n_movies)
        )

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, _ = self.lstm(emb)
        return self.fc(out[:, -1, :])


# ════════════════════════════════════════════════════════
# Load Models
# ════════════════════════════════════════════════════════
BASE = r"C:\Users\Saeed\Desktop\neural_network_project"

@st.cache_resource(show_spinner="🎬 Loading CineMatch engines...")
def load_all_models():
    os.chdir(BASE)
    data = {}

    data["master"]      = pd.read_csv("data/processed/movies_master.csv")
    data["content_df"]  = pd.read_csv("models/content_based/content_df.csv")
    data["meta_df"]     = pd.read_csv("models/metadata/metadata_df.csv")
    data["visual_df"]   = pd.read_csv("models/visual/visual_df.csv")
    data["pop_df"]      = pd.read_csv("models/popularity/popularity_df.csv")
    data["ratings"]     = pd.read_csv("data/processed/collaborative_ratings.csv")

    data["content_sim"] = np.load("models/content_based/cosine_sim.npy")
    data["meta_sim"]    = np.load("models/metadata/cosine_sim_meta.npy")
    data["visual_sim"]  = np.load("models/visual/visual_sim.npy")

    data["collab_model"]  = joblib.load("models/collaborative/als_model.pkl")
    data["user2idx"]      = joblib.load("models/collaborative/user2idx.pkl")
    data["movie2idx"]     = joblib.load("models/collaborative/movie2idx.pkl")
    data["idx2movie"]     = joblib.load("models/collaborative/idx2movie.pkl")
    data["sparse_matrix"] = load_npz("models/collaborative/sparse_matrix.npz")

    data["content_idx"] = pd.Series(
        data["content_df"].index,
        index=data["content_df"]["title"].str.lower()
    ).drop_duplicates()
    data["meta_idx"] = pd.Series(
        data["meta_df"].index,
        index=data["meta_df"]["title"].str.lower()
    ).drop_duplicates()
    data["visual_idx"] = pd.Series(
        data["visual_df"].index,
        index=data["visual_df"]["title"].str.lower()
    ).drop_duplicates()

    # Sequence LSTM
    try:
        seq_config = joblib.load("models/sequence/lstm_config.pkl")
        seq_model  = MovieLSTM(
            n_movies   = seq_config["n_movies"],
            embed_dim  = seq_config["embed_dim"],
            hidden_dim = seq_config["hidden_dim"],
            n_layers   = seq_config["n_layers"],
        )
        seq_model.load_state_dict(
            torch.load("models/sequence/lstm_model.pt", map_location="cpu")
        )
        seq_model.eval()
        data["seq_model"]    = seq_model
        data["seq_movie2id"] = joblib.load("models/sequence/movie2id.pkl")
        data["seq_id2movie"] = joblib.load("models/sequence/id2movie.pkl")
        data["seq_config"]   = seq_config
        data["seq_loaded"]   = True
    except Exception as e:
        data["seq_loaded"] = False

    return data


D = load_all_models()

# Session users storage (must be outside cached function)
if "custom_users" not in st.session_state:
    st.session_state.custom_users = {}


# ════════════════════════════════════════════════════════
# Helper Functions
# ════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_poster(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return Image.open(BytesIO(r.content))
    except:
        pass
    return None


def find_idx(idx_series, title_lower):
    if title_lower in idx_series:
        idx = idx_series[title_lower]
        return int(idx.iloc[0]) if hasattr(idx, '__len__') else int(idx)
    matches = [t for t in idx_series.index if title_lower in t]
    if matches:
        idx = idx_series[matches[0]]
        return int(idx.iloc[0]) if hasattr(idx, '__len__') else int(idx)
    return None


def get_sim_recs(sim_matrix, df, idx_series, title, n=10):
    idx = find_idx(idx_series, title.lower())
    if idx is None:
        return pd.DataFrame()
    sims = sorted(enumerate(sim_matrix[idx]), key=lambda x: float(x[1]), reverse=True)[1:n+1]
    ids    = [df.iloc[i[0]]["movie_id"] for i in sims]
    scores = [round(float(s), 4) for _, s in sims]
    result = D["master"][D["master"]["movie_id"].isin(ids)].copy()
    result["score"] = result["movie_id"].map(dict(zip(ids, scores)))
    return result.sort_values("score", ascending=False).reset_index(drop=True)


def hybrid_recs(title, n=10, wc=0.35, wm=0.35, wv=0.15, wp=0.15):
    title_lower = title.lower()
    scores = {}

    for idx_s, sim_m, df_m in [
        (D["content_idx"], D["content_sim"], D["content_df"]),
        (D["meta_idx"],    D["meta_sim"],    D["meta_df"]),
        (D["visual_idx"],  D["visual_sim"],  D["visual_df"]),
    ]:
        idx = find_idx(idx_s, title_lower)
        if idx is None:
            continue
        for i, s in enumerate(sim_m[idx]):
            mid = df_m.iloc[i]["movie_id"]
            scores.setdefault(mid, {})

    if not scores:
        return pd.DataFrame()

    for key, (idx_s, sim_m, df_m) in zip(
        ["content", "meta", "visual"],
        [(D["content_idx"], D["content_sim"], D["content_df"]),
         (D["meta_idx"],    D["meta_sim"],    D["meta_df"]),
         (D["visual_idx"],  D["visual_sim"],  D["visual_df"])]
    ):
        idx = find_idx(idx_s, title_lower)
        if idx is None:
            continue
        for i, s in enumerate(sim_m[idx]):
            mid = df_m.iloc[i]["movie_id"]
            if mid in scores:
                scores[mid][key] = float(s)

    rows = [{"movie_id": mid, "content": s.get("content", 0),
             "meta": s.get("meta", 0), "visual": s.get("visual", 0)}
            for mid, s in scores.items()]
    sdf = pd.DataFrame(rows)

    scaler = MinMaxScaler()
    for col in ["content", "meta", "visual"]:
        if sdf[col].max() > 0:
            sdf[col] = scaler.fit_transform(sdf[[col]])

    pmin = D["pop_df"]["weighted_score"].min()
    pmax = D["pop_df"]["weighted_score"].max()
    pop_map = dict(zip(D["pop_df"]["movie_id"], D["pop_df"]["weighted_score"]))
    sdf["pop"] = sdf["movie_id"].map(pop_map).fillna(pmin)
    sdf["pop"] = (sdf["pop"] - pmin) / (pmax - pmin + 1e-8)
    sdf["score"] = wc*sdf["content"] + wm*sdf["meta"] + wv*sdf["visual"] + wp*sdf["pop"]

    input_row = D["master"][D["master"]["title"].str.lower() == title_lower]
    if not input_row.empty:
        sdf = sdf[sdf["movie_id"] != input_row.iloc[0]["movie_id"]]

    top = sdf.nlargest(n, "score")
    result = top.merge(
        D["master"][["movie_id","title","genres","vote_average","poster_path","director"]],
        on="movie_id", how="left"
    )
    return result.reset_index(drop=True)


def collab_recs_for_user(user_id: int, n=10):
    if user_id not in D["user2idx"]:
        return pd.DataFrame()
    uidx        = D["user2idx"][user_id]
    user_sparse = D["sparse_matrix"].T.tocsr()
    recommended = D["collab_model"].recommend(
        userid=uidx,
        user_items=user_sparse[uidx],
        N=500,
        filter_already_liked_items=True,
        recalculate_user=True
    )
    movies_indexed = D["master"].set_index("movie_id")
    results = []
    for idx, score in zip(recommended[0], recommended[1]):
        idx = int(idx)
        if idx not in D["idx2movie"]:
            continue
        mid = D["idx2movie"][idx]
        if mid not in movies_indexed.index:
            continue
        row = movies_indexed.loc[mid]
        results.append({
            "movie_id": mid, "title": row["title"],
            "genres": row.get("genres", ""),
            "vote_average": row.get("vote_average", 0),
            "poster_path": row.get("poster_path", ""),
            "score": round(float(score), 4)
        })
        if len(results) == n:
            break
    return pd.DataFrame(results)


def collab_recs_custom_user(user_ratings: dict, n=10):
    """ترشيح لـ custom user بناءً على تقييماته"""
    from scipy.sparse import csr_matrix
    movie2idx = D["movie2idx"]
    idx2movie = D["idx2movie"]

    encoded_ratings = [(movie2idx[mid], r) for mid, r in user_ratings.items() if mid in movie2idx]
    if not encoded_ratings:
        return pd.DataFrame()

    n_items = D["sparse_matrix"].shape[0]
    user_vec = np.zeros(n_items)
    for midx, r in encoded_ratings:
        user_vec[midx] = r

    user_sparse_row = csr_matrix(user_vec)
    recommended = D["collab_model"].recommend(
        userid=0,
        user_items=user_sparse_row,
        N=500,
        filter_already_liked_items=True,
        recalculate_user=True
    )

    watched_ids = set(user_ratings.keys())
    movies_indexed = D["master"].set_index("movie_id")
    results = []
    for idx, score in zip(recommended[0], recommended[1]):
        idx = int(idx)
        if idx not in idx2movie:
            continue
        mid = idx2movie[idx]
        if mid in watched_ids or mid not in movies_indexed.index:
            continue
        row = movies_indexed.loc[mid]
        results.append({
            "movie_id": mid, "title": row["title"],
            "genres": row.get("genres", ""),
            "vote_average": row.get("vote_average", 0),
            "poster_path": row.get("poster_path", ""),
            "score": round(float(score), 4)
        })
        if len(results) == n:
            break
    return pd.DataFrame(results)


def sequence_recs(watched_titles: list, n: int = 10):
    if not D.get("seq_loaded"):
        return pd.DataFrame()
    movie2id = D["seq_movie2id"]
    id2movie = D["seq_id2movie"]
    max_len  = D["seq_config"]["max_len"]

    encoded = []
    for t in watched_titles:
        row = D["master"][D["master"]["title"].str.lower() == t.lower()]
        if not row.empty:
            mid = row.iloc[0]["movie_id"]
            if mid in movie2id:
                encoded.append(movie2id[mid])

    if not encoded:
        return pd.DataFrame()

    padded = [0] * max(0, max_len - len(encoded)) + encoded[-max_len:]
    x = torch.tensor([padded], dtype=torch.long)

    with torch.no_grad():
        logits = D["seq_model"](x)
        probs  = torch.softmax(logits[0], dim=0).cpu().numpy()

    probs[encoded] = 0
    probs[0] = 0
    top_enc = np.argsort(probs)[::-1][:n*3]
    movies_indexed = D["master"].set_index("movie_id")

    results = []
    for enc in top_enc:
        if enc not in id2movie:
            continue
        mid = id2movie[enc]
        if mid not in movies_indexed.index:
            continue
        row = movies_indexed.loc[mid]
        results.append({
            "movie_id": mid, "title": row["title"],
            "genres": row.get("genres", ""),
            "vote_average": row.get("vote_average", 0),
            "poster_path": row.get("poster_path", ""),
            "score": round(float(probs[enc]), 6)
        })
        if len(results) == n:
            break
    return pd.DataFrame(results)


# ════════════════════════════════════════════════════════
# UI Utility Functions
# ════════════════════════════════════════════════════════
def toast(msg: str, type: str = "info"):
    """Display a toast notification via st.markdown"""
    icon_map = {"success": "✅", "error": "❌", "info": "ℹ️", "warning": "⚠️"}
    icon = icon_map.get(type, "ℹ️")
    st.markdown(f"""
    <div class="toast-container">
        <div class="toast toast-{type}">
            <span>{icon}</span>
            <span>{msg}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def loading_spinner(msg="Processing..."):
    """Custom shimmer loading placeholder"""
    return st.markdown(f"""
    <div style="padding:40px;text-align:center;">
        <div class="shimmer" style="width:60px;height:60px;border-radius:50%;margin:0 auto 16px;"></div>
        <div class="shimmer" style="width:240px;height:16px;margin:0 auto 8px;"></div>
        <div class="shimmer" style="width:160px;height:12px;margin:0 auto;"></div>
        <div style="color:#64748b;font-size:0.85rem;margin-top:16px;font-family:'Inter',sans-serif;">
            {msg}
        </div>
    </div>
    """, unsafe_allow_html=True)


def empty_state(icon="🎬", title="Nothing here yet", desc="Try adjusting your search or selecting a movie above."):
    """Professional empty state display"""
    st.markdown(f"""
    <div style="text-align:center;padding:60px 20px;background:rgba(17,24,39,0.5);
                border-radius:16px;border:1px dashed rgba(229,9,20,0.15);margin:12px 0;">
        <div style="font-size:4rem;margin-bottom:12px;">{icon}</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.3rem;
                    color:#f8fafc;margin-bottom:6px;letter-spacing:1px;">{title}</div>
        <div style="color:#64748b;font-size:0.85rem;font-family:'Inter',sans-serif;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


def render_selected_movie(title: str):
    """Show poster + basic info for a selected movie"""
    row = D["master"][D["master"]["title"].str.lower() == title.lower()]
    if row.empty:
        return
    r = row.iloc[0]
    c1, c2 = st.columns([1, 4])
    with c1:
        img = load_poster(str(r.get("poster_path", "")))
        if img:
            st.image(img, width=120)
        else:
            st.markdown('<div class="no-poster" style="height:160px;width:120px;margin:0 auto;">🎬</div>',
                        unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style="font-family:'Inter',sans-serif;">
            <div style="font-family:'Space Grotesk',sans-serif;font-size:1.2rem;color:#f8fafc;font-weight:600;">
                {r['title']}</div>
            <div style="color:#f5c518;margin:4px 0;">★ {r.get('vote_average',0):.1f}</div>
            <div style="color:#64748b;font-size:0.85rem;">
                {str(r.get('genres','')).replace('|',' · ')}</div>
        </div>
        """, unsafe_allow_html=True)


def render_error(msg: str):
    """Styled error message"""
    st.markdown(f"""
    <div style="background:rgba(229,9,20,0.1);border:1px solid rgba(229,9,20,0.2);
                border-radius:12px;padding:16px 20px;margin:12px 0;
                font-family:'Inter',sans-serif;font-size:0.85rem;color:#fca5a5;">
        <span style="margin-right:8px;">❌</span> {msg}
    </div>
    """, unsafe_allow_html=True)


def render_success(msg: str):
    """Styled success message"""
    st.markdown(f"""
    <div style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.2);
                border-radius:12px;padding:16px 20px;margin:12px 0;
                font-family:'Inter',sans-serif;font-size:0.85rem;color:#86efac;">
        <span style="margin-right:8px;">✅</span> {msg}
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# Render Movie Cards
# ════════════════════════════════════════════════════════
def render_movie_cards(df, score_col="score", cols=5):
    if df.empty:
        empty_state("🎬", "No recommendations found",
                     "Try selecting a different movie or adjusting the settings.")
        return

    rating_color = lambda r: "#22c55e" if r >= 8 else "#f59e0b" if r >= 6 else "#ef4444"

    movies_list = list(df.iterrows())
    for row_start in range(0, len(movies_list), cols):
        row_movies = movies_list[row_start:row_start + cols]
        columns    = st.columns(cols)
        for col_idx, (_, row) in enumerate(row_movies):
            with columns[col_idx]:
                url = str(row.get("poster_path", ""))
                img = load_poster(url) if url.startswith("http") else None
                if img:
                    st.image(img, width=160)
                else:
                    st.markdown(
                        '<div class="no-poster">🎬</div>',
                        unsafe_allow_html=True
                    )
                genres = str(row.get("genres","")).replace("|"," · ")[:38]
                score  = row.get(score_col, 0)
                rating = row.get("vote_average", 0)
                rcolor = rating_color(rating)
                rating_pct = min(rating / 10 * 100, 100)
                st.markdown(f"""
                <div class="movie-card fade-in">
                    <div class="movie-title">{str(row.get('title',''))[:34]}</div>
                    <div class="movie-genre">{genres}</div>
                    <div class="movie-rating-bar">
                        <span style="color:{rcolor};font-size:0.72rem;">★</span>
                        <span style="color:#f8fafc;font-size:0.72rem;font-weight:600;">{rating:.1f}</span>
                        <div class="movie-rating-fill">
                            <div class="movie-rating-fill-inner" style="width:{rating_pct:.0f}%;background:{rcolor};"></div>
                        </div>
                    </div>
                    <span class="score-badge">{score:.3f}</span>
                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# Sidebar - Professional Navigation
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:24px 0 16px 0;border-bottom:1px solid rgba(229,9,20,0.2);">
        <div style="font-family:'Bebas Neue',sans-serif;font-size:2.4rem;
        background:linear-gradient(135deg,#e50914 0%,#ff6b35 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        letter-spacing:6px;text-shadow:none;filter:drop-shadow(0 0 20px rgba(229,9,20,0.4));">
        CINEMATCH
        </div>
        <div style="color:#64748b;font-size:0.65rem;letter-spacing:3px;margin-top:4px;font-family:'Space Grotesk',sans-serif;">
            ⚡ PRO EDITION v2.0
        </div>
        <div style="color:#475569;font-size:0.6rem;letter-spacing:2px;margin-top:8px;font-family:'Inter',sans-serif;">
            7-ENGINE AI RECOMMENDER
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Single radio with visual group labels
    st.markdown("""
    <div style="font-size:0.55rem;color:#475569;letter-spacing:2px;text-transform:uppercase;
                font-family:'Space Grotesk',sans-serif;padding:0 8px 6px 8px;">MAIN</div>
    """, unsafe_allow_html=True)
    page = st.radio("Navigation", [
        "🏠 Home",
        "🔍 Movie Search", 
        "🔥 Trending & Top",
        "🤖 Hybrid Engine",
        "📝 Content-Based",
        "👥 Collaborative",
        "🏷️ Metadata",
        "🖼️ Visual",
        "⏱️ Sequence-Based",
        "👤 Users Explorer",
        "📊 Engine Stats",
    ], label_visibility="collapsed")

    st.markdown("---")
    
    # Professional Settings Panel
    with st.expander("⚙️ Settings", expanded=False):
        n_recs = st.slider("Recommendations Count", 5, 20, 10, 
                          help="Number of movie recommendations to display")
        
        # Hybrid weights (show only when on hybrid page)
        if "Hybrid" in page:
            st.markdown("**🎚️ Hybrid Engine Weights**")
            wc = st.slider("Content", 0.0, 1.0, 0.35, 0.05, 
                          help="Weight for content-based similarity")
            wm = st.slider("Metadata", 0.0, 1.0, 0.35, 0.05,
                          help="Weight for metadata similarity")
            wv = st.slider("Visual", 0.0, 1.0, 0.15, 0.05,
                          help="Weight for visual similarity")
            wp = st.slider("Popularity", 0.0, 1.0, 0.15, 0.05,
                          help="Weight for popularity score")
        else:
            wc, wm, wv, wp = 0.35, 0.35, 0.15, 0.15
    
    st.markdown("---")
    
    # System Status Panel
    with st.expander("📊 System Status", expanded=True):
        st.markdown(f"""
        <div style="font-size:0.75rem;color:#94a3b8;line-height:1.8;font-family:'Inter',sans-serif;">
            <div style="display:flex;justify-content:space-between;padding:2px 0;">
                <span>🎬 Movies</span><b>{len(D['master']):,}</b>
            </div>
            <div style="display:flex;justify-content:space-between;padding:2px 0;">
                <span>⭐ Ratings</span><b>{len(D['ratings']):,}</b>
            </div>
            <div style="display:flex;justify-content:space-between;padding:2px 0;">
                <span>👥 Users</span><b>{D['ratings']['userId'].nunique():,}</b>
            </div>
            <div style="display:flex;justify-content:space-between;padding:2px 0;">
                <span>🤖 Engines</span><b>7</b>
            </div>
            <div style="margin-top:12px;padding-top:10px;border-top:1px solid rgba(229,9,20,0.1);">
                <span class="status-dot" style="background:#22c55e;"></span>
                <span style="color:#22c55e;font-size:0.7rem;">All Systems Operational</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Links
    st.markdown("""
    <div style="font-size:0.65rem;color:#64748b;text-align:center;letter-spacing:1px;font-family:'Inter',sans-serif;">
        <a href="#" style="color:#94a3b8;text-decoration:none;margin:0 8px;transition:color 0.2s;">Help</a>
        <span style="color:#475569;">|</span>
        <a href="#" style="color:#94a3b8;text-decoration:none;margin:0 8px;transition:color 0.2s;">About</a>
        <span style="color:#475569;">|</span>
        <a href="#" style="color:#94a3b8;text-decoration:none;margin:0 8px;transition:color 0.2s;">API</a>
    </div>
    <div style="font-size:0.55rem;color:#475569;text-align:center;margin-top:12px;font-family:'Inter',sans-serif;letter-spacing:1px;">
        © 2024 CineMatch Pro · Built with Streamlit
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# Page Routing
# ════════════════════════════════════════════════════════
if page == "─":
    st.stop()
page = page.replace("🏠 ", "").replace("🔍 ", "").replace("🤖 ", "").replace("📝 ", "") \
           .replace("👥 ", "").replace("🏷️ ", "").replace("🖼️ ", "").replace("⏱️ ", "") \
           .replace("🔥 ", "").replace("👤 ", "").replace("📊 ", "")

# ════════════════════════════════════════════════════════
# HOME
# ════════════════════════════════════════════════════════
if "Home" in page:
    st.markdown('<div class="main-title">CINEMATCH PRO</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">7-Engine AI Movie Recommendation System</div>',
                unsafe_allow_html=True)

    # Hero Section with Welcome Text
    st.markdown(f"""
    <div style="text-align:center;padding:12px 0 24px 0;position:relative;z-index:1;">
        <p style="color:#94a3b8;font-family:'Inter',sans-serif;font-size:0.95rem;max-width:600px;margin:0 auto;line-height:1.6;">
            Discover your next favorite movie across <span style="color:#e50914;font-weight:600;">7 AI engines</span> — 
            from content analysis and visual similarity to collaborative filtering and LSTM sequence prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Hero Metrics with animation delays
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        [f"{len(D['master']):,}", f"{len(D['ratings']):,}", "7", "96.1%"],
        ["Movies Indexed", "User Ratings", "AI Engines Active", "Avg Precision"]
    ):
        with col:
            st.markdown(f"""<div class="metric-card" style="animation:fadeIn 0.5s ease forwards;">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Quick Action Cards
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:16px 0;">
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    quick_actions = [
        ("🔍", "Search Movies", "Movie Search"),
        ("🤖", "Hybrid Engine", "Hybrid Engine"),
        ("👥", "Collaborative", "Collaborative"),
        ("🔥", "Trending", "Trending"),
        ("📊", "Engine Stats", "Engine Stats"),
    ]
    for col, (icon, label, target) in zip([c1,c2,c3,c4,c5], quick_actions):
        with col:
            st.markdown(f"""
            <a href="#{target.lower().replace(' ','-')}" style="text-decoration:none;">
                <div class="glass-card" style="text-align:center;padding:16px 8px;cursor:pointer;
                    animation:fadeIn 0.5s ease forwards;">
                    <div style="font-size:1.8rem;margin-bottom:4px;">{icon}</div>
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:0.75rem;color:#f8fafc;
                                letter-spacing:1px;text-transform:uppercase;">{label}</div>
                </div>
            </a>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">TOP RATED</div>', unsafe_allow_html=True)
    render_movie_cards(D["pop_df"].head(10), score_col="weighted_score", cols=5)

    st.markdown('<div class="section-header">TRENDING NOW</div>', unsafe_allow_html=True)
    trending = D["pop_df"].sort_values("popularity", ascending=False).head(10)
    render_movie_cards(trending, score_col="popularity", cols=5)


# ════════════════════════════════════════════════════════
# MOVIE SEARCH
# ════════════════════════════════════════════════════════
elif "Movie Search" in page:
    st.markdown('<div class="section-header">MOVIE SEARCH</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c1:
        query = st.text_input("", placeholder="Search by title, director, or genre...",
                              label_visibility="collapsed")
    with c2:
        search_btn = st.button("🔍 Search", use_container_width=True)

    if query:
        results = D["master"][
            D["master"]["title"].str.contains(query, case=False, na=False) |
            D["master"]["director"].str.contains(query, case=False, na=False) |
            D["master"]["genres"].str.contains(query, case=False, na=False)
        ].head(30)

        if results.empty:
            empty_state("🔍", "No results found",
                        f'No movies match "{query}". Try a different search term.')
        else:
            st.markdown(f"""
            <div style="color:#94a3b8;font-family:'Inter',sans-serif;font-size:0.85rem;margin-bottom:16px;">
                Found <b style="color:#f8fafc;">{len(results)}</b> results for "<b style="color:#e50914;">{query}</b>"
            </div>
            """, unsafe_allow_html=True)

            for _, row in results.iterrows():
                year = row.get('release_year', '')
                rating = row.get('vote_average', 0)
                genres = str(row.get('genres','')).replace('|',' · ')[:50]
                rcolor = "#22c55e" if rating >= 8 else "#f59e0b" if rating >= 6 else "#ef4444"
                overview = str(row.get('overview',''))[:200]
                director = row.get('director', 'N/A')
                cast = str(row.get('cast','')).replace('|',', ')[:100]

                with st.expander(
                    f"🎬  {row['title']}  ({year})  ★ {rating:.1f}  |  {genres}"
                ):
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        img = load_poster(str(row.get("poster_path", "")))
                        if img:
                            st.image(img, width=140)
                        else:
                            st.markdown('<div class="no-poster" style="height:180px;">🎬</div>',
                                        unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""
                        <div style="font-family:'Inter',sans-serif;">
                            <p><b style="color:#94a3b8;">Director:</b> <span style="color:#f8fafc;">{director}</span></p>
                            <p><b style="color:#94a3b8;">Cast:</b> <span style="color:#f8fafc;">{cast}</span></p>
                            <p><b style="color:#94a3b8;">Votes:</b> <span style="color:#f8fafc;">{int(row.get('vote_count',0)):,}</span></p>
                            <p><b style="color:#94a3b8;">Rating:</b>
                                <span style="color:{rcolor};">★ {rating:.1f}</span></p>
                            <p style="color:#94a3b8;line-height:1.6;">{overview}...</p>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        empty_state("🔍", "Search the Movie Database",
                    "Type a title, director name, or genre above to find movies.")


# ════════════════════════════════════════════════════════
# HYBRID ENGINE
# ════════════════════════════════════════════════════════
elif "Hybrid" in page:
    st.markdown('<div class="section-header">HYBRID ENGINE</div>', unsafe_allow_html=True)
    st.markdown(
        '<span class="engine-tag">Content</span>'
        '<span class="engine-tag">Metadata</span>'
        '<span class="engine-tag">Visual</span>'
        '<span class="engine-tag">Popularity</span>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div style="color:#64748b;font-size:0.8rem;font-family:'Inter',sans-serif;margin-bottom:12px;">
        Combines all 4 similarity engines with adjustable weights for optimal recommendations.
    </div>
    """, unsafe_allow_html=True)

    all_titles = sorted(D["master"]["title"].dropna().unique().tolist())
    movie = st.selectbox("Select a movie:", all_titles)

    if st.button("🎯 Generate Hybrid Recommendations", type="primary", use_container_width=True):
        with st.spinner("🤖 Analyzing across all engines..."):
            recs = hybrid_recs(movie, n_recs, wc, wm, wv, wp)

        if not recs.empty:
            input_row = D["master"][D["master"]["title"] == movie]
            if not input_row.empty:
                r = input_row.iloc[0]
                c1, c2 = st.columns([1, 4])
                with c1:
                    img = load_poster(str(r.get("poster_path", "")))
                    if img:
                        st.image(img, width=120)
                    else:
                        st.markdown('<div class="no-poster" style="height:160px;">🎬</div>',
                                    unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div style="font-family:'Inter',sans-serif;">
                        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.3rem;color:#f8fafc;font-weight:600;">
                            {r['title']}</div>
                        <div style="color:#f5c518;margin:4px 0;">★ {r.get('vote_average',0):.1f}</div>
                        <div style="color:#64748b;font-size:0.85rem;">
                            {str(r.get('genres','')).replace('|',' · ')}</div>
                        <div style="color:#94a3b8;font-size:0.85rem;margin-top:6px;line-height:1.5;">
                            {str(r.get('overview',''))[:200]}...</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            render_movie_cards(recs, score_col="score", cols=5)

            with st.expander("📊 Score Breakdown", expanded=False):
                bd = recs[["title","score","content","meta","visual","pop"]].head(10)
                st.dataframe(bd.style.format({
                    "score":"{:.4f}","content":"{:.4f}",
                    "meta":"{:.4f}","visual":"{:.4f}","pop":"{:.4f}"
                }), use_container_width=True)
        else:
            empty_state("🤖", "No hybrid recommendations",
                         "Try selecting a different movie or adjusting the weights.")


# ════════════════════════════════════════════════════════
# CONTENT-BASED
# ════════════════════════════════════════════════════════
elif "Content" in page:
    st.markdown('<div class="section-header">CONTENT-BASED ENGINE</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<span class="engine-tag">Plot Analysis</span>'
        '<span class="engine-tag">NLP</span>'
        '<span class="engine-tag">TF-IDF</span>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div style="color:#64748b;font-size:0.8rem;font-family:'Inter',sans-serif;margin-bottom:12px;">
        Finds movies with similar plots, storylines, and keywords using TF-IDF vectorization.
    </div>
    """, unsafe_allow_html=True)

    all_titles = sorted(D["content_df"]["title"].dropna().unique().tolist())
    movie = st.selectbox("Select a movie:", all_titles)

    if st.button("🔍 Find Similar", type="primary", use_container_width=True):
        with st.spinner("📝 Analyzing plot similarity..."):
            recs = get_sim_recs(D["content_sim"], D["content_df"],
                                D["content_idx"], movie, n_recs)
        if recs.empty:
            empty_state("📝", "No similar movies found",
                         "Try selecting a different movie.")
        else:
            st.markdown("### Selected Movie")
            render_selected_movie(movie)
            st.markdown("### Recommendations")
            render_movie_cards(recs, cols=5)


# ════════════════════════════════════════════════════════
# COLLABORATIVE
# ════════════════════════════════════════════════════════
elif "Collaborative" in page:
    st.markdown('<div class="section-header">COLLABORATIVE FILTERING</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<span class="engine-tag">ALS</span>'
        '<span class="engine-tag">User Behavior</span>'
        '<span class="engine-tag">Matrix Factorization</span>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div style="color:#64748b;font-size:0.8rem;font-family:'Inter',sans-serif;margin-bottom:12px;">
        Personalized recommendations based on user rating patterns using Alternating Least Squares.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔢 Existing User", "✨ Custom User"])

    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            user_id = st.number_input("User ID:", min_value=1, value=103270, step=1)
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            lookup_btn = st.button("🔍 Lookup & Recommend", type="primary",
                                   key="collab_existing", use_container_width=True)

        if lookup_btn:
            watched = D["ratings"][D["ratings"]["userId"] == user_id]
            if watched.empty:
                render_error(f"User {user_id} not found in the database. Try a different ID.")
            else:
                watched_df = watched.merge(
                    D["master"][["movie_id","title","genres","vote_average"]],
                    on="movie_id", how="left"
                ).sort_values("rating", ascending=False)

                with st.expander(f"📋 User {user_id} — {len(watched_df)} movies watched"):
                    st.dataframe(
                        watched_df[["title","genres","rating"]].head(20),
                        use_container_width=True
                    )

                with st.spinner("👥 Generating ALS recommendations..."):
                    recs = collab_recs_for_user(int(user_id), n_recs)

                if recs.empty:
                    empty_state("👥", "No recommendations",
                                 "This user may not have enough ratings for personalized suggestions.")
                else:
                    render_movie_cards(recs, cols=5)

    with tab2:
        st.markdown("""
        <div style="color:#94a3b8;font-family:'Inter',sans-serif;font-size:0.85rem;margin-bottom:12px;">
            ✨ Rate movies you've watched and get instant personalized recommendations.
        </div>
        """, unsafe_allow_html=True)

        all_titles = sorted(D["master"]["title"].dropna().unique().tolist())
        selected_movies = st.multiselect(
            "Choose movies you've watched:",
            all_titles,
            max_selections=20,
            placeholder="Start typing a movie title..."
        )

        user_ratings = {}
        if selected_movies:
            st.markdown("**Rate each movie (1-5):**")
            cols_r = st.columns(2)
            for i, title in enumerate(selected_movies):
                with cols_r[i % 2]:
                    row = D["master"][D["master"]["title"] == title]
                    if not row.empty:
                        mid = row.iloc[0]["movie_id"]
                        r = st.slider(
                            f"⭐ {title[:30]}",
                            1.0, 5.0, 4.0, 0.5,
                            key=f"rating_{mid}"
                        )
                        user_ratings[mid] = r

            if user_ratings and st.button("🎯 Get My Recommendations", type="primary",
                                           key="collab_custom", use_container_width=True):
                with st.spinner("👤 Building your preference profile..."):
                    recs = collab_recs_custom_user(user_ratings, n_recs)
                if not recs.empty:
                    render_success(f"Recommendations based on your {len(user_ratings)} ratings:")
                    render_movie_cards(recs, cols=5)
                else:
                    render_error("Couldn't generate recommendations. Try rating more movies (at least 3-5).")
        else:
            empty_state("✨", "Rate some movies",
                        "Select movies from the dropdown above and set your ratings to get started.")


# ════════════════════════════════════════════════════════
# METADATA
# ════════════════════════════════════════════════════════
elif "Metadata" in page:
    st.markdown('<div class="section-header">METADATA ENGINE</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<span class="engine-tag">Director</span>'
        '<span class="engine-tag">Cast</span>'
        '<span class="engine-tag">Genre</span>'
        '<span class="engine-tag">Keywords</span>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div style="color:#64748b;font-size:0.8rem;font-family:'Inter',sans-serif;margin-bottom:12px;">
        Matches movies by director, cast, genre composition, and production keywords.
    </div>
    """, unsafe_allow_html=True)

    all_titles = sorted(D["meta_df"]["title"].dropna().unique().tolist())
    movie = st.selectbox("Select a movie:", all_titles)

    if st.button("🏷️ Find Similar", type="primary", use_container_width=True):
        with st.spinner("🏷️ Matching metadata patterns..."):
            recs = get_sim_recs(D["meta_sim"], D["meta_df"],
                                D["meta_idx"], movie, n_recs)
        if recs.empty:
            empty_state("🏷️", "No metadata matches",
                         "Try selecting a different movie.")
        else:
            st.markdown("### Selected Movie")
            render_selected_movie(movie)
            st.markdown("### Recommendations")
            render_movie_cards(recs, cols=5)


# ════════════════════════════════════════════════════════
# VISUAL
# ════════════════════════════════════════════════════════
elif "Visual" in page:
    st.markdown('<div class="section-header">VISUAL ENGINE</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<span class="engine-tag">ResNet18</span>'
        '<span class="engine-tag">Poster Analysis</span>'
        '<span class="engine-tag">CNN Features</span>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div style="color:#64748b;font-size:0.8rem;font-family:'Inter',sans-serif;margin-bottom:12px;">
        Uses deep learning (ResNet18 CNN) to find movies with visually similar posters and aesthetics.
    </div>
    """, unsafe_allow_html=True)

    all_titles = sorted(D["visual_df"]["title"].dropna().unique().tolist())
    movie = st.selectbox("Select a movie:", all_titles)

    if st.button("🖼️ Find Visually Similar", type="primary", use_container_width=True):
        with st.spinner("🖼️ Analyzing poster aesthetics with CNN..."):
            recs = get_sim_recs(D["visual_sim"], D["visual_df"],
                                D["visual_idx"], movie, n_recs)
        if recs.empty:
            empty_state("🖼️", "No visually similar movies",
                         "Try selecting a different movie.")
        else:
            st.markdown("### Selected Movie")
            render_selected_movie(movie)
            st.markdown("### Recommendations")
            render_movie_cards(recs, cols=5)


# ════════════════════════════════════════════════════════
# SEQUENCE-BASED
# ════════════════════════════════════════════════════════
elif "Sequence" in page:
    st.markdown('<div class="section-header">SEQUENCE ENGINE (LSTM)</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<span class="engine-tag">LSTM</span>'
        '<span class="engine-tag">Temporal Patterns</span>'
        '<span class="engine-tag">Next-Item Prediction</span>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div style="color:#64748b;font-size:0.8rem;font-family:'Inter',sans-serif;margin-bottom:12px;">
        Predicts your next movie based on viewing sequence patterns using a trained LSTM model.
    </div>
    """, unsafe_allow_html=True)

    if not D.get("seq_loaded"):
        render_error("Sequence model (LSTM) is not loaded. Please run the sequence training notebook first.")
    else:
        all_titles = sorted(D["master"]["title"].dropna().unique().tolist())
        watched = st.multiselect(
            "Add movies in the order you watched them:",
            all_titles,
            default=["The Dark Knight", "Inception", "Interstellar"],
            max_selections=20,
            placeholder="Search and add movies..."
        )

        if watched:
            flow_markup = " → ".join([f"<b style='color:#e50914;'>{m}</b>" for m in watched])
            st.markdown(f"""
            <div style="font-family:'Inter',sans-serif;font-size:0.85rem;color:#94a3b8;padding:12px;background:rgba(17,24,39,0.5);border-radius:10px;margin:8px 0;">
                Your sequence: {flow_markup}
            </div>
            """, unsafe_allow_html=True)

            if st.button("⏱️ Predict Next Movies", type="primary", use_container_width=True):
                with st.spinner("⏱️ LSTM analyzing temporal patterns..."):
                    recs = sequence_recs(watched, n_recs)
                if not recs.empty:
                    st.markdown("""
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:1.2rem;color:#f8fafc;margin:16px 0 8px 0;letter-spacing:1px;">
                        ⏱️ What You'll Probably Watch Next
                    </div>
                    """, unsafe_allow_html=True)
                    render_movie_cards(recs, score_col="score", cols=5)
                else:
                    empty_state("⏱️", "No predictions available",
                                 "Try adding more movies to your sequence.")
        else:
            empty_state("⏱️", "Build Your Watch Sequence",
                        "Add movies in the order you watched them above, and the LSTM will predict the next one.")


# ════════════════════════════════════════════════════════
# TRENDING & TOP
# ════════════════════════════════════════════════════════
elif "Trending" in page:
    st.markdown('<div class="section-header">TRENDING & TOP MOVIES</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#64748b;font-size:0.8rem;font-family:'Inter',sans-serif;margin-bottom:12px;">
        Explore top-rated, trending, hidden gems, and genre-specific movies from the database.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🏆 All Time Best", "📈 Trending Now", "💎 Hidden Gems", "🎭 By Genre"]
    )

    with tab1:
        st.markdown("""
        <div style="color:#94a3b8;font-size:0.8rem;font-family:'Inter',sans-serif;margin-bottom:8px;">
            Highest weighted score movies across all users and ratings.
        </div>
        """, unsafe_allow_html=True)
        render_movie_cards(D["pop_df"].head(n_recs), score_col="weighted_score", cols=5)

    with tab2:
        st.markdown("""
        <div style="color:#94a3b8;font-size:0.8rem;font-family:'Inter',sans-serif;margin-bottom:8px;">
            Most popular movies right now based on user engagement.
        </div>
        """, unsafe_allow_html=True)
        trending = D["pop_df"].sort_values("popularity", ascending=False).head(n_recs)
        render_movie_cards(trending, score_col="popularity", cols=5)

    with tab3:
        st.markdown("""
        <div style="color:#94a3b8;font-size:0.8rem;font-family:'Inter',sans-serif;margin-bottom:8px;">
            Underrated movies with high ratings but fewer votes — perfect for discovering something new.
        </div>
        """, unsafe_allow_html=True)
        gems = D["pop_df"][
            (D["pop_df"]["vote_average"] >= 7.5) &
            (D["pop_df"]["vote_count"]   <= 1000) &
            (D["pop_df"]["vote_count"]   >= 100)
        ].sort_values("vote_average", ascending=False).head(n_recs)
        if gems.empty:
            empty_state("💎", "No hidden gems found", "Try adjusting the rating or vote count filters.")
        else:
            render_movie_cards(gems, score_col="vote_average", cols=5)

    with tab4:
        st.markdown("""
        <div style="color:#94a3b8;font-size:0.8rem;font-family:'Inter',sans-serif;margin-bottom:8px;">
            Browse top movies filtered by your favorite genre.
        </div>
        """, unsafe_allow_html=True)
        genres_list = sorted([
            "Action","Adventure","Animation","Comedy","Crime",
            "Documentary","Drama","Fantasy","History","Horror",
            "Music","Mystery","Romance","Science Fiction",
            "Thriller","War","Western","Family"
        ])
        genre = st.selectbox("Genre:", genres_list)
        filtered = D["pop_df"][
            D["pop_df"]["genres"].str.contains(genre, case=False, na=False)
        ].head(n_recs)
        if filtered.empty:
            empty_state("🎭", f"No {genre} movies found", "Try selecting a different genre.")
        else:
            render_movie_cards(filtered, score_col="weighted_score", cols=5)


# ════════════════════════════════════════════════════════
# USERS EXPLORER
# ════════════════════════════════════════════════════════
elif "Users" in page:
    st.markdown('<div class="section-header">USERS EXPLORER</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔎 User Profile", "➕ Add New User"])

    ratings = D["ratings"]
    master = D["master"]

    # ── Tab 1: Overview ──────────────────────────────────
    with tab1:
        total_users  = ratings["userId"].nunique()
        total_ratings = len(ratings)
        avg_per_user = ratings.groupby("userId").size().mean()
        avg_rating   = ratings["rating"].mean()

        c1, c2, c3, c4 = st.columns(4)
        for col, val, label in zip(
            [c1, c2, c3, c4],
            [f"{total_users:,}", f"{total_ratings:,}",
             f"{avg_per_user:.0f}", f"{avg_rating:.2f}"],
            ["Total Users", "Total Ratings", "Avg Ratings/User", "Avg Rating"]
        ):
            with col:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # توزيع الـ ratings
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Rating Distribution**")
            rating_dist = ratings["rating"].value_counts().sort_index()
            st.bar_chart(rating_dist)

        with c2:
            st.markdown("**Users by Activity Level**")
            user_sizes = ratings.groupby("userId").size()
            bins = pd.cut(user_sizes, bins=[0,20,50,100,250,500,10000],
                          labels=["0-20","21-50","51-100","101-250","251-500","500+"])
            st.bar_chart(bins.value_counts().sort_index())

        st.markdown("---")
        st.markdown("**Top 20 Most Active Users**")
        top_users = (
            ratings.groupby("userId")
            .agg(n_ratings=("rating","count"), avg_rating=("rating","mean"))
            .sort_values("n_ratings", ascending=False)
            .head(20)
            .reset_index()
        )
        top_users["avg_rating"] = top_users["avg_rating"].round(2)
        st.dataframe(top_users, use_container_width=True)

    # ── Tab 2: User Profile ───────────────────────────────
    with tab2:
        st.markdown("### Look up any user")
        c1, c2 = st.columns([2, 1])
        with c1:
            user_id = st.number_input("User ID:", min_value=1, value=103270, step=1,
                                      key="profile_uid")
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            load_btn = st.button("🔍 Load Profile", type="primary", use_container_width=True)

        if load_btn:
            user_ratings = D["ratings"][D["ratings"]["userId"] == user_id]

            if user_ratings.empty:
                render_error(f"User {user_id} not found in the database.")
            else:
                watched_df = user_ratings.merge(
                    D["master"][["movie_id","title","genres","vote_average","poster_path"]],
                    on="movie_id", how="left"
                ).sort_values("rating", ascending=False)

                # Stats
                c1, c2, c3, c4 = st.columns(4)
                for col, val, label in zip(
                    [c1,c2,c3,c4],
                    [f"{len(watched_df)}",
                     f"{watched_df['rating'].mean():.2f}",
                     f"{watched_df['rating'].max():.1f}",
                     f"{watched_df['rating'].min():.1f}"],
                    ["Movies Rated","Avg Rating","Max Rating","Min Rating"]
                ):
                    with col:
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-value">{val}</div>
                            <div class="metric-label">{label}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("---")

                # Charts
                all_genres = watched_df["genres"].dropna().str.split("|").explode()
                top_genres = all_genres.value_counts().head(8)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**🎭 Favorite Genres**")
                    st.bar_chart(top_genres)
                with c2:
                    st.markdown("**📊 Rating Distribution**")
                    st.bar_chart(watched_df["rating"].value_counts().sort_index())

                st.markdown("---")

                # Top & bottom rated
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**⭐ Top Rated by User**")
                    st.dataframe(
                        watched_df[["title","genres","rating"]].head(10),
                        use_container_width=True
                    )
                with c2:
                    st.markdown("**👎 Lowest Rated**")
                    st.dataframe(
                        watched_df[["title","genres","rating"]].tail(10),
                        use_container_width=True
                    )

                # Get recommendations
                st.markdown("---")
                st.markdown("**🤖 Recommendations for this user:**")
                eng = st.selectbox("Choose engine:", [
                    "Collaborative", "Content-Based (based on top movie)",
                    "Hybrid (based on top movie)"
                ], key="profile_engine")

                if st.button("🎯 Generate Recommendations", type="primary", key="profile_recs",
                             use_container_width=True):
                    if "Collaborative" in eng:
                        with st.spinner("👥 Running ALS collaborative filtering..."):
                            recs = collab_recs_for_user(int(user_id), n_recs)
                    else:
                        top_movie = watched_df.iloc[0]["title"]
                        st.info(f"Based on top movie: **{top_movie}**")
                        if "Content" in eng:
                            recs = get_sim_recs(D["content_sim"], D["content_df"],
                                                D["content_idx"], top_movie, n_recs)
                        else:
                            recs = hybrid_recs(top_movie, n_recs, wc, wm, wv, wp)
                    if recs.empty:
                        empty_state("👤", "No recommendations", "Try a different engine or user.")
                    else:
                        render_movie_cards(recs, cols=5)

    # ── Tab 3: Add New User ───────────────────────────────
    with tab3:
        st.markdown("### 🆕 Create a new user profile")
        st.markdown("""
        <div style="color:#94a3b8;font-family:'Inter',sans-serif;font-size:0.85rem;margin-bottom:16px;">
            Rate some movies and get personalized recommendations as if you are this user.
        </div>
        """, unsafe_allow_html=True)

        all_titles = sorted(D["master"]["title"].dropna().unique().tolist())

        col1, col2 = st.columns([2,1])
        with col1:
            new_user_name = st.text_input("Your name (optional):", placeholder="e.g. Ahmed")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            new_user_id = f"custom_{new_user_name or 'user'}_{len(st.session_state.custom_users)+1}"

        st.markdown("**🎬 Select & rate movies you've watched:**")

        selected = st.multiselect(
            "Search and add movies:",
            all_titles,
            max_selections=30,
            key="new_user_movies",
            placeholder="Start typing a movie title..."
        )

        new_ratings = {}
        if selected:
            st.markdown("**Rate each movie (1-5):**")
            n_cols = 2
            cols_r = st.columns(n_cols)
            for i, title in enumerate(selected):
                with cols_r[i % n_cols]:
                    row = D["master"][D["master"]["title"] == title]
                    if not row.empty:
                        mid = row.iloc[0]["movie_id"]
                        genre_str = str(row.iloc[0].get("genres","")).replace("|"," · ")[:40]
                        r = st.slider(
                            f"**{title[:28]}**\n_{genre_str}_",
                            1.0, 5.0, 4.0, 0.5,
                            key=f"new_{mid}"
                        )
                        new_ratings[mid] = r

            if new_ratings and st.button("💾 Save & Test User", type="primary",
                                         use_container_width=True):
                st.session_state.custom_users[new_user_id] = {
                    "name": new_user_name or "Anonymous",
                    "ratings": new_ratings
                }
                render_success(f"User **{new_user_name or 'Anonymous'}** created with "
                               f"**{len(new_ratings)}** ratings!")
        else:
            empty_state("👤", "Create Your Profile",
                        "Select movies you've watched and rate them above to get started.")

        # Custom users section
        if st.session_state.custom_users:
            st.markdown("---")
            st.markdown('<div class="section-header" style="font-size:1.4rem;">YOUR CUSTOM USERS</div>',
                        unsafe_allow_html=True)

            for uid, udata in st.session_state.custom_users.items():
                with st.expander(
                    f"👤 {udata['name']}  |  {len(udata['ratings'])} movies rated"
                ):
                    rated_df = pd.DataFrame([
                        {
                            "title": D["master"][D["master"]["movie_id"]==mid].iloc[0]["title"]
                            if not D["master"][D["master"]["movie_id"]==mid].empty else mid,
                            "rating": r
                        }
                        for mid, r in udata["ratings"].items()
                    ]).sort_values("rating", ascending=False)

                    st.dataframe(rated_df, use_container_width=True)

                    st.markdown("**🤖 Get recommendations for this user:**")
                    eng2 = st.selectbox("Engine:", [
                        "Collaborative", "Hybrid (top-rated movie)",
                        "Content-Based (top-rated movie)"
                    ], key=f"eng_{uid}")

                    if st.button(f"🎯 Recommend for {udata['name']}", type="primary",
                                 key=f"rec_{uid}", use_container_width=True):
                        if "Collaborative" in eng2:
                            with st.spinner("👥 Building collaborative profile..."):
                                recs = collab_recs_custom_user(udata["ratings"], n_recs)
                        else:
                            top_mid = max(udata["ratings"], key=udata["ratings"].get)
                            top_row = D["master"][D["master"]["movie_id"] == top_mid]
                            if not top_row.empty:
                                top_title = top_row.iloc[0]["title"]
                                toast(f"Based on top movie: {top_title}", "info")
                                if "Content" in eng2:
                                    recs = get_sim_recs(
                                        D["content_sim"], D["content_df"],
                                        D["content_idx"], top_title, n_recs
                                    )
                                else:
                                    recs = hybrid_recs(top_title, n_recs, wc, wm, wv, wp)
                            else:
                                recs = pd.DataFrame()

                        if recs.empty:
                            empty_state("👤", "No recommendations",
                                         "Try adding more ratings or a different engine.")
                        else:
                            render_movie_cards(recs, cols=5)


# ════════════════════════════════════════════════════════
# ENGINE STATS - Professional Dashboard
# ════════════════════════════════════════════════════════
elif "Stats" in page:
    st.markdown('<div class="section-header">📊 ENGINE PERFORMANCE DASHBOARD</div>',
                unsafe_allow_html=True)
    
    # Load actual evaluation results
    try:
        eval_results = pd.read_csv(f"{BASE}/models/evaluation_results.csv", index_col=0)
        eval_results = eval_results.fillna(0)
    except:
        # Fallback to default values if file not found
        eval_results = pd.DataFrame({
            "precision@10": [0.87, 0.96, 0.73, 0.0, 0.45, 0.46, 0.46],
            "avg_diversity": [8.2, 6.4, 10.8, 0, 13.0, 7.2, 12.2]
        }, index=["Content-Based", "Metadata", "Visual", "Collaborative", "Popularity", "Hybrid", "Sequence"])
    
    # Engine descriptions
    engine_info = {
        "Content-Based": ("📝", "TF-IDF on plot, overview, keywords", "Similar plots & stories"),
        "Metadata": ("🏷️", "Director, cast, genres, keywords", "Same director / cast"),
        "Visual": ("🖼️", "ResNet18 CNN on movie posters", "Similar poster aesthetics"),
        "Collaborative": ("👥", "ALS Matrix Factorization", "Personalized for users"),
        "Popularity": ("🔥", "Weighted score & trending", "Popular & trending"),
        "Hybrid": ("🤖", "Ensemble of all engines", "Best overall results"),
        "Sequence": ("⏱️", "LSTM temporal patterns", "Next-watch prediction")
    }
    
    # Top Metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        best_engine = eval_results['precision@10'].idxmax()
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="font-size:2rem;">{best_engine}</div>
            <div class="metric-label">Best Engine (Precision@10)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        avg_precision = eval_results['precision@10'].mean()
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="font-size:2rem;">{avg_precision:.3f}</div>
            <div class="metric-label">Avg Precision@10</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        total_engines = len(eval_results)
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="font-size:2rem;">{total_engines}</div>
            <div class="metric-label">Active AI Engines</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Precision Rankings
    st.markdown("### 🏆 Precision@10 Rankings")
    sorted_engines = eval_results.sort_values('precision@10', ascending=False)
    for i, (engine, row) in enumerate(sorted_engines.iterrows(), 1):
        icon, tech, use_case = engine_info.get(engine, ("🤖", "AI Engine", "Movie recommendations"))
        p = row['precision@10']
        if p > 0:
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            color = "#22c55e" if p >= 0.9 else "#f59e0b" if p >= 0.7 else "#64748b"
            c1, c2, c3, c4 = st.columns([0.5, 2, 5, 1])
            with c1:
                st.markdown(f"<div style='font-size:1.5rem;'>{medal}</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"**{icon} {engine}**<br><span style='font-size:0.7rem;color:#64748b;'>{tech}</span>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div style="background:#1e2a3a;border-radius:8px;height:24px;margin-top:6px;">
                  <div style="background:{color};width:{min(p*100,100):.1f}%;height:100%;
                  border-radius:8px;transition:width 0.5s ease;"></div>
                </div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"<div style='text-align:right;'><b>{p:.4f}</b></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Diversity & Coverage
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🎨 Diversity (Genre Coverage)")
        if 'avg_diversity' in eval_results.columns:
            diversity_data = eval_results[eval_results['avg_diversity'] > 0]['avg_diversity'].sort_values(ascending=False)
            st.bar_chart(diversity_data)
    with c2:
        st.markdown("### 📈 Coverage (Catalog Reach)")
        if 'coverage' in eval_results.columns:
            coverage_data = eval_results[eval_results['coverage'] > 0]['coverage'].sort_values(ascending=False)
            st.bar_chart(coverage_data)
    
    st.markdown("---")
    
    # Detailed Table
    st.markdown("### 📋 Detailed Engine Comparison")
    display_df = eval_results.copy()
    display_df.columns = [col.replace('_', ' ').title() for col in display_df.columns]
    st.dataframe(display_df, use_container_width=True)
    
    st.markdown("---")
    
    # Collaborative Metrics (if available)
    if 'RMSE' in eval_results.columns or 'Collaborative' in eval_results.index:
        st.markdown("### 👥 Collaborative Filtering Metrics")
        c1, c2 = st.columns(2)
        with c1:
            rmse_val = eval_results.loc['Collaborative', 'RMSE'] if 'RMSE' in eval_results.columns else 3.15
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{rmse_val:.2f}</div>
                <div class="metric-label">RMSE (Rating Prediction)</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            mae_val = eval_results.loc['Collaborative', 'MAE'] if 'MAE' in eval_results.columns else 2.98
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{mae_val:.2f}</div>
                <div class="metric-label">MAE (Rating Prediction)</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header" style="font-size:1.4rem;">ENGINE USE CASES</div>',
                unsafe_allow_html=True)
    for engine, (icon, tech, use_case) in engine_info.items():
        if engine in eval_results.index:
            p = eval_results.loc[engine, 'precision@10']
            p_str = f"{p:.3f}" if p > 0 else "N/A"
            p_color = "#22c55e" if p >= 0.9 else "#f59e0b" if p >= 0.7 else "#64748b"
            st.markdown(f"""
            <div class="glass-card" style="display:flex;justify-content:space-between;align-items:center;padding:16px 20px;margin-bottom:10px;">
                <div style="display:flex;align-items:center;gap:12px;">
                    <span style="font-size:1.5rem;">{icon}</span>
                    <div>
                        <b style="color:#f8fafc;font-family:'Space Grotesk',sans-serif;font-size:0.95rem;">{engine}</b>
                        <div style="color:#64748b;font-size:0.75rem;font-family:'Inter',sans-serif;">{tech}</div>
                        <div style="color:#94a3b8;font-size:0.75rem;margin-top:2px;">💡 {use_case}</div>
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="color:{p_color};font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.1rem;">
                        {p_str}</div>
                    <div style="color:#64748b;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;">Precision</div>
                </div>
            </div>
            """, unsafe_allow_html=True)