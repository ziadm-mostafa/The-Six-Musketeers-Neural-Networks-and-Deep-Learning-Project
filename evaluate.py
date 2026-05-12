"""
Movie Recommender System - Evaluation Script
============================================
يقيّم كل الـ engines ويقارن بينهم - نسخة محسّنة بدقة عالية
"""

import os
import ast
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from scipy.sparse import load_npz
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

os.chdir(r"C:\Users\Saeed\Desktop\neural_network_project")

# ════════════════════════════════════════════════════════
# تحميل كل البيانات والموديلات
# ════════════════════════════════════════════════════════
print("Loading models and data...")

master     = pd.read_csv("data/processed/movies_master.csv")
content_df = pd.read_csv("models/content_based/content_df.csv")
meta_df    = pd.read_csv("models/metadata/metadata_df.csv")
visual_df  = pd.read_csv("models/visual/visual_df.csv")
pop_df     = pd.read_csv("models/popularity/popularity_df.csv")
ratings    = pd.read_csv("data/processed/collaborative_ratings.csv")

content_sim = np.load("models/content_based/cosine_sim.npy")
meta_sim    = np.load("models/metadata/cosine_sim_meta.npy")
visual_sim  = np.load("models/visual/visual_sim.npy")

collab_model  = joblib.load("models/collaborative/als_model.pkl")
user2idx      = joblib.load("models/collaborative/user2idx.pkl")
movie2idx     = joblib.load("models/collaborative/movie2idx.pkl")
idx2movie     = joblib.load("models/collaborative/idx2movie.pkl")
sparse_matrix = load_npz("models/collaborative/sparse_matrix.npz")

content_idx = pd.Series(content_df.index,
                         index=content_df["title"].str.lower()).drop_duplicates()
meta_idx    = pd.Series(meta_df.index,
                         index=meta_df["title"].str.lower()).drop_duplicates()
visual_idx  = pd.Series(visual_df.index,
                         index=visual_df["title"].str.lower()).drop_duplicates()

print("All loaded!\n")


# ════════════════════════════════════════════════════════
# Load Sequence (LSTM) Model
# ════════════════════════════════════════════════════════
print("Loading Sequence (LSTM) Model...")

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config and mappings
    lstm_config = joblib.load("models/sequence/lstm_config.pkl")
    seq_movie2id = joblib.load("models/sequence/movie2id.pkl")
    seq_id2movie = joblib.load("models/sequence/id2movie.pkl")
    
    # Define LSTM Model class
    class MovieLSTM(nn.Module):
        def __init__(self, n_movies, embed_dim=64, hidden_dim=128, n_layers=2, dropout=0.3):
            super().__init__()
            self.embedding = nn.Embedding(num_embeddings=n_movies, embedding_dim=embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=n_layers,
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
            last = out[:, -1, :]
            return self.fc(last)
    
    # Initialize and load weights
    seq_model = MovieLSTM(
        n_movies=lstm_config["n_movies"],
        embed_dim=lstm_config["embed_dim"],
        hidden_dim=lstm_config["hidden_dim"],
        n_layers=lstm_config["n_layers"]
    ).to(device)
    seq_model.load_state_dict(torch.load("models/sequence/lstm_model.pt", map_location=device))
    seq_model.eval()
    
    MAX_SEQ_LEN = lstm_config["max_len"]
    print(f"   LSTM loaded: {lstm_config['n_movies']:,} movies, {lstm_config['n_layers']} layers")
    print(f"   Device: {device}")
    seq_model_available = True
    
except Exception as e:
    print(f"   Warning: Could not load Sequence model: {e}")
    seq_model_available = False
    seq_movie2id = {}
    seq_id2movie = {}
    seq_model = None
    MAX_SEQ_LEN = 20

print()


# ════════════════════════════════════════════════════════
# Enhanced Helper Functions
# ════════════════════════════════════════════════════════
def get_genres(movie_id):
    row = master[master["movie_id"] == movie_id]
    if row.empty:
        return set()
    genres = str(row.iloc[0]["genres"])
    return set(genres.split("|")) if genres != "nan" else set()

def get_year(movie_id):
    """جيب سنة الفيلم"""
    row = master[master["movie_id"] == movie_id]
    if row.empty:
        return None
    try:
        return int(row.iloc[0]["release_year"])
    except:
        return None

# ─── Genre-Based Metrics ─────────────────────────────────
def genre_precision_at_k(input_id, rec_ids, k=10):
    """كم % من الترشيحات بتشارك نفس الـ genre"""
    input_genres = get_genres(input_id)
    if not input_genres:
        return 0.0
    hits = sum(
        1 for mid in rec_ids[:k]
        if get_genres(mid) & input_genres
    )
    return hits / min(k, len(rec_ids)) if rec_ids else 0.0

def genre_recall_at_k(input_id, rec_ids, all_similar_ids, k=10):
    """كم % من الأفلام المشابهة تم استرجاعها"""
    input_genres = get_genres(input_id)
    if not input_genres or not all_similar_ids:
        return 0.0
    relevant = [mid for mid in all_similar_ids if get_genres(mid) & input_genres]
    if not relevant:
        return 0.0
    retrieved_relevant = sum(1 for mid in rec_ids[:k] if mid in relevant)
    return retrieved_relevant / len(relevant)

# ─── Content-Based Diversity ─────────────────────────────
def genre_diversity(rec_ids):
    """كم الترشيحات متنوعة في الـ genres"""
    all_genres = set()
    for mid in rec_ids:
        all_genres |= get_genres(mid)
    return len(all_genres)

def year_diversity(rec_ids):
    """التنوع في سنوات الإنتاج"""
    years = [get_year(mid) for mid in rec_ids if get_year(mid)]
    if len(years) < 2:
        return 0.0
    return np.std(years)

def coverage(all_rec_ids, total_movies):
    """كم فيلم مختلف بيظهر في الترشيحات"""
    return len(set(all_rec_ids)) / total_movies if total_movies > 0 else 0.0

# ─── Advanced Ranking Metrics ────────────────────────────
def ndcg_at_k(relevances, k=10):
    """Normalized Discounted Cumulative Gain"""
    relevances = np.array(relevances[:k])
    if relevances.size == 0:
        return 0.0
    
    # DCG
    dcg = np.sum((2**relevances - 1) / np.log2(np.arange(2, len(relevances) + 2)))
    
    # Ideal DCG
    ideal_relevances = np.sort(relevances)[::-1]
    idcg = np.sum((2**ideal_relevances - 1) / np.log2(np.arange(2, len(ideal_relevances) + 2)))
    
    return dcg / idcg if idcg > 0 else 0.0

def calculate_ndcg_for_movie(input_id, rec_ids, k=10):
    """حساب NDCG لفيلم معين بناءً على تشابه الـ genres"""
    input_genres = get_genres(input_id)
    if not input_genres:
        return 0.0
    
    relevances = []
    for mid in rec_ids[:k]:
        common_genres = len(get_genres(mid) & input_genres)
        # Normalized relevance between 0 and 1
        rel = min(common_genres / max(len(input_genres), 1), 1.0)
        relevances.append(rel)
    
    return ndcg_at_k(relevances, k)

def mrr_at_k(rec_ids, relevant_ids, k=10):
    """Mean Reciprocal Rank - أول ظهور لـ relevant item فين"""
    for i, mid in enumerate(rec_ids[:k]):
        if mid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

def hit_rate_at_k(rec_ids, relevant_ids, k=10):
    """هل فيه relevant item في الـ top-k"""
    return 1.0 if any(mid in relevant_ids for mid in rec_ids[:k]) else 0.0

def average_precision_at_k(rec_ids, relevant_ids, k=10):
    """Average Precision@K"""
    if not relevant_ids:
        return 0.0
    
    precisions = []
    hits = 0
    for i, mid in enumerate(rec_ids[:k]):
        if mid in relevant_ids:
            hits += 1
            precisions.append(hits / (i + 1))
    
    return np.mean(precisions) if precisions else 0.0

# ─── Statistical Significance ────────────────────────────
def paired_t_test(scores_a, scores_b):
    """اختبار إحصائي للفرق بين methodين"""
    from scipy import stats
    if len(scores_a) != len(scores_b) or len(scores_a) < 2:
        return None, None
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    return t_stat, p_value

def get_recs_by_sim(sim_matrix, df, idx_series, movie_id, k=10):
    """جيب ترشيحات من similarity matrix"""
    row = df[df["movie_id"] == movie_id]
    if row.empty:
        return []
    title = row.iloc[0]["title"].lower()
    if title not in idx_series:
        return []
    
    # تأكد إن الـ idx scalar مش Series
    idx = idx_series[title]
    if hasattr(idx, '__len__'):
        idx = int(idx.iloc[0])
    else:
        idx = int(idx)
    
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: float(x[1]), reverse=True)[1:k+1]
    return [df.iloc[i[0]]["movie_id"] for i in sim_scores]


def get_sequence_recs(movie_ids, k=10):
    """جيب ترشيحات من Sequence (LSTM) Model"""
    if not seq_model_available or not seq_model:
        return []
    
    # Convert movie_ids to encoded IDs
    encoded = [seq_movie2id[mid] for mid in movie_ids if mid in seq_movie2id]
    if not encoded:
        return []
    
    # Padding and prepare input
    padded = [0] * max(0, MAX_SEQ_LEN - len(encoded)) + encoded[-MAX_SEQ_LEN:]
    x = torch.tensor([padded], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits = seq_model(x)
        probs = torch.softmax(logits[0], dim=0)
    
    # Get top-k recommendations (excluding watched movies)
    probs_np = probs.cpu().numpy()
    probs_np[encoded] = 0  # Exclude watched
    probs_np[0] = 0        # Exclude padding
    
    top_encoded = np.argsort(probs_np)[::-1][:k*3]  # Get more for filtering
    
    results = []
    for enc_id in top_encoded:
        if enc_id not in seq_id2movie:
            continue
        mid = seq_id2movie[enc_id]
        results.append(mid)
        if len(results) == k:
            break
    
    return results


# ════════════════════════════════════════════════════════
# Enhanced Test Set Selection
# ════════════════════════════════════════════════════════
def create_balanced_test_set(master_df, n_movies=50, random_state=42):
    """إنشاء عينة متوازنة من الأفلام من مختلف الأنواع والعقود"""
    np.random.seed(random_state)
    
    movies = master_df.copy()
    
    # استخراج العقود
    movies["decade"] = (movies["release_year"] // 10) * 10
    
    # توزيع الأفلام
    selected = []
    
    # اختيار أفلام من كل عقد (80s, 90s, 2000s, 2010s, 2020s)
    for decade in [1980, 1990, 2000, 2010, 2020]:
        decade_movies = movies[movies["decade"] == decade]
        if len(decade_movies) > 0:
            n_select = min(n_movies // 5, len(decade_movies))
            selected.extend(decade_movies.sample(n_select)["movie_id"].tolist())
    
    # إضافة أفلام شائعة مضمونة
    popular = movies.nlargest(n_movies // 2, "popularity")["movie_id"].tolist()
    selected.extend(popular)
    
    # إضافة أفلام عشوائية للتنوع
    remaining = n_movies - len(selected)
    if remaining > 0:
        available = movies[~movies["movie_id"].isin(selected)]
        if len(available) > 0:
            selected.extend(available.sample(min(remaining, len(available)))["movie_id"].tolist())
    
    return list(dict.fromkeys(selected))  # Remove duplicates while preserving order

# إنشاء عينة التقييم
test_ids = create_balanced_test_set(master, n_movies=50)
test_df = master[master["movie_id"].isin(test_ids)][["movie_id", "title", "genres", "release_year"]].drop_duplicates()
print(f"Test movies: {len(test_ids)}")
decades = sorted(test_df['release_year'].dropna()//10*10)
unique_decades = sorted(set(decades))
print(f"Decades covered: {unique_decades}")

# عينة مستخدمين للـ Collaborative Filtering (من اللي موجودين في الـ model)
print("\nSelecting test users for Collaborative Filtering...")
# نستخدم فقط المستخدمين اللي indicesهم صالحة في الـ model
max_user_idx = collab_model.user_factors.shape[0]
valid_users = [uid for uid, idx in user2idx.items() if idx < max_user_idx]

if len(valid_users) > 100:
    np.random.seed(42)
    test_user_ids = np.random.choice(valid_users, size=100, replace=False).tolist()
else:
    test_user_ids = valid_users
print(f"Test users for Collaborative: {len(test_user_ids)}")


# ════════════════════════════════════════════════════════
# تقييم كل Engine بدقة عالية
# ════════════════════════════════════════════════════════
K = 10
results = {}

# تخزين جميع النتائج التفصيلية للمقارنات الإحصائية
detailed_results = defaultdict(lambda: defaultdict(list))

# ─── Engine 1: Content-Based ────────────────────────────
print("\n[1/7] Evaluating Content-Based with Advanced Metrics...")
precisions, recalls, ndcgs, mrrs, hit_rates, maps, diversities, year_divs, all_recs = [], [], [], [], [], [], [], [], []

for mid in test_ids:
    recs = get_recs_by_sim(content_sim, content_df, content_idx, mid, K)
    if recs and len(recs) >= K//2:
        # Basic metrics
        prec = genre_precision_at_k(mid, recs, K)
        precisions.append(prec)
        
        # Advanced metrics
        ndcg = calculate_ndcg_for_movie(mid, recs, K)
        ndcgs.append(ndcg)
        
        # MRR and Hit Rate - considering top 5 similar movies as relevant
        similar_movies = get_recs_by_sim(content_sim, content_df, content_idx, mid, 50)
        relevant = set(similar_movies[:20]) if similar_movies else set()
        
        mrr = mrr_at_k(recs, relevant, K)
        mrrs.append(mrr)
        
        hit = hit_rate_at_k(recs, relevant, K)
        hit_rates.append(hit)
        
        map_score = average_precision_at_k(recs, relevant, K)
        maps.append(map_score)
        
        # Diversity metrics
        diversities.append(genre_diversity(recs))
        year_divs.append(year_diversity(recs))
        
        all_recs.extend(recs)
        
        # Store detailed results
        detailed_results["Content-Based"]["precision"].append(prec)
        detailed_results["Content-Based"]["ndcg"].append(ndcg)
        detailed_results["Content-Based"]["mrr"].append(mrr)
        detailed_results["Content-Based"]["hit_rate"].append(hit)

results["Content-Based"] = {
    "precision@10"  : round(np.mean(precisions), 4) if precisions else 0.0,
    "ndcg@10"       : round(np.mean(ndcgs), 4) if ndcgs else 0.0,
    "mrr@10"        : round(np.mean(mrrs), 4) if mrrs else 0.0,
    "hit_rate@10"   : round(np.mean(hit_rates), 4) if hit_rates else 0.0,
    "map@10"        : round(np.mean(maps), 4) if maps else 0.0,
    "avg_diversity" : round(np.mean(diversities), 2) if diversities else 0.0,
    "year_diversity": round(np.mean(year_divs), 2) if year_divs else 0.0,
    "coverage"      : round(coverage(all_recs, len(master)), 4),
    "tested_movies" : len(precisions),
    "std_precision" : round(np.std(precisions), 4) if precisions else 0.0,
    "std_ndcg"      : round(np.std(ndcgs), 4) if ndcgs else 0.0
}
print(f"   Precision@10 : {results['Content-Based']['precision@10']:.4f} (±{results['Content-Based']['std_precision']:.4f})")
print(f"   NDCG@10      : {results['Content-Based']['ndcg@10']:.4f} (±{results['Content-Based']['std_ndcg']:.4f})")
print(f"   MRR@10       : {results['Content-Based']['mrr@10']:.4f}")
print(f"   Hit Rate@10  : {results['Content-Based']['hit_rate@10']:.4f}")
print(f"   MAP@10       : {results['Content-Based']['map@10']:.4f}")


# ─── Engine 2: Metadata ─────────────────────────────────
print("\n[2/7] Evaluating Metadata with Advanced Metrics...")
precisions, ndcgs, mrrs, hit_rates, maps, diversities, year_divs, all_recs = [], [], [], [], [], [], [], []

for mid in test_ids:
    recs = get_recs_by_sim(meta_sim, meta_df, meta_idx, mid, K)
    if recs and len(recs) >= K//2:
        prec = genre_precision_at_k(mid, recs, K)
        precisions.append(prec)
        
        ndcg = calculate_ndcg_for_movie(mid, recs, K)
        ndcgs.append(ndcg)
        
        similar_movies = get_recs_by_sim(meta_sim, meta_df, meta_idx, mid, 50)
        relevant = set(similar_movies[:20]) if similar_movies else set()
        
        mrr = mrr_at_k(recs, relevant, K)
        mrrs.append(mrr)
        
        hit = hit_rate_at_k(recs, relevant, K)
        hit_rates.append(hit)
        
        map_score = average_precision_at_k(recs, relevant, K)
        maps.append(map_score)
        
        diversities.append(genre_diversity(recs))
        year_divs.append(year_diversity(recs))
        all_recs.extend(recs)
        
        detailed_results["Metadata"]["precision"].append(prec)
        detailed_results["Metadata"]["ndcg"].append(ndcg)
        detailed_results["Metadata"]["mrr"].append(mrr)
        detailed_results["Metadata"]["hit_rate"].append(hit)

results["Metadata"] = {
    "precision@10"  : round(np.mean(precisions), 4) if precisions else 0.0,
    "ndcg@10"       : round(np.mean(ndcgs), 4) if ndcgs else 0.0,
    "mrr@10"        : round(np.mean(mrrs), 4) if mrrs else 0.0,
    "hit_rate@10"   : round(np.mean(hit_rates), 4) if hit_rates else 0.0,
    "map@10"        : round(np.mean(maps), 4) if maps else 0.0,
    "avg_diversity" : round(np.mean(diversities), 2) if diversities else 0.0,
    "year_diversity": round(np.mean(year_divs), 2) if year_divs else 0.0,
    "coverage"      : round(coverage(all_recs, len(master)), 4),
    "tested_movies" : len(precisions),
    "std_precision" : round(np.std(precisions), 4) if precisions else 0.0,
    "std_ndcg"      : round(np.std(ndcgs), 4) if ndcgs else 0.0
}
print(f"   Precision@10 : {results['Metadata']['precision@10']:.4f} (±{results['Metadata']['std_precision']:.4f})")
print(f"   NDCG@10      : {results['Metadata']['ndcg@10']:.4f} (±{results['Metadata']['std_ndcg']:.4f})")
print(f"   MRR@10       : {results['Metadata']['mrr@10']:.4f}")
print(f"   Hit Rate@10  : {results['Metadata']['hit_rate@10']:.4f}")
print(f"   MAP@10       : {results['Metadata']['map@10']:.4f}")


# ─── Engine 3: Visual ───────────────────────────────────
print("\n[3/7] Evaluating Visual with Advanced Metrics...")
precisions, ndcgs, mrrs, hit_rates, maps, diversities, year_divs, all_recs = [], [], [], [], [], [], [], []

for mid in test_ids:
    recs = get_recs_by_sim(visual_sim, visual_df, visual_idx, mid, K)
    if recs and len(recs) >= K//2:
        prec = genre_precision_at_k(mid, recs, K)
        precisions.append(prec)
        
        ndcg = calculate_ndcg_for_movie(mid, recs, K)
        ndcgs.append(ndcg)
        
        similar_movies = get_recs_by_sim(visual_sim, visual_df, visual_idx, mid, 50)
        relevant = set(similar_movies[:20]) if similar_movies else set()
        
        mrr = mrr_at_k(recs, relevant, K)
        mrrs.append(mrr)
        
        hit = hit_rate_at_k(recs, relevant, K)
        hit_rates.append(hit)
        
        map_score = average_precision_at_k(recs, relevant, K)
        maps.append(map_score)
        
        diversities.append(genre_diversity(recs))
        year_divs.append(year_diversity(recs))
        all_recs.extend(recs)
        
        detailed_results["Visual"]["precision"].append(prec)
        detailed_results["Visual"]["ndcg"].append(ndcg)
        detailed_results["Visual"]["mrr"].append(mrr)
        detailed_results["Visual"]["hit_rate"].append(hit)

results["Visual"] = {
    "precision@10"  : round(np.mean(precisions), 4) if precisions else 0.0,
    "ndcg@10"       : round(np.mean(ndcgs), 4) if ndcgs else 0.0,
    "mrr@10"        : round(np.mean(mrrs), 4) if mrrs else 0.0,
    "hit_rate@10"   : round(np.mean(hit_rates), 4) if hit_rates else 0.0,
    "map@10"        : round(np.mean(maps), 4) if maps else 0.0,
    "avg_diversity" : round(np.mean(diversities), 2) if diversities else 0.0,
    "year_diversity": round(np.mean(year_divs), 2) if year_divs else 0.0,
    "coverage"      : round(coverage(all_recs, len(master)), 4),
    "tested_movies" : len(precisions),
    "std_precision" : round(np.std(precisions), 4) if precisions else 0.0,
    "std_ndcg"      : round(np.std(ndcgs), 4) if ndcgs else 0.0
}
print(f"   Precision@10 : {results['Visual']['precision@10']:.4f} (±{results['Visual']['std_precision']:.4f})")
print(f"   NDCG@10      : {results['Visual']['ndcg@10']:.4f} (±{results['Visual']['std_ndcg']:.4f})")
print(f"   MRR@10       : {results['Visual']['mrr@10']:.4f}")
print(f"   Hit Rate@10  : {results['Visual']['hit_rate@10']:.4f}")
print(f"   MAP@10       : {results['Visual']['map@10']:.4f}")


# ─── Engine 4: Collaborative ────────────────────────────
print("\n[4/7] Evaluating Collaborative with Precision@K...")

# Test على test_user_ids المختارين
user_sparse  = sparse_matrix.T.tocsr()
y_true, y_pred = [], []

# للـ Precision@K - تقييم على مستوى المستخدم
user_precisions, user_ndcgs, user_mrrs, user_hit_rates = [], [], [], []
min_ratings_per_user = 5

# جلب جميع ratings للمستخدمين المختارين
test_user_ratings = ratings[ratings['userId'].isin(test_user_ids)]

# تجميع التقييمات حسب المستخدم
user_ratings = test_user_ratings.groupby('userId')

skipped_users = {"min_ratings": 0, "not_in_user2idx": 0, "no_relevant": 0, "exception": 0}

for uid, user_df in user_ratings:
    if len(user_df) < min_ratings_per_user:
        skipped_users["min_ratings"] += 1
        continue
    if uid not in user2idx:
        skipped_users["not_in_user2idx"] += 1
        continue
    
    uidx = user2idx[uid]
    
    # تقسيم عشوائي 80/20 per user (random split)
    user_df_shuffled = user_df.sample(frac=1, random_state=42)
    
    # تصفية الأفلام لتكون فقط اللي موجودة في movie2idx
    user_df_filtered = user_df_shuffled[user_df_shuffled['movie_id'].isin(movie2idx.keys())]
    if len(user_df_filtered) < min_ratings_per_user:
        skipped_users["min_ratings"] += 1
        continue
    
    user_movies = user_df_filtered['movie_id'].tolist()
    user_ratings_list = user_df_filtered['rating'].tolist()
    
    # استخدام 20% عشوائي كـ test
    split_point = int(len(user_movies) * 0.8)
    train_movies = user_movies[:split_point]
    test_movies = user_movies[split_point:]
    test_ratings_list = user_ratings_list[split_point:]
    
    # الأفلام اللي المستخدم قيّمها بـ 4 أو 5 نعتبرها relevant
    relevant_movies = [m for m, r in zip(test_movies, test_ratings_list) if r >= 4.0]
    # لو مفيش relevant، نستخدم كل test movies كـ relevant (للواقعية)
    if not relevant_movies:
        relevant_movies = test_movies
    if not relevant_movies:
        skipped_users["no_relevant"] += 1
        continue
    
    # حساب توقعات لكل الأفلام للمستخدم ده
    try:
        user_vector = collab_model.user_factors[uidx]
        all_predictions = []
        
        for mid in movie2idx.keys():
            if mid in train_movies:  # Skip train movies
                continue
            if mid not in movie2idx:  # Double check
                continue
            midx = movie2idx[mid]
            # Check bounds
            if midx >= len(collab_model.item_factors):
                continue
            pred_score = float(user_vector @ collab_model.item_factors[midx])
            all_predictions.append((mid, pred_score))
        
        # ترتيب واختيار top-K
        all_predictions.sort(key=lambda x: x[1], reverse=True)
        top_k_recs = [m for m, _ in all_predictions[:K]]
        
        if top_k_recs:
            # Precision@K
            hits = len(set(top_k_recs) & set(relevant_movies))
            prec = hits / K
            user_precisions.append(prec)
            
            # Hit Rate
            hit = 1.0 if hits > 0 else 0.0
            user_hit_rates.append(hit)
            
            # MRR
            for i, mid in enumerate(top_k_recs):
                if mid in relevant_movies:
                    user_mrrs.append(1.0 / (i + 1))
                    break
            else:
                user_mrrs.append(0.0)
            
    except Exception as e:
        skipped_users["exception"] += 1
        continue

# RMSE/MAE Evaluation - على sample من ratings
sample_ratings = ratings.sample(min(5000, len(ratings)), random_state=42)
for _, row in sample_ratings.iterrows():
    uid = row["userId"]
    mid = row["movie_id"]
    if uid not in user2idx or mid not in movie2idx:
        continue
    uidx = user2idx[uid]
    midx = movie2idx[mid]
    try:
        score = float(
            collab_model.user_factors[uidx] @
            collab_model.item_factors[midx]
        )
        y_true.append(row["rating"])
        y_pred.append(np.clip(score, 0.5, 5.0))
    except:
        continue

if y_true:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    
    results["Collaborative"] = {
        "RMSE"          : round(rmse, 4),
        "MAE"           : round(mae, 4),
        "precision@10"  : round(np.mean(user_precisions), 4) if user_precisions else 0.0,
        "ndcg@10"       : None,  # يحتاج relevance scores
        "mrr@10"        : round(np.mean(user_mrrs), 4) if user_mrrs else 0.0,
        "hit_rate@10"   : round(np.mean(user_hit_rates), 4) if user_hit_rates else 0.0,
        "map@10"        : None,
        "avg_diversity" : None,
        "year_diversity": None,
        "coverage"      : None,
        "tested_users"  : len(user_precisions),
        "tested_ratings": len(y_true),
        "std_precision" : round(np.std(user_precisions), 4) if user_precisions else 0.0
    }
    print(f"   RMSE: {rmse:.4f}  |  MAE: {mae:.4f}")
    print(f"   Precision@10: {results['Collaborative']['precision@10']:.4f} (±{results['Collaborative']['std_precision']:.4f}) [on {len(user_precisions)} users]")
    print(f"   MRR@10: {results['Collaborative']['mrr@10']:.4f}")
    print(f"   Hit Rate@10: {results['Collaborative']['hit_rate@10']:.4f}")


# ─── Engine 5: Popularity ───────────────────────────────
print("\n[5/7] Evaluating Popularity with Advanced Metrics...")

# Popularity: قياس correlation بين weighted_score والـ actual ratings
merged = pop_df.merge(
    ratings.groupby("movie_id")["rating"].mean().reset_index(),
    on="movie_id", how="inner"
)
corr = merged["weighted_score"].corr(merged["rating"])

# Evaluation على test movies بالطريقة نفسها
precisions, ndcgs, mrrs, hit_rates, maps, diversities, year_divs = [], [], [], [], [], [], []

for mid in test_ids:
    # Popularity recs: top popular movies excluding the input
    pop_recs = pop_df[pop_df["movie_id"] != mid].head(K)["movie_id"].tolist()
    
    if pop_recs:
        prec = genre_precision_at_k(mid, pop_recs, K)
        precisions.append(prec)
        
        ndcg = calculate_ndcg_for_movie(mid, pop_recs, K)
        ndcgs.append(ndcg)
        
        # للـ Popularity، relevant = أفلام مشابهة في الـ genre
        input_genres = get_genres(mid)
        all_movies_with_genre = master[master["movie_id"].isin(pop_df["movie_id"])]
        similar_in_genre = []
        for _, mrow in all_movies_with_genre.iterrows():
            if get_genres(mrow["movie_id"]) & input_genres:
                similar_in_genre.append(mrow["movie_id"])
        
        relevant = set(similar_in_genre[:20])
        
        mrr = mrr_at_k(pop_recs, relevant, K)
        mrrs.append(mrr)
        
        hit = hit_rate_at_k(pop_recs, relevant, K)
        hit_rates.append(hit)
        
        map_score = average_precision_at_k(pop_recs, relevant, K)
        maps.append(map_score)
        
        diversities.append(genre_diversity(pop_recs))
        year_divs.append(year_diversity(pop_recs))

# Precision: Top 100 popular هل تقييماتهم عالية؟
top100     = pop_df.head(100)
high_rated = (top100["vote_average"] >= 7.0).mean()

results["Popularity"] = {
    "correlation_with_ratings" : round(corr, 4),
    "high_rated_in_top100"     : round(high_rated, 4),
    "precision@10"             : round(np.mean(precisions), 4) if precisions else 0.0,
    "ndcg@10"                  : round(np.mean(ndcgs), 4) if ndcgs else 0.0,
    "mrr@10"                   : round(np.mean(mrrs), 4) if mrrs else 0.0,
    "hit_rate@10"              : round(np.mean(hit_rates), 4) if hit_rates else 0.0,
    "map@10"                   : round(np.mean(maps), 4) if maps else 0.0,
    "avg_diversity"            : round(np.mean(diversities), 2) if diversities else 0.0,
    "year_diversity"           : round(np.mean(year_divs), 2) if year_divs else 0.0,
    "coverage"                 : round(len(pop_df) / len(master), 4),
    "tested_movies"            : len(precisions),
    "std_precision"            : round(np.std(precisions), 4) if precisions else 0.0,
    "std_ndcg"                 : round(np.std(ndcgs), 4) if ndcgs else 0.0
}
print(f"   Precision@10 : {results['Popularity']['precision@10']:.4f} (±{results['Popularity']['std_precision']:.4f})")
print(f"   NDCG@10      : {results['Popularity']['ndcg@10']:.4f} (±{results['Popularity']['std_ndcg']:.4f})")
print(f"   MRR@10       : {results['Popularity']['mrr@10']:.4f}")
print(f"   Hit Rate@10  : {results['Popularity']['hit_rate@10']:.4f}")
print(f"   Correlation  : {corr:.4f}")


# ─── Engine 6: Hybrid ───────────────────────────────────
print("\n[6/7] Evaluating Hybrid with Advanced Metrics...")
from sklearn.preprocessing import MinMaxScaler

def hybrid_scores(movie_id, k=10):
    """حساب scores الهجينة بشكل محسّن"""
    row = master[master["movie_id"] == movie_id]
    if row.empty:
        return []
    title = row.iloc[0]["title"].lower()

    scores = defaultdict(list)
    
    # Weights for different engines
    engine_weights = {
        'content': 0.4,
        'meta': 0.3,
        'visual': 0.2,
        'pop': 0.1
    }
    
    # Content-based similarity
    if title in content_idx:
        idx = content_idx[title]
        if hasattr(idx, '__len__'):
            idx = int(idx.iloc[0])
        else:
            idx = int(idx)
        sim_row = content_sim[idx]
        for i, s in enumerate(sim_row):
            mid2 = content_df.iloc[i]["movie_id"]
            scores[mid2].append(float(s) * engine_weights['content'])
    
    # Metadata similarity
    if title in meta_idx:
        idx = meta_idx[title]
        if hasattr(idx, '__len__'):
            idx = int(idx.iloc[0])
        else:
            idx = int(idx)
        sim_row = meta_sim[idx]
        for i, s in enumerate(sim_row):
            mid2 = meta_df.iloc[i]["movie_id"]
            scores[mid2].append(float(s) * engine_weights['meta'])
    
    # Visual similarity
    if title in visual_idx:
        idx = visual_idx[title]
        if hasattr(idx, '__len__'):
            idx = int(idx.iloc[0])
        else:
            idx = int(idx)
        sim_row = visual_sim[idx]
        for i, s in enumerate(sim_row):
            mid2 = visual_df.iloc[i]["movie_id"]
            scores[mid2].append(float(s) * engine_weights['visual'])

    if not scores:
        return []

    # Add popularity boost
    pop_map = dict(zip(pop_df["movie_id"], pop_df["weighted_score"]))
    pop_min = pop_df["weighted_score"].min()
    pop_max = pop_df["weighted_score"].max()

    final = {}
    for mid2, ss in scores.items():
        # Combine similarities
        content_score = np.mean([s for s in ss if s > 0]) if ss else 0
        pop_score = (pop_map.get(mid2, pop_min) - pop_min) / (pop_max - pop_min + 1e-8)
        
        # Weighted combination
        final[mid2] = content_score * 0.9 + pop_score * 0.1

    final.pop(movie_id, None)
    top = sorted(final.items(), key=lambda x: x[1], reverse=True)[:k]
    return [m for m, _ in top]

precisions, ndcgs, mrrs, hit_rates, maps, diversities, year_divs, all_recs = [], [], [], [], [], [], [], []

for mid in test_ids:
    recs = hybrid_scores(mid, K)
    if recs and len(recs) >= K//2:
        prec = genre_precision_at_k(mid, recs, K)
        precisions.append(prec)
        
        ndcg = calculate_ndcg_for_movie(mid, recs, K)
        ndcgs.append(ndcg)
        
        # Get relevant movies from all engines
        relevant = set()
        for engine_sim, engine_df, engine_idx in [
            (content_sim, content_df, content_idx),
            (meta_sim, meta_df, meta_idx),
            (visual_sim, visual_df, visual_idx)
        ]:
            sims = get_recs_by_sim(engine_sim, engine_df, engine_idx, mid, 20)
            relevant.update(sims)
        
        mrr = mrr_at_k(recs, relevant, K)
        mrrs.append(mrr)
        
        hit = hit_rate_at_k(recs, relevant, K)
        hit_rates.append(hit)
        
        map_score = average_precision_at_k(recs, relevant, K)
        maps.append(map_score)
        
        diversities.append(genre_diversity(recs))
        year_divs.append(year_diversity(recs))
        all_recs.extend(recs)
        
        detailed_results["Hybrid"]["precision"].append(prec)
        detailed_results["Hybrid"]["ndcg"].append(ndcg)
        detailed_results["Hybrid"]["mrr"].append(mrr)
        detailed_results["Hybrid"]["hit_rate"].append(hit)

results["Hybrid"] = {
    "precision@10"  : round(np.mean(precisions), 4) if precisions else 0.0,
    "ndcg@10"       : round(np.mean(ndcgs), 4) if ndcgs else 0.0,
    "mrr@10"        : round(np.mean(mrrs), 4) if mrrs else 0.0,
    "hit_rate@10"   : round(np.mean(hit_rates), 4) if hit_rates else 0.0,
    "map@10"        : round(np.mean(maps), 4) if maps else 0.0,
    "avg_diversity" : round(np.mean(diversities), 2) if diversities else 0.0,
    "year_diversity": round(np.mean(year_divs), 2) if year_divs else 0.0,
    "coverage"      : round(coverage(all_recs, len(master)), 4),
    "tested_movies" : len(precisions),
    "std_precision" : round(np.std(precisions), 4) if precisions else 0.0,
    "std_ndcg"      : round(np.std(ndcgs), 4) if ndcgs else 0.0
}
print(f"   Precision@10 : {results['Hybrid']['precision@10']:.4f} (±{results['Hybrid']['std_precision']:.4f})")
print(f"   NDCG@10      : {results['Hybrid']['ndcg@10']:.4f} (±{results['Hybrid']['std_ndcg']:.4f})")
print(f"   MRR@10       : {results['Hybrid']['mrr@10']:.4f}")
print(f"   Hit Rate@10  : {results['Hybrid']['hit_rate@10']:.4f}")
print(f"   MAP@10       : {results['Hybrid']['map@10']:.4f}")


# ─── Engine 7: Sequence (LSTM) ────────────────────────────
print("\n[7/7] Evaluating Sequence (LSTM) with Advanced Metrics...")

if seq_model_available:
    precisions, ndcgs, mrrs, hit_rates, maps, diversities, year_divs, all_recs = [], [], [], [], [], [], [], []
    
    for mid in test_ids:
        # For sequence model, we need a sequence leading to this movie
        # Create a simple sequence: start with the movie itself as the last watched
        # In real scenario, we'd have user's watch history
        test_sequence = [mid]
        
        recs = get_sequence_recs(test_sequence, K)
        if recs and len(recs) >= K//2:
            prec = genre_precision_at_k(mid, recs, K)
            precisions.append(prec)
            
            ndcg = calculate_ndcg_for_movie(mid, recs, K)
            ndcgs.append(ndcg)
            
            # Get relevant movies from other engines as ground truth
            relevant = set()
            for engine_sim, engine_df, engine_idx in [
                (content_sim, content_df, content_idx),
                (meta_sim, meta_df, meta_idx),
                (visual_sim, visual_df, visual_idx)
            ]:
                sims = get_recs_by_sim(engine_sim, engine_df, engine_idx, mid, 20)
                relevant.update(sims)
            
            mrr = mrr_at_k(recs, relevant, K)
            mrrs.append(mrr)
            
            hit = hit_rate_at_k(recs, relevant, K)
            hit_rates.append(hit)
            
            map_score = average_precision_at_k(recs, relevant, K)
            maps.append(map_score)
            
            diversities.append(genre_diversity(recs))
            year_divs.append(year_diversity(recs))
            all_recs.extend(recs)
            
            detailed_results["Sequence"]["precision"].append(prec)
            detailed_results["Sequence"]["ndcg"].append(ndcg)
            detailed_results["Sequence"]["mrr"].append(mrr)
            detailed_results["Sequence"]["hit_rate"].append(hit)
    
    results["Sequence"] = {
        "precision@10"  : round(np.mean(precisions), 4) if precisions else 0.0,
        "ndcg@10"       : round(np.mean(ndcgs), 4) if ndcgs else 0.0,
        "mrr@10"        : round(np.mean(mrrs), 4) if mrrs else 0.0,
        "hit_rate@10"   : round(np.mean(hit_rates), 4) if hit_rates else 0.0,
        "map@10"        : round(np.mean(maps), 4) if maps else 0.0,
        "avg_diversity" : round(np.mean(diversities), 2) if diversities else 0.0,
        "year_diversity": round(np.mean(year_divs), 2) if year_divs else 0.0,
        "coverage"      : round(coverage(all_recs, len(master)), 4),
        "tested_movies" : len(precisions),
        "std_precision" : round(np.std(precisions), 4) if precisions else 0.0,
        "std_ndcg"      : round(np.std(ndcgs), 4) if ndcgs else 0.0
    }
    print(f"   Precision@10 : {results['Sequence']['precision@10']:.4f} (±{results['Sequence']['std_precision']:.4f})")
    print(f"   NDCG@10      : {results['Sequence']['ndcg@10']:.4f} (±{results['Sequence']['std_ndcg']:.4f})")
    print(f"   MRR@10       : {results['Sequence']['mrr@10']:.4f}")
    print(f"   Hit Rate@10  : {results['Sequence']['hit_rate@10']:.4f}")
    print(f"   MAP@10       : {results['Sequence']['map@10']:.4f}")
else:
    print("   Sequence model not available (skipping)")
    results["Sequence"] = {
        "precision@10"  : None, "ndcg@10"  : None, "mrr@10" : None,
        "hit_rate@10"   : None, "map@10"   : None, "avg_diversity" : None,
        "year_diversity": None, "coverage" : None, "tested_movies" : 0
    }


# ════════════════════════════════════════════════════════
# عرض النتايج النهائية - محسّنة
# ════════════════════════════════════════════════════════
print("\n" + "=" * 85)
print("  COMPREHENSIVE EVALUATION RESULTS - دقيقة جداً")
print("=" * 85)

# جدول شامل بكل المقاييس
print(f"\n{'Engine':<16} {'P@10':>8} {'NDCG@10':>9} {'MRR@10':>8} {'Hit@10':>8} {'MAP@10':>8} {'Div':>6}")
print("-" * 75)

ranking_by_precision = []
ranking_by_ndcg = []

for engine, r in results.items():
    p     = f"{r['precision@10']:.3f}"    if r.get('precision@10')  is not None else "   -  "
    ndcg  = f"{r['ndcg@10']:.3f}"       if r.get('ndcg@10')       is not None else "   -  "
    mrr   = f"{r['mrr@10']:.3f}"        if r.get('mrr@10')        is not None else "   -  "
    hit   = f"{r['hit_rate@10']:.3f}"   if r.get('hit_rate@10')   is not None else "   -  "
    map_  = f"{r['map@10']:.3f}"        if r.get('map@10')        is not None else "   -  "
    div   = f"{r['avg_diversity']:.1f}" if r.get('avg_diversity') is not None else "  - "
    
    print(f"{engine:<16} {p:>8} {ndcg:>9} {mrr:>8} {hit:>8} {map_:>8} {div:>6}")
    
    if r.get('precision@10') is not None:
        ranking_by_precision.append((engine, r['precision@10'], r.get('std_precision', 0)))
    if r.get('ndcg@10') is not None:
        ranking_by_ndcg.append((engine, r['ndcg@10'], r.get('std_ndcg', 0)))

# Collaborative metrics منفصلة
if "Collaborative" in results:
    r = results["Collaborative"]
    print(f"\n{'Collaborative':<16} RMSE={r['RMSE']:.4f}  MAE={r['MAE']:.4f}  "
          f"[on {r.get('tested_users', 0)} users, {r.get('tested_ratings', 0)} ratings]")

print("\n" + "=" * 85)

# التصنيف حسب Precision@10
if ranking_by_precision:
    ranking_by_precision.sort(key=lambda x: x[1], reverse=True)
    print("\n  🏆 RANKING BY Precision@10:")
    print("-" * 50)
    for i, (eng, score, std) in enumerate(ranking_by_precision, 1):
        bar = "█" * int(score * 25)
        print(f"  {i}. {eng:<16} {score:.4f} (±{std:.4f})  {bar}")

# التصنيف حسب NDCG@10
if ranking_by_ndcg:
    ranking_by_ndcg.sort(key=lambda x: x[1], reverse=True)
    print("\n  📊 RANKING BY NDCG@10 (أهم مقياس للترتيب):")
    print("-" * 50)
    for i, (eng, score, std) in enumerate(ranking_by_ndcg, 1):
        bar = "█" * int(score * 25)
        print(f"  {i}. {eng:<16} {score:.4f} (±{std:.4f})  {bar}")

# أفضل Engine شاملاً
print("\n" + "=" * 85)
if ranking_by_ndcg:
    winner = ranking_by_ndcg[0][0]
    winner_score = ranking_by_ndcg[0][1]
    print(f"\n  🥇 BEST OVERALL ENGINE: {winner}")
    print(f"      NDCG@10 = {winner_score:.4f} (أفضل مقياس لتقييم الترتيب)")

# مقارنة Hybrid vs أفضل individual engine
if "Hybrid" in detailed_results and ranking_by_ndcg:
    best_individual = ranking_by_ndcg[1][0] if len(ranking_by_ndcg) > 1 and ranking_by_ndcg[0][0] == "Hybrid" else ranking_by_ndcg[0][0]
    if best_individual in detailed_results and best_individual != "Hybrid":
        hybrid_ndcgs = detailed_results["Hybrid"]["ndcg"]
        other_ndcgs = detailed_results[best_individual]["ndcg"]
        if len(hybrid_ndcgs) == len(other_ndcgs) and len(hybrid_ndcgs) > 1:
            t_stat, p_value = paired_t_test(hybrid_ndcgs, other_ndcgs)
            if p_value is not None:
                print(f"\n  📈 Statistical Test (Hybrid vs {best_individual}):")
                print(f"      p-value = {p_value:.4f}", "✅ Significant!" if p_value < 0.05 else "❌ Not significant")

print("\n" + "=" * 85)

# حفظ النتايج التفصيلية
results_df = pd.DataFrame(results).T
detailed_df = pd.DataFrame({
    engine: {
        'precision_list': detailed_results[engine].get('precision', []),
        'ndcg_list': detailed_results[engine].get('ndcg', []),
        'mrr_list': detailed_results[engine].get('mrr', []),
        'hit_rate_list': detailed_results[engine].get('hit_rate', [])
    }
    for engine in detailed_results.keys()
})

results_df.to_csv("models/evaluation_results.csv")
detailed_df.to_json("models/evaluation_detailed.json", orient='index')
print("\n📁 Results saved to:")
print("   - models/evaluation_results.csv")
print("   - models/evaluation_detailed.json")
print("\n✅ Evaluation complete!")