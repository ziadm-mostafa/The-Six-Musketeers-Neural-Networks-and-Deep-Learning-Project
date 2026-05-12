"""
TMDB Professional Movie Scraper

يجمع بيانات شاملة لكل الموديلات الستة في run واحد.
"""

import os
import time
import json
import requests
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime

#  إعداد الـ Logger 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TMDBScraper:
    """Scraper احترافي لـ TMDB API"""

    def __init__(self, api_key: str, delay: float = 0.25):
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self.image_base = "https://image.tmdb.org/t/p/w500"
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "MovieRecommender/1.0"
        })

    # Request مع Retry تلقائي 
    def _get(self, endpoint: str, params: dict = {}) -> dict:
        params["api_key"] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(3):  # 3 محاولات
            try:
                response = self.session.get(url, params=params, timeout=10)
                
                if response.status_code == 429:  # Rate limit
                    wait = int(response.headers.get("Retry-After", 10))
                    logger.warning(f"Rate limit! Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.RequestException as e:
                logger.error(f"Attempt {attempt+1}/3 failed: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return {}

    # جمع معرفات الأفلام من discover 
    def fetch_movie_ids(self, total_pages: int = 500) -> list:
        """جمع IDs الأفلام من discover endpoint"""
        movie_ids = set()
        
        logger.info(f"🎬 Fetching movie IDs ({total_pages} pages)...")
        
        for page in tqdm(range(1, total_pages + 1), desc="Fetching IDs"):
            data = self._get("discover/movie", {
                "sort_by": "popularity.desc",
                "vote_count.gte": 50,
                "page": page,
                "language": "en-US"
            })
            
            for movie in data.get("results", []):
                movie_ids.add(movie["id"])
            
            time.sleep(self.delay)
        
        logger.info(f"✅ Found {len(movie_ids)} unique movie IDs")
        return list(movie_ids)

    # جمع تفاصيل فيلم واحد 
    def fetch_movie_details(self, movie_id: int) -> dict | None:
        """جمع كل بيانات فيلم واحد (details + credits + keywords)"""
        
        # Details الأساسية
        details = self._get(f"movie/{movie_id}", {
            "append_to_response": "credits,keywords",
            "language": "en-US"
        })
        
        if not details or details.get("status_code") == 34:
            return None  # فيلم مش موجود
        
        # استخراج البيانات 
        
        # الممثلين (أول 5)
        cast = details.get("credits", {}).get("cast", [])
        cast_names = [c["name"] for c in cast[:5]]
        
        # طاقم العمل
        crew = details.get("credits", {}).get("crew", [])
        director = next(
            (p["name"] for p in crew if p["job"] == "Director"), None
        )
        writer = next(
            (p["name"] for p in crew if p["job"] in ["Writer", "Screenplay"]),
            None
        )
        
        # الكلمات المفتاحية
        keywords = details.get("keywords", {}).get("keywords", [])
        keyword_names = [k["name"] for k in keywords[:10]]
        
        # الأنواع
        genres = [g["name"] for g in details.get("genres", [])]
        
        # شركات الإنتاج
        companies = [c["name"] for c in details.get("production_companies", [])[:3]]
        
        # بناء الـ Record
        record = {
            # معلومات أساسية 
            "movie_id":        details.get("id"),
            "title":           details.get("title"),
            "original_title":  details.get("original_title"),
            "release_date":    details.get("release_date"),
            "release_year":    str(details.get("release_date", ""))[:4],
            "runtime":         details.get("runtime"),
            "status":          details.get("status"),
            "language":        details.get("original_language"),
            
            # للموديل 1: Content-Based (NLP)
            "overview":        details.get("overview", ""),
            "tagline":         details.get("tagline", ""),
            
            # للموديل 3: Metadata 
            "genres":          "|".join(genres),
            "cast":            "|".join(cast_names),
            "director":        director,
            "writer":          writer,
            "keywords":        "|".join(keyword_names),
            "production_companies": "|".join(companies),
            "collection":      details.get("belongs_to_collection", {}).get("name") if details.get("belongs_to_collection") else None,
            
            # للموديل 4: Visual 
            "poster_path":     f"https://image.tmdb.org/t/p/w500{details.get('poster_path')}" if details.get("poster_path") else None,
            "backdrop_path":   f"https://image.tmdb.org/t/p/w780{details.get('backdrop_path')}" if details.get("backdrop_path") else None,
            
            # للموديل 6: Popularity 
            "vote_average":    details.get("vote_average"),
            "vote_count":      details.get("vote_count"),
            "popularity":      details.get("popularity"),
            "budget":          details.get("budget"),
            "revenue":         details.get("revenue"),
            
            # للعرض 
            "tmdb_url":        f"https://www.themoviedb.org/movie/{movie_id}",
            "imdb_id":         details.get("imdb_id"),
        }
        
        return record

    # Run الرئيسي
    def run(self, total_pages: int = 500, output_path: str = "data/raw/movies.csv"):
        """تشغيل الـ Scraper الكامل"""
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # جمع IDs
        movie_ids = self.fetch_movie_ids(total_pages)
        
        # جمع التفاصيل
        movies = []
        failed = []
        
        logger.info(f"📥 Fetching details for {len(movie_ids)} movies...")
        
        for i, movie_id in enumerate(tqdm(movie_ids, desc="Movie Details")):
            record = self.fetch_movie_details(movie_id)
            
            if record:
                movies.append(record)
            else:
                failed.append(movie_id)
            
            # حفظ تدريجي كل 500 فيلم
            if (i + 1) % 500 == 0:
                self._save_checkpoint(movies, output_path)
                logger.info(f"💾 Checkpoint: {len(movies)} movies saved")
            
            time.sleep(self.delay)
        
        # حفظ نهائي
        df = pd.DataFrame(movies)
        df = self._clean_data(df)
        df.to_csv(output_path, index=False, encoding="utf-8")
        
        logger.info(f"✅ Done! {len(df)} movies saved to {output_path}")
        logger.info(f"❌ Failed: {len(failed)} movies")
        
        return df

    # تنظيف أولي
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """تنظيف أولي للبيانات"""
        # إزالة أفلام بدون overview
        df = df[df["overview"].str.len() > 20]
        
        # إزالة duplicates
        df = df.drop_duplicates(subset="movie_id")
        
        # فلتر جودة
        df = df[
            (df["vote_count"] >= 50) & 
            (df["vote_average"] >= 3.0) &
            (df["overview"].notna())
        ]
        
        df = df.reset_index(drop=True)
        return df

    def _save_checkpoint(self, movies: list, output_path: str):
        checkpoint_path = output_path.replace(".csv", "_checkpoint.csv")
        pd.DataFrame(movies).to_csv(checkpoint_path, index=False)


# Main
if __name__ == "__main__":
    from config import TMDB_API_KEY, TOTAL_PAGES
    
    scraper = TMDBScraper(api_key=TMDB_API_KEY, delay=0.25)
    df = scraper.run(
        total_pages=TOTAL_PAGES,
        output_path="data/raw/movies.csv"
    )
    
    print("\n عينة من البيانات:")
    print(df[["title", "genres", "director", "vote_average"]].head(10))
    print(f"\n الإجمالي: {len(df)} فيلم")