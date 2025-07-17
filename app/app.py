# app/app.py

import streamlit as st
import json
import os

from src.scraper import fetch_hashtag_posts, save_posts_to_json
from src.cleaner import clean_captions
from src.sentiment import load_sentiment_model, analyze_sentiment
from src.keywords import load_keyword_model, extract_keywords, get_keyword_frequency
from src.search import search_by_keyword
from visualization.charts import plot_sentiment_distribution
from visualization.wordcloud import generate_wordcloud


DATA_DIR = "../data"
FONT_PATH = "c:/Windows/Fonts/malgun.ttf"  # 한글 지원 폰트 (윈도우 기준)


def run_pipeline(hashtag, max_posts):
    st.info("📥 게시글 수집 중...")
    posts = fetch_hashtag_posts(hashtag, max_count=max_posts)
    save_path = os.path.join(DATA_DIR, f"{hashtag}_raw.json")
    save_posts_to_json(posts, save_path)

    st.success(f"✅ 게시글 {len(posts)}개 수집 완료")

    st.info("🧹 텍스트 정제 중...")
    posts = clean_captions(posts)

    st.info("📊 감정 분석 중...")
    sentiment_model = load_sentiment_model()
    posts = analyze_sentiment(posts, sentiment_model)

    st.info("🔑 키워드 추출 중...")
    keyword_model = load_keyword_model()
    posts = extract_keywords(posts, keyword_model)

    output_path = os.path.join(DATA_DIR, f"{hashtag}_final.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)

    st.success("🎉 전체 파이프라인 완료!")
    return posts


def main():
    st.set_page_config(page_title="인스타그램 해시태그 분석기", layout="wide")
    st.title("📸 인스타그램 해시태그 분석기 (공개 데이터 기반)")

    hashtag = st.text_input("분석할 해시태그를 입력하세요 (예: 여행, ootd):").strip().replace("#", "")
    max_posts = st.slider("수집할 게시글 수", min_value=20, max_value=500, value=100, step=10)

    if st.button("분석 시작"):
        if hashtag:
            posts = run_pipeline(hashtag, max_posts)

            st.subheader("📈 감정 분석 결과")
            plot_sentiment_distribution(posts)

            st.subheader("☁️ 워드클라우드")
            generate_wordcloud(posts, font_path=FONT_PATH)

            st.subheader("🔎 키워드로 검색")
            search_term = st.text_input("검색할 키워드 입력:")
            if search_term:
                results = search_by_keyword(posts, search_term)
                st.write(f"'{search_term}' 키워드 포함 게시글: {len(results)}개")
                for i, post in enumerate(results[:5], 1):
                    st.markdown(f"**{i}.** {post['cleaned_caption']}")

        else:
            st.warning("해시태그를 입력해주세요.")


if __name__ == "__main__":
    main()
