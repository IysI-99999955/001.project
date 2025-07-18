# app/app.py

import sys
import os
# --- 모듈 검색 경로 설정 시작 ---
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
project_root_dir = os.path.join(current_dir, "..")
sys.path.insert(0, project_root_dir)
# --- 모듈 검색 경로 설정 끝 ---

import json 
from datetime import datetime, timedelta
from collections import Counter
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv

# src 및 visualization 모듈 임포트
from src.cleaner import clean_captions
from src.sentiment import load_sentiment_model, analyze_sentiment
from src.keywords import load_keyword_model, extract_keywords, get_keyword_frequency
from src.search import search_by_keyword
from visualization.charts import plot_sentiment_distribution
from visualization.wordcloud import generate_wordcloud

# Langchain 및 Upstage 관련 모듈 임포트
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate

# Apify 클라이언트 임포트
from apify_client import ApifyClient

# 환경 변수 로드
load_dotenv()
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")

# 전역 설정
DATA_DIR = "../data"
INSTAGRAM_SCRAPER_ACTOR_ID = "apify/instagram-hashtag-scraper"

# 폰트 경로 설정
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
project_root_dir = os.path.join(current_dir, "..")
FONT_PATH = os.path.join(project_root_dir, "fonts", "NotoSansKR-Regular.ttf")


def initialize_session_state():
    """세션 상태 초기화"""
    defaults = {
        "analyzed_posts": [],
        "vectorstore": None,
        "chat_history": [],
        "current_hashtag": "",
        "current_max_posts": 50
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def fetch_posts_from_apify(hashtag: str, max_count: int, apify_api_token: str) -> List[Dict]:
    """Apify를 통한 인스타그램 릴스 수집"""
    if not apify_api_token:
        st.error("❌ Apify API 토큰이 설정되지 않았습니다.")
        return []

    client = ApifyClient(apify_api_token)
    run_input = {
        "hashtags": [hashtag],
        "resultsLimit": max_count,
        "proxyConfiguration": {"use": "AUTO_POOL"}
    }

    st.info(f"🚀 해시태그 #{hashtag} 릴스 수집 중... (최대 {max_count}개)")
    
    try:
        run = client.actor(INSTAGRAM_SCRAPER_ACTOR_ID).call(
            run_input=run_input,
            timeout_secs=300
        )

        posts = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            caption = item.get("caption", "")
            if caption:
                date_str = item.get("timestamp", "")
                date_only = date_str.split("T")[0] if "T" in date_str else date_str.split(" ")[0]
                
                posts.append({
                    "caption": caption,
                    "date": date_only,
                    "shortcode": item.get("shortcode"),
                    "url": item.get("url"),
                    "hashtag": hashtag,
                    "likes": item.get("likesCount", 0),
                    "comments": item.get("commentsCount", 0),
                    "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
        st.success(f"✅ {len(posts)}개 릴스 수집 완료")
        return posts

    except Exception as e:
        st.error(f"❌ 릴스 수집 중 오류: {e}")
        return []


def save_posts_to_json(posts: List[Dict], filename: str):
    """JSON 파일 저장"""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(posts, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"파일 저장 실패: {e}")


def run_analysis_pipeline(hashtag: str, max_posts: int) -> List[Dict]:
    """전체 분석 파이프라인 실행"""
    with st.spinner("분석 파이프라인 실행 중..."):
        # 1. 데이터 수집
        posts = fetch_posts_from_apify(hashtag, max_posts, APIFY_API_TOKEN)
        if not posts:
            return []

        # 2. 데이터 저장
        save_posts_to_json(posts, f"instagram_{hashtag}_raw.json")

        # 3. 텍스트 정제
        st.info("🧹 텍스트 정제 중...")
        posts = clean_captions(posts)

        # 4. 감정 분석
        st.info("📊 감정 분석 중...")
        sentiment_model = load_sentiment_model()
        posts = analyze_sentiment(posts, sentiment_model)

        # 5. 키워드 추출
        st.info("🔑 키워드 추출 중...")
        keyword_model = load_keyword_model()
        posts = extract_keywords(posts, keyword_model)

        # 6. 최종 결과 저장
        save_posts_to_json(posts, f"instagram_{hashtag}_final.json")

        st.success("🎉 분석 완료!")
        return posts


def build_vectorstore(posts: List[Dict]) -> FAISS:
    """벡터 스토어 구축"""
    docs = [
        Document(page_content=p["cleaned_caption"]) 
        for p in posts 
        if p.get("cleaned_caption") and len(p["cleaned_caption"].strip()) > 0
    ]

    if not docs:
        st.warning("⚠️ 벡터 스토어 구축에 필요한 유효한 텍스트가 없습니다.")
        return None

    embeddings = UpstageEmbeddings(model="embedding-passage")
    return FAISS.from_documents(docs, embeddings)


def get_rag_answer(question: str, history: list, vectorstore: FAISS) -> str:
    """RAG 기반 질문 답변"""
    if not UPSTAGE_API_KEY:
        return "오류: Upstage API 키가 설정되지 않았습니다."
    
    if not vectorstore:
        return "분석된 릴스 데이터가 없습니다. 먼저 해시태그 분석을 실행해 주세요."

    # 컨텍스트 검색
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    context_docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in context_docs])

    # 대화 기록 구성
    chat_history = "\n".join([f"사용자: {q}\nAI: {a}" for q, a in history])

    # 프롬프트 생성
    prompt = ChatPromptTemplate.from_template("""
다음은 인스타그램 릴스 내용입니다. 이전 대화와 릴스 내용을 참고하여 질문에 답변해주세요.

[이전 대화]
{chat_history}

[릴스 내용]
{context}

[질문]
{question}

[답변]
""")
    
    # LLM 호출
    llm = ChatUpstage(model="solar-1-mini-chat", api_key=UPSTAGE_API_KEY)
    chain = prompt | llm
    
    response = chain.invoke({
        "question": question,
        "context": context,
        "chat_history": chat_history
    })
    return response.content


def get_hashtag_frequency(posts: List[Dict], top_n: int = 10) -> List[tuple]:
    """해시태그 빈도 계산"""
    all_hashtags = []
    for post in posts:
        caption = post.get("caption", "")
        hashtags = [word[1:] for word in caption.split() if word.startswith("#")]
        all_hashtags.extend(hashtags)
    
    return Counter(all_hashtags).most_common(top_n)


def render_analysis_results(posts: List[Dict]):
    """분석 결과 렌더링"""
    if not posts:
        return

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 감정 분석")
        plot_sentiment_distribution(posts, font_path=FONT_PATH)
        
        st.subheader("🔗 관련 해시태그")
        top_hashtags = get_hashtag_frequency(posts, top_n=10)
        if top_hashtags:
            for tag, count in top_hashtags:
                if st.button(f"#{tag} ({count})", key=f"tag_{tag}"):
                    st.session_state.current_hashtag = tag
                    st.rerun()
    
    with col2:
        st.subheader("☁️ 워드클라우드")
        if os.path.exists(FONT_PATH):
            try:
                generate_wordcloud(posts, font_path=FONT_PATH)
            except Exception as e:
                st.error(f"워드클라우드 생성 오류: {e}")
        else:
            st.error("폰트 파일을 찾을 수 없습니다.")
        
        st.subheader("🔥 인기 릴스")
        sorted_posts = sorted(posts, key=lambda x: x.get('likes', 0), reverse=True)
        for i, post in enumerate(sorted_posts[:3], 1):
            st.markdown(f"**{i}.** 👍 {post.get('likes', 0)}")
            st.markdown(f"{post.get('caption', '')[:100]}...")
            if post.get('url'):
                st.markdown(f"[보기]({post['url']})")
            st.markdown("---")


def render_search_section(posts: List[Dict]):
    """키워드 검색 섹션"""
    st.subheader("🔎 키워드 검색")
    
    if not posts:
        st.info("먼저 해시태그 분석을 실행해 주세요.")
        return
    
    search_term = st.text_input("검색할 키워드:", key="search_keyword")
    if search_term:
        results = search_by_keyword(posts, search_term)
        st.write(f"**'{search_term}'** 포함 릴스: **{len(results)}개**")
        
        for i, post in enumerate(results[:5], 1):
            st.markdown(f"**{i}.** {post.get('cleaned_caption', '')[:200]}...")


def render_qa_section(posts: List[Dict], vectorstore: FAISS):
    """질문답변 섹션"""
    st.subheader("🧠 질문 기반 분석")
    
    if not posts or not vectorstore:
        st.info("먼저 해시태그 분석을 실행해 주세요.")
        return
    
    question = st.text_input("질문을 입력하세요:", key="question_input")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💬 질문하기"):
            if question:
                with st.spinner("답변 생성 중..."):
                    answer = get_rag_answer(question, st.session_state.chat_history, vectorstore)
                    st.session_state.chat_history.append((question, answer))
    
    with col2:
        if st.button("🧹 대화 초기화"):
            st.session_state.chat_history = []
            st.success("대화 기록이 초기화되었습니다.")
    
    # 대화 기록 표시
    if st.session_state.chat_history:
        st.markdown("### 💬 대화 기록")
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
            with st.expander(f"Q{len(st.session_state.chat_history) - i + 1}: {q}"):
                st.markdown(f"**답변:** {a}")


def main():
    st.set_page_config(page_title="인스타그램 해시태그 분석기", layout="wide")
    st.title("📱 인스타그램 해시태그 분석기")

    # API 키 확인
    if not UPSTAGE_API_KEY or not APIFY_API_TOKEN:
        st.error("❌ API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        st.stop()

    # 세션 상태 초기화
    initialize_session_state()

    # 메인 입력 섹션
    st.header("🔍 해시태그 분석")
    
    col1, col2, col3 = st.columns([3, 2, 1])
    
    with col1:
        hashtag = st.text_input(
            "분석할 해시태그 입력!!!",
            value=st.session_state.current_hashtag,
            placeholder="예: 여행, ootd, 사랑 (기본값: 일상)"
        ).strip().replace("#", "")
        
    with col2:
        max_posts = st.slider(
            "수집할 릴스 수:", 
            min_value=20, 
            max_value=500, 
            value=st.session_state.current_max_posts
        )
        
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)  # 버튼 위치 조정
        analyze_button = st.button("🚀 분석 시작", type="primary")

    # 세션 상태 업데이트
    st.session_state.current_hashtag = hashtag
    st.session_state.current_max_posts = max_posts

    # 분석 실행
    if analyze_button:
        hashtag_to_analyze = hashtag if hashtag else "일상"
        posts = run_analysis_pipeline(hashtag_to_analyze, max_posts)
        
        if posts:
            st.session_state.analyzed_posts = posts
            with st.spinner("벡터 스토어 구축 중..."):
                vectorstore = build_vectorstore(posts)
                st.session_state.vectorstore = vectorstore

    # 결과 표시 - 좌측 75% + 우측 25% 레이아웃
    posts = st.session_state.analyzed_posts
    vectorstore = st.session_state.vectorstore

    if posts:
        st.markdown("---")
        
        # 메인 레이아웃: 좌측 75% (분석 결과) + 우측 25% (검색/질문답변)
        main_col, sidebar_col = st.columns([3, 1])
        
        with main_col:
            # 분석 결과 (2열 레이아웃)
            render_analysis_results(posts)
            
        with sidebar_col:
            # 우측 사이드바: 검색 & 질문답변 (세로 배치)
            render_search_section(posts)
            st.markdown("---")
            render_qa_section(posts, vectorstore)
            
    else:
        # 메인 레이아웃 유지 (분석 전에도 우측 영역 표시)
        main_col, sidebar_col = st.columns([3, 1])
        
        with main_col:
            st.info("해시태그를 입력하고 '분석 시작' 버튼을 클릭하여 분석을 시작하세요.")
            
        with sidebar_col:
            st.info("분석 후 검색과 질문답변 기능을 이용하세요.")


if __name__ == "__main__":
    # 모듈 경로 설정
    current_script_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_script_path)
    project_root_dir = os.path.join(current_dir, "..")
    sys.path.insert(0, project_root_dir)
    
    main()