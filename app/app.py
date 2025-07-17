# app/app.py

import sys
import os
from datetime import datetime, timedelta # datetime, timedelta 임포트 추가

# --- 모듈 검색 경로 설정 시작 ---
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
project_root_dir = os.path.join(current_dir, "..")
sys.path.insert(0, project_root_dir)
# --- 모듈 검색 경로 설정 끝 ---

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

# .env 파일에서 환경 변수 로드
load_dotenv()
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")

# 전역 설정 변수
DATA_DIR = "../data"

# 폰트 경로 설정: 프로젝트 루트의 'fonts' 폴더 내 Noto Sans KR 폰트 파일 참조
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
project_root_dir = os.path.join(current_dir, "..") 

FONT_FILENAME = "NotoSansKR-Regular.ttf" 
FONT_PATH = os.path.join(project_root_dir, "fonts", FONT_FILENAME)

# Apify Instagram Hashtag Scraper Actor ID
# 이 ID를 사용하려는 Instagram Hashtag Scraper Actor의 정확한 ID로 교체해야함..
INSTAGRAM_SCRAPER_ACTOR_ID = "apify/instagram-hashtag-scraper" 


def build_vectorstore_from_posts(posts: list) -> FAISS:
    """
    릴스 리스트에서 'cleaned_caption'을 사용하여 FAISS 벡터 스토어를 구축합니다.
    """
    docs = [
        Document(page_content=p["cleaned_caption"]) 
        for p in posts 
        if "cleaned_caption" in p and p["cleaned_caption"] and isinstance(p["cleaned_caption"], str) and len(p["cleaned_caption"].strip()) > 0
    ]

    if not docs:
        st.warning("⚠️ 벡터 스토어를 구축할 유효한 텍스트가 없습니다. 수집된 릴스의 캡션이 비어있거나 유효하지 않을 수 있습니다.")
        return None 

    embeddings = UpstageEmbeddings(model="embedding-passage") 
    return FAISS.from_documents(docs, embeddings)


def solar_rag_answer_multi(question: str, history: list, vectorstore: FAISS, k: int = 5,
                           model_name: str = "solar-1-mini-chat", api_key: str = None) -> str:
    """
    Solar API와 RAG, 멀티턴 대화를 사용하여 질문에 답변합니다.
    """
    if not api_key:
        return "오류: Upstage API 키가 설정되지 않았습니다. .env 파일을 확인해주세요."
    
    if vectorstore is None:
        return "죄송합니다. 분석된 릴스 내용이 없어 답변하기 어렵습니다. 먼저 '해시태그 분석' 탭에서 분석을 실행해 주세요."


    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    context_docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in context_docs])

    chat_history = "\n".join([f"사용자: {q}\nAI: {a}" for q, a in history])

    prompt = ChatPromptTemplate.from_template("""
다음은 인스타그램에서 수집한 릴스 내용입니다. 이전 대화와 릴스를 참고하여 사용자 질문에 성실하게 응답해주세요.
릴스 내용과 직접적인 관련이 없거나, 답변하기 어려운 질문에는 "죄송합니다. 현재 릴스 내용으로는 답변하기 어렵습니다."라고 답변해주세요.

[이전 대화]
{chat_history}

[릴스 내용]
{context}

[현재 질문]
{question}

[AI의 답변]
""")
    
    llm = ChatUpstage(model=model_name, api_key=api_key)
    chain = prompt | llm
    
    response = chain.invoke({
        "question": question,
        "context": context,
        "chat_history": chat_history
    })
    return response.content


def fetch_posts_from_apify(
    hashtag: str,
    max_count: int,
    apify_api_token: str
) -> List[Dict]:
    """
    Apify Instagram Hashtag Scraper Actor를 사용하여 해시태그 릴스를 수집합니다.
    """
    if not apify_api_token:
        st.error("❌ Apify API 토큰이 설정되지 않았습니다. .env 파일을 확인해주세요.")
        return []

    client = ApifyClient(apify_api_token)
    actor_id = INSTAGRAM_SCRAPER_ACTOR_ID # Instagram Hashtag Scraper Actor ID 사용

    run_input = {
        "hashtags": [hashtag],
        "resultsLimit": max_count,
        "proxyConfiguration": { "use": "AUTO_POOL" },
        "extendOutputFunction": """
            async ({ data, item, page, request, customData, basicCrawler, Apify }) => {
                return item;
            }
        """,
        "extendOutputFunctionVars": {},
    }

    st.info(f"🚀 Apify 인스타그램 해시태그 스크래퍼 실행 중... 해시태그: #{hashtag}, 최대 {max_count}개 릴스")
    st.info(f"Apify 콘솔에서 진행 상황을 확인할 수 있습니다: https://console.apify.com/actors/{actor_id}")

    try:
        run = client.actor(actor_id).call(
            run_input=run_input,
            timeout_secs=300 # 5분 타임아웃
        )

        apify_posts = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            caption = item.get("caption", "")
            
            date_str = item.get("timestamp")
            if date_str:
                date_only = date_str.split("T")[0] if "T" in date_str else date_str.split(" ")[0]
            else:
                date_only = None

            if caption: # 캡션이 비어있지 않은 경우에만 추가
                apify_posts.append({
                    "caption": caption,
                    "date": date_only,
                    "shortcode": item.get("shortcode"),
                    "url": item.get("url"),
                    "hashtag": hashtag, # 입력 해시태그 저장
                    "likes": item.get("likesCount", 0),
                    "comments": item.get("commentsCount", 0),
                    "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
        st.success(f"✅ Apify에서 {len(apify_posts)}개 릴스 수집 완료.")
        return apify_posts

    except Exception as e:
        st.error(f"❌ Apify 스크래핑 중 오류 발생: {e}")
        st.warning("Apify API 토큰, Actor ID, 또는 크레딧 잔액을 확인해주세요. 네트워크 문제일 수도 있습니다.")
        return []

def _save_posts_to_json(posts: List[Dict], save_path: str) -> None:
    """JSON 파일 저장 (임시 함수, src/utils.py 등으로 분리 권장)"""
    import json 
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(posts, f, ensure_ascii=False, indent=2)
        print(f"✅ 파일 저장 완료: {save_path}")
    except Exception as e:
        print(f"❌ 파일 저장 실패: {e}")

def filter_recent_posts(posts: List[Dict], days: int = 1) -> List[Dict]:
    """
    릴스 리스트에서 최근 N일 이내의 릴스만 필터링합니다.
    :param posts: 릴스 리스트 (각 릴스에 'date' 필드 필요, YYYY-MM-DD 형식)
    :param days: 최근 N일 (기본값 1일)
    :return: 필터링된 릴스 리스트
    """
    recent_posts = []
    time_threshold = datetime.now() - timedelta(days=days)
    for post in posts:
        post_date_str = post.get("date")
        if post_date_str:
            try:
                # 날짜 문자열을 datetime 객체로 변환
                post_date = datetime.strptime(post_date_str, "%Y-%m-%d")
                if post_date >= time_threshold:
                    recent_posts.append(post)
            except ValueError:
                # 날짜 형식이 다르거나 유효하지 않은 경우 건너뜁니다.
                continue
    return recent_posts


def run_pipeline(hashtag: str, max_posts: int):
    """
    인스타그램 릴스 수집부터 키워드 추출까지의 전체 파이프라인을 실행합니다.
    """
    progress_text = st.empty()
    progress_bar = st.progress(0)

    progress_text.text(f"📥 인스타그램 릴스 수집 중... (Apify Actor 실행)")
    
    try:
        posts = fetch_posts_from_apify(hashtag, max_posts, APIFY_API_TOKEN)
        
        # 최근 1일 이내 릴스 필터링 (이 줄을 주석 처리하거나 삭제하여 필터링을 건너뛸 수 있습니다.)
        initial_collected_count = len(posts)
        # posts = filter_recent_posts(posts, days=1) # <-- 이 줄을 주석 처리하거나 삭제합니다.
        if initial_collected_count > 0: # 초기 수집 릴스이 있을 경우에만 메시지 표시
            st.info(f"⏳ 최근 1일 이내 릴스 {len(posts)}개 필터링 완료 (총 {initial_collected_count}개 중).")

        save_path = os.path.join(DATA_DIR, f"instagram_{hashtag}_raw.json")
        _save_posts_to_json(posts, save_path) 

        progress_bar.progress(100)
        progress_text.text(f"✅ 릴스 {len(posts)}개 수집 완료")
        st.success(f"✅ 릴스 {len(posts)}개 수집 완료")

    except Exception as e:
        st.error(f"❌ 릴스 수집 중 오류 발생: {e}")
        st.warning("Apify 스크래핑 중 문제가 발생했습니다. Apify 콘솔에서 Actor 실행 로그를 확인해주세요.")
        progress_text.empty()
        progress_bar.empty()
        return []

    if not posts:
        st.warning("수집된 릴스이 없습니다. 해시태그나 수집 설정을 확인해주세요.")
        return []

    st.info("🧹 텍스트 정제 중...")
    posts = clean_captions(posts)

    st.info("📊 감정 분석 중...")
    sentiment_model = load_sentiment_model()
    posts = analyze_sentiment(posts, sentiment_model)

    st.info("🔑 키워드 추출 중...")
    keyword_model = load_keyword_model()
    posts = extract_keywords(posts, keyword_model)

    output_path = os.path.join(DATA_DIR, f"instagram_{hashtag}_final.json")
    with open(output_path, "w", encoding="utf-8") as f:
        import json 
        json.dump(posts, f, ensure_ascii=False, indent=2)

    st.success("🎉 파이프라인 완료!")
    return posts

def get_hashtag_frequency(posts: List[Dict], top_n: int = 10) -> List[tuple]:
    """
    수집된 릴스 내에서 가장 많이 언급된 해시태그를 추출하고 빈도를 계산합니다.
    (주의: 현재는 캡션에서 #으로 시작하는 단어를 간단히 파싱합니다.
    src/cleaner.py에서 캡션 정제 시 더 정교하게 해시태그를 추출하여
    post['hashtags_in_caption'] 필드에 저장하는 것을 권장합니다.)
    """
    all_hashtags = []
    for post in posts:
        caption = post.get("caption", "")
        found_hashtags = [word[1:] for word in caption.split() if word.startswith("#")]
        all_hashtags.extend(found_hashtags)
    
    hashtag_counts = Counter(all_hashtags)
    return hashtag_counts.most_common(top_n)


def main():
    st.set_page_config(page_title="인스타그램 해시태그 분석기", layout="wide")
    st.title("📸 인스타그램 해시태그 분석기 (공개 데이터 기반)")

    if not UPSTAGE_API_KEY:
        st.error("❌ UPSTAGE_API_KEY 환경 변수가 설정되지 않았습니다. `.env` 파일을 확인하거나, API 키를 설정해주세요.")
        st.stop()
    if not APIFY_API_TOKEN:
        st.error("❌ APIFY_API_TOKEN 환경 변수가 설정되지 않았습니다. `.env` 파일을 확인하거나, API 키를 설정해주세요.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["🔍 해시태그 분석", "🔎 키워드 검색", "🧠 질문 기반 요약 (멀티턴)"])

    with tab1:
        st.header("해시태그 분석 파이프라인")
        
        initial_hashtag_value = ""
        if "suggested_hashtag" in st.session_state:
            initial_hashtag_value = st.session_state.suggested_hashtag
            del st.session_state.suggested_hashtag

        hashtag = st.text_input(
            "분석할 해시태그를 입력하세요 (예: 여행, ootd). 입력하지 않으면 '일상'으로 자동 분석됩니다:",
            value=initial_hashtag_value,
            key="main_hashtag_input"
        ).strip().replace("#", "")
        
        # 해시태그 입력이 없을 경우 '일상'으로 자동 설정
        if not hashtag:
            hashtag_to_analyze = "일상"
            st.info(f"해시태그가 입력되지 않아 기본 해시태그 '#{hashtag_to_analyze}'으로 분석을 시작합니다.")
        else:
            hashtag_to_analyze = hashtag

        max_posts = st.slider("수집할 릴스 수", min_value=20, max_value=500, value=50, step=1)

        if st.button("분석 시작"):
            if hashtag_to_analyze:
                posts = run_pipeline(hashtag_to_analyze, max_posts)
                
                if posts:
                    st.session_state["analyzed_posts"] = posts
                    
                    with st.spinner("📚 벡터 스토어 구축 중..."):
                        vectorstore_result = build_vectorstore_from_posts(posts)
                        if vectorstore_result:
                            st.session_state["vectorstore"] = vectorstore_result
                            st.success("✅ 벡터 스토어 구축 완료!")
                        else:
                            st.warning("⚠️ 벡터 스토어를 구축할 유효한 텍스트가 없어 RAG 기능을 사용할 수 없습니다.")
                            st.session_state["vectorstore"] = None 

                    st.subheader("📈 감정 분석 결과")
                    # charts.py에 font_path 전달
                    plot_sentiment_distribution(posts, font_path=FONT_PATH)

                    st.subheader("☁️ 워드클라우드")
                    
                    # 폰트 파일 존재 여부 확인 및 경고 메시지 추가
                    # st.info(f"워드클라우드 폰트 경로 확인 중: '{FONT_PATH}'") # 폰트 경로 출력
                    st.info(f"워드클라우드 폰트 경로 확인 중") # 폰트 경로 출력 삭제
                    if not os.path.exists(FONT_PATH):
                        st.error(f"❌ 폰트 파일을 찾을 수 없습니다: '{FONT_PATH}'")
                        st.warning("워드클라우드를 생성하려면 프로젝트 루트 폴더 아래 'fonts' 폴더에 'NotoSansKR-Regular.ttf' 파일을 넣어주세요. 파일명과 경로, 그리고 파일 읽기 권한을 확인해주세요.")
                    else:
                        try:
                            generate_wordcloud(posts, font_path=FONT_PATH)
                            st.success("✅ 워드클라우드 생성 완료!")
                        except OSError as e:
                            st.error(f"❌ 워드클라우드 폰트 로드 중 오류 발생: {e}")
                            st.warning(f"폰트 파일 '{FONT_PATH}'에 접근할 수 없거나 파일이 손상되었을 수 있습니다. 파일의 읽기 권한을 확인하거나, 유효한 폰트 파일로 교체해주세요.")
                        except Exception as e:
                            st.error(f"❌ 워드클라우드 생성 중 예상치 못한 오류 발생: {e}")


                    st.subheader("🔗 관련 해시태그 제안")
                    top_hashtags = get_hashtag_frequency(posts, top_n=15)
                    if top_hashtags:
                        st.write("수집된 릴스에서 가장 많이 언급된 해시태그들입니다:")
                        cols = st.columns(5)
                        for i, (tag, count) in enumerate(top_hashtags):
                            with cols[i % 5]:
                                if st.button(f"#{tag} ({count})", key=f"suggested_tag_{tag}"):
                                    st.session_state.suggested_hashtag = tag
                                    st.experimental_rerun()
                    else:
                        st.info("수집된 릴스 내에서 다른 해시태그를 찾을 수 없습니다.")

                    st.subheader("🔥 최근 인기 릴스")
                    sorted_posts = sorted(posts, key=lambda x: x.get('likes', 0), reverse=True)
                    if sorted_posts:
                        st.write("최근 1일 이내 수집된 릴스 중 좋아요가 많은 릴스입니다:")
                        for i, post in enumerate(sorted_posts[:5]):
                            st.markdown(f"**{i+1}.** **좋아요: {post.get('likes', 0)}**")
                            st.markdown(f"   **캡션:** {post.get('caption', '')[:150]}...")
                            st.markdown(f"   [릴스 보기]({post.get('url', '#')})")
                            st.markdown("---")
                    else:
                        st.info("최근 1일 이내의 인기 릴스를 찾을 수 없습니다.")

                else:
                    st.warning("분석을 위한 릴스이 충분히 수집되지 않았습니다.")
            else:
                st.warning("해시태그를 입력해주세요.")


    with tab2:
        st.header("키워드 검색")
        posts = st.session_state.get("analyzed_posts", [])

        if posts:
            st.info("좌측 '해시태그 분석' 탭에서 분석된 데이터를 기반으로 검색합니다.")
            search_term = st.text_input("검색할 키워드 입력:")
            if search_term:
                results = search_by_keyword(posts, search_term)
                st.write(f"'{search_term}' 키워드 포함 릴스: {len(results)}개")
                for i, post in enumerate(results[:5], 1):
                    st.markdown(f"**{i}.** {post['cleaned_caption']}")
            elif search_term == "":
                st.info("검색어를 입력해주세요.")
        else:
            st.info("먼저 '해시태그 분석' 탭에서 분석을 실행해 주세요.")

    with tab3:
        st.header("질문 기반 요약 (멀티턴)")
        posts = st.session_state.get("analyzed_posts", [])
        vectorstore = st.session_state.get("vectorstore", None)

        if posts and vectorstore:
            st.info("좌측 '해시태그 분석' 탭에서 분석된 데이터를 기반으로 질문에 답변합니다.")
            
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            question = st.text_input("🗣 질문을 입력하세요", key="multi_q")

            if st.button("💬 질문 보내기"):
                if question:
                    with st.spinner("Solar 응답 생성 중..."):
                        answer = solar_rag_answer_multi(
                            question=question,
                            history=st.session_state.chat_history,
                            vectorstore=vectorstore,
                            api_key=UPSTAGE_API_KEY
                        )
                        st.session_state.chat_history.append((question, answer))
                else:
                    st.warning("질문을 입력해주세요.")

            if st.button("🧹 대화 이력 초기화"):
                st.session_state.chat_history = []
                st.success("대화 기록이 초기화되었습니다.")

            st.markdown("---")
            st.markdown("## 💬 대화 이력")
            if st.session_state.chat_history:
                for idx, (q, a) in enumerate(st.session_state.chat_history[::-1], 1):
                    with st.expander(f"Q{len(st.session_state.chat_history) - idx + 1}: {q}"):
                        st.markdown(f"**AI:** {a}")
            else:
                st.info("아직 대화 이력이 없습니다. 질문을 시작해 보세요!")
        else:
            st.info("먼저 '해시태그 분석' 탭에서 분석을 실행해 주세요.")


if __name__ == "__main__":
    # --- 모듈 검색 경로 설정 시작 ---
    current_script_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_script_path)
    project_root_dir = os.path.join(current_dir, "..")
    sys.path.insert(0, project_root_dir)
    # --- 모듈 검색 경로 설정 끝 ---
    main()