# app/app.py

import streamlit as st
import json
import os
from dotenv import load_dotenv # dotenv 라이브러리 임포트

# src 및 visualization 모듈 임포트
from src.scraper import fetch_hashtag_posts, save_posts_to_json
from src.cleaner import clean_captions
from src.sentiment import load_sentiment_model, analyze_sentiment
from src.keywords import load_keyword_model, extract_keywords, get_keyword_frequency # get_keyword_frequency는 현재 사용되지 않지만, 기존 코드에 있었으므로 유지
from src.search import search_by_keyword
from visualization.charts import plot_sentiment_distribution
from visualization.wordcloud import generate_wordcloud

# Langchain 및 Upstage 관련 모듈 임포트
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate

# .env 파일에서 환경 변수 로드
load_dotenv()
API_KEY = os.getenv("UPSTAGE_API_KEY") # Upstage API 키 로드

# 전역 설정 변수
DATA_DIR = "../data"
FONT_PATH = "c:/Windows/Fonts/malgun.ttf" # 한글 지원 폰트 (윈도우 기준)


def build_vectorstore_from_posts(posts: list) -> FAISS:
    """
    게시글 리스트에서 cleaned_caption을 사용하여 FAISS 벡터 스토어를 구축합니다.
    """
    # cleaned_caption이 있는 게시글만 Document로 변환
    docs = [Document(page_content=p["cleaned_caption"]) for p in posts if "cleaned_caption" in p]
    embeddings = UpstageEmbeddings()
    return FAISS.from_documents(docs, embeddings)


def solar_rag_answer_multi(question: str, history: list, vectorstore: FAISS, k: int = 5,
                           model_name: str = "solar-1-mini-chat", api_key: str = None) -> str:
    """
    Solar API와 RAG, 멀티턴 대화를 사용하여 질문에 답변합니다.
    :param question: 현재 사용자 질문
    :param history: 이전 대화 이력 (질문, 답변 튜플 리스트)
    :param vectorstore: 구축된 FAISS 벡터 스토어
    :param k: 검색할 관련 문서의 수
    :param model_name: 사용할 Solar 모델 이름
    :param api_key: Upstage API 키
    :return: Solar 모델의 답변
    """
    if not api_key:
        return "오류: Upstage API 키가 설정되지 않았습니다. .env 파일을 확인해주세요."

    # 벡터 스토어에서 관련 문서 검색
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    context_docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in context_docs])

    # 이전 대화 이력을 프롬프트에 포함하기 위해 포맷팅
    chat_history = "\n".join([f"사용자: {q}\nAI: {a}" for q, a in history])

    # ChatPromptTemplate을 사용하여 프롬프트 정의
    prompt = ChatPromptTemplate.from_template("""
다음은 인스타그램에서 수집한 게시글 내용입니다. 이전 대화와 게시글을 참고하여 사용자 질문에 성실하게 응답해주세요.
게시글 내용과 직접적인 관련이 없거나, 답변하기 어려운 질문에는 "죄송합니다. 현재 게시글 내용으로는 답변하기 어렵습니다."라고 답변해주세요.

[이전 대화]
{chat_history}

[게시글 내용]
{context}

[현재 질문]
{question}

[AI의 답변]
""")
    
    # Solar 모델 초기화
    llm = ChatUpstage(model=model_name, api_key=api_key)
    
    # 프롬프트와 LLM을 연결하는 체인 생성
    chain = prompt | llm
    
    # 체인 실행 및 응답 반환
    response = chain.invoke({
        "question": question,
        "context": context,
        "chat_history": chat_history
    })
    return response.content


def run_pipeline(hashtag, max_posts):
    """
    인스타그램 게시글 수집부터 키워드 추출까지의 전체 파이프라인을 실행합니다.
    """
    st.info("📥 게시글 수집 중...")
    try:
        posts = fetch_hashtag_posts(hashtag, max_count=max_posts)
        save_path = os.path.join(DATA_DIR, f"{hashtag}_raw.json")
        save_posts_to_json(posts, save_path)
        st.success(f"✅ 게시글 {len(posts)}개 수집 완료")
    except Exception as e:
        st.error(f"❌ 게시글 수집 중 오류 발생: {e}")
        st.warning("Instaloader는 비공식 API를 사용하므로, IP 차단이나 캡챠 등의 문제가 발생할 수 있습니다. 잠시 후 다시 시도하거나, 딜레이 설정을 확인해주세요.")
        return [] # 오류 발생 시 빈 리스트 반환

    if not posts: # 수집된 게시글이 없으면 파이프라인 중단
        st.warning("수집된 게시글이 없습니다. 해시태그나 수집 설정을 확인해주세요.")
        return []

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

    # API 키 유효성 검사
    if not API_KEY:
        st.error("❌ UPSTAGE_API_KEY 환경 변수가 설정되지 않았습니다. `.env` 파일을 확인하거나, API 키를 설정해주세요.")
        st.stop() # API 키가 없으면 앱 실행 중단

    # 탭 메뉴 구성
    tab1, tab2, tab3 = st.tabs(["🔍 해시태그 분석", "🔎 키워드 검색", "🧠 질문 기반 요약 (멀티턴)"])

    with tab1:
        st.header("해시태그 분석 파이프라인")
        hashtag = st.text_input("분석할 해시태그를 입력하세요 (예: 여행, ootd):").strip().replace("#", "")
        max_posts = st.slider("수집할 게시글 수", min_value=20, max_value=500, value=100, step=10)

        if st.button("분석 시작"):
            if hashtag:
                # 파이프라인 실행 및 결과 저장
                posts = run_pipeline(hashtag, max_posts)
                
                if posts: # 게시글이 성공적으로 수집 및 분석되었을 경우
                    st.session_state["analyzed_posts"] = posts # 세션 상태에 저장
                    
                    # 벡터 스토어 구축 및 세션 상태에 저장 (RAG 탭에서 재사용)
                    with st.spinner("📚 벡터 스토어 구축 중..."):
                        st.session_state["vectorstore"] = build_vectorstore_from_posts(posts)
                    st.success("✅ 벡터 스토어 구축 완료!")

                    st.subheader("📈 감정 분석 결과")
                    plot_sentiment_distribution(posts)

                    st.subheader("☁️ 워드클라우드")
                    generate_wordcloud(posts, font_path=FONT_PATH)
                else:
                    st.warning("분석을 위한 게시글이 충분히 수집되지 않았습니다.")
            else:
                st.warning("해시태그를 입력해주세요.")

    with tab2:
        st.header("키워드 검색")
        # 'analyzed_posts'가 세션 상태에 있는지 확인
        posts = st.session_state.get("analyzed_posts", [])

        if posts:
            st.info("좌측 '해시태그 분석' 탭에서 분석된 데이터를 기반으로 검색합니다.")
            search_term = st.text_input("검색할 키워드 입력:")
            if search_term:
                results = search_by_keyword(posts, search_term)
                st.write(f"'{search_term}' 키워드 포함 게시글: {len(results)}개")
                # 검색 결과는 5개까지만 표시
                for i, post in enumerate(results[:5], 1):
                    st.markdown(f"**{i}.** {post['cleaned_caption']}")
            elif search_term == "": # 입력창이 비어있을 때
                st.info("검색어를 입력해주세요.")
        else:
            st.info("먼저 '해시태그 분석' 탭에서 게시글을 수집하고 분석해 주세요.")

    with tab3:
        st.header("질문 기반 요약 (멀티턴)")
        # 'analyzed_posts'와 'vectorstore'가 세션 상태에 있는지 확인
        posts = st.session_state.get("analyzed_posts", [])
        vectorstore = st.session_state.get("vectorstore", None)

        if posts and vectorstore:
            st.info("좌측 '해시태그 분석' 탭에서 분석된 데이터를 기반으로 질문에 답변합니다.")
            
            # 대화 이력 초기화
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # 사용자 질문 입력
            question = st.text_input("🗣 게시글 내용에 대해 궁금한 점을 질문하세요:", key="multi_q")

            # 질문 보내기 버튼
            if st.button("💬 질문 보내기"):
                if question:
                    with st.spinner("Solar 응답 생성 중..."):
                        answer = solar_rag_answer_multi(
                            question=question,
                            history=st.session_state.chat_history,
                            vectorstore=vectorstore, # 구축된 벡터 스토어 전달
                            api_key=API_KEY
                        )
                        st.session_state.chat_history.append((question, answer))
                else:
                    st.warning("질문을 입력해주세요.")

            # 대화 이력 초기화 버튼
            if st.button("🧹 대화 이력 초기화"):
                st.session_state.chat_history = []
                st.success("대화 기록이 초기화되었습니다.")

            st.markdown("---")
            st.markdown("## 💬 대화 이력")
            # 최신 대화가 위에 오도록 역순으로 출력
            if st.session_state.chat_history:
                for idx, (q, a) in enumerate(st.session_state.chat_history[::-1], 1):
                    with st.expander(f"Q{len(st.session_state.chat_history) - idx + 1}: {q}"): # 질문 번호 조정
                        st.markdown(f"**AI:** {a}")
            else:
                st.info("아직 대화 이력이 없습니다. 질문을 시작해 보세요!")
        else:
            st.info("먼저 '해시태그 분석' 탭에서 게시글을 수집하고 분석해 주세요. (벡터 스토어 구축 필요)")


if __name__ == "__main__":
    main()
