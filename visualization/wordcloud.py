# visualization/wordcloud.py

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re # 정규 표현식 모듈 임포트
from typing import List, Dict

# Streamlit 사용 시 경고 제거를 위해 st 임포트 (필요 시)
import streamlit as st
import os # os 모듈 임포트

def generate_wordcloud(posts: List[Dict], font_path: str = None):
    """
    cleaned_caption에서 키워드를 추출하여 워드클라우드를 생성합니다.
    """
    if not posts:
        st.warning("⚠️ 워드클라우드를 생성할 릴스 데이터가 없습니다.")
        return

    all_keywords = []
    for post in posts:
        if 'keywords' in post and isinstance(post['keywords'], list):
            all_keywords.extend(post['keywords'])

    if not all_keywords:
        st.info("키워드가 추출되지 않아 워드클라우드를 생성할 수 없습니다. 텍스트 정제 및 키워드 추출 단계를 확인해주세요.")
        return

    # 키워드 빈도 계산
    freq = Counter(all_keywords)

    # 폰트 경로 유효성 검사 및 설정
    if font_path and os.path.exists(font_path):
        # WordCloud 객체 생성 시 폰트 경로 지정
        wc = WordCloud(
            font_path=font_path,
            background_color="white",
            width=800,
            height=600,
            max_words=100,
            max_font_size=200,
            random_state=42,
            collocations=False # '굿'과 '굿굿'을 별개로 처리하기 위함
        )
    else:
        st.warning(f"⚠️ 폰트 파일을 찾을 수 없거나 경로가 유효하지 않아 워드클라우드에 기본 폰트를 사용합니다: {font_path}")
        st.info("워드클라우드의 한글이 깨져 보일 수 있습니다. 'fonts/NotoSansKR-Regular.ttf' 파일의 존재와 경로를 확인해주세요.")
        # 폰트 경로가 유효하지 않은 경우, WordCloud에 font_path를 지정하지 않음
        wc = WordCloud(
            background_color="white",
            width=600,
            height=480,
            max_words=100,
            max_font_size=100,
            random_state=42,
            collocations=False
        )


    wc.generate_from_frequencies(freq)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('릴스 키워드 워드클라우드')

    st.pyplot(fig) # Streamlit에 그래프 표시
    plt.close(fig) # 메모리 해제