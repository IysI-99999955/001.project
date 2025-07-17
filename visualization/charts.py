# visualization/charts.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import streamlit as st
from typing import List, Dict

# matplotlib 폰트 설정
# app.py에서 FONT_PATH를 넘겨받아 사용하도록 수정
def plot_sentiment_distribution(posts: List[Dict], font_path: str = None):
    """
    릴스 감정 분석 결과의 분포를 시각화합니다!
    """
    if not posts:
        st.warning("⚠️ 감정 분석을 위한 릴스 데이터가 없습니다.")
        return

    # 한글 폰트 설정
    if font_path and os.path.exists(font_path):
        plt.rcParams['font.family'] = 'Noto Sans KR'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지
        # 폰트 매니저에 폰트 추가
        from matplotlib import font_manager, rc
        font_manager.fontManager.addfont(font_path)
        rc('font', family='Noto Sans KR')
    else:
        st.warning(f"⚠️ 폰트 파일을 찾을 수 없거나 경로가 유효하지 않아 기본 폰트를 사용합니다: {font_path}")
        st.info("워드클라우드 및 차트의 한글이 깨져 보일 수 있습니다. 'fonts/NotoSansKR-Regular.ttf' 파일의 존재와 경로를 확인해주세요.")

    sentiments = [p.get('sentiment') for p in posts if p.get('sentiment') in ['긍정', '중립', '부정']]
    if not sentiments:
        st.info("감정 분석 데이터가 충분하지 않습니다.")
        return

    sentiment_counts = pd.Series(sentiments).value_counts().reindex(['긍정', '중립', '부정'])
    sentiment_counts = sentiment_counts.fillna(0) # 없는 감정은 0으로 채움

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette='viridis')
    ax.set_title('릴스 감정 분포')
    ax.set_xlabel('감정')
    ax.set_ylabel('릴스 수')
    
    # 각 막대 위에 값 표시
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig) # Streamlit에 그래프 표시
    plt.close(fig) # 메모리 해제