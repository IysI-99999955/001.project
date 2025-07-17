# visualization/charts.py

import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict


def plot_sentiment_distribution(posts: List[Dict]) -> None:
    """
    감정 분석 결과를 막대 그래프로 시각화
    :param posts: 게시글 리스트 (sentiment 필드 포함)
    """
    sentiments = [post.get("sentiment", "Unknown") for post in posts]
    counts = Counter(sentiments)

    labels = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=["#66bb6a", "#ffee58", "#ef5350"])
    plt.title("감정 분석 결과 분포")
    plt.xlabel("감정")
    plt.ylabel("게시글 수")

    # 값 표시
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, int(yval), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
