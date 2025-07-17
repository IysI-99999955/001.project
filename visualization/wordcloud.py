# visualization/wordcloud.py

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict


def generate_wordcloud(posts: List[Dict], font_path: str = None) -> None:
    """
    keywords 필드가 있는 게시글로부터 워드클라우드를 생성
    :param posts: 게시글 리스트 (keywords 포함)
    :param font_path: 한글 폰트 경로 (예: "/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
    """
    all_keywords = []
    for post in posts:
        all_keywords.extend(post.get("keywords", []))

    freq = Counter(all_keywords)

    wc = WordCloud(
        font_path=font_path,
        width=800,
        height=400,
        background_color="white"
    ).generate_from_frequencies(freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
