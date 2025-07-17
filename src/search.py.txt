# src/search.py

from typing import List, Dict


def search_by_keyword(posts: List[Dict], keyword: str) -> List[Dict]:
    """
    cleaned_caption 또는 keywords에 특정 단어가 포함된 게시글을 반환
    :param posts: 게시글 리스트 (keywords, cleaned_caption 포함)
    :param keyword: 검색할 키워드
    :return: 매칭되는 게시글 리스트
    """
    keyword = keyword.lower()
    result = []

    for post in posts:
        caption = post.get("cleaned_caption", "").lower()
        keywords = [k.lower() for k in post.get("keywords", [])]

        if keyword in caption or keyword in keywords:
            result.append(post)

    return result


if __name__ == "__main__":
    import json

    keyword = input("검색할 키워드를 입력하세요: ").strip()

    with open("../data/여행_posts_keywords.json", "r", encoding="utf-8") as f:
        posts = json.load(f)

    matches = search_by_keyword(posts, keyword)

    print(f"\n🔍 '{keyword}'가 포함된 게시글 수: {len(matches)}개\n")
    for i, post in enumerate(matches[:5], 1):
        print(f"{i}. [{post['date']}] {post['cleaned_caption'][:80]}...")
