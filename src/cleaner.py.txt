# src/cleaner.py

import re
import emoji
import json
import os
from typing import List, Dict, Optional


def clean_caption(text: str) -> str:
    """
    개별 캡션을 정제하여 분석 가능한 형태로 변환합니다.
    - 해시태그, 멘션, URL, 이모지, 특수문자 제거
    - 한국어 텍스트 정제 최적화
    :param text: 원본 캡션
    :return: 정제된 문자열
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 원본 텍스트 보존
    original_text = text
    
    try:
        # 1. 해시태그 제거 (한국어 해시태그 포함)
        text = re.sub(r"#\S+", "", text)
        
        # 2. 멘션 제거
        text = re.sub(r"@\S+", "", text)
        
        # 3. URL 제거 (다양한 형태)
        text = re.sub(r"http[s]?://\S+|www\.\S+", "", text)
        text = re.sub(r"\S+\.(com|kr|net|org|co\.kr)\S*", "", text)
        
        # 4. 이모지 제거
        text = emoji.replace_emoji(text, replace='')
        
        # 5. 특수 문자 정제 (한국어 텍스트 고려)
        # 기본 문장부호는 유지하되 반복되는 특수문자 제거
        text = re.sub(r"[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ.,!?~\-]", "", text)
        
        # 6. 반복 문자 정제 (ㅋㅋㅋ, ㅎㅎㅎ, !!!!, ???? 등)
        text = re.sub(r"([ㅋㅎ!?~.])\1{2,}", r"\1\1", text)
        
        # 7. 연속된 공백 제거
        text = re.sub(r"\s+", " ", text)
        
        # 8. 앞뒤 공백 제거
        text = text.strip()
        
        # 9. 너무 짧은 텍스트 처리
        if len(text) < 3:
            return ""
        
        # 10. 숫자만 있는 텍스트 제거
        if text.isdigit():
            return ""
        
        return text
        
    except Exception as e:
        print(f"⚠️  텍스트 정제 중 오류 발생: {e}")
        print(f"   원본 텍스트: {original_text[:50]}...")
        return ""


def clean_captions(posts: List[Dict]) -> List[Dict]:
    """
    여러 게시글 리스트의 캡션을 정제합니다.
    :param posts: 원본 게시글 리스트 (caption 포함)
    :return: 정제된 caption 추가된 리스트
    """
    if not posts:
        print("⚠️  정제할 게시글이 없습니다.")
        return []
    
    print(f"📝 {len(posts)}개 게시글 텍스트 정제 시작...")
    
    cleaned_count = 0
    empty_count = 0
    
    for i, post in enumerate(posts):
        # caption 필드 확인
        caption = post.get("caption", "")
        
        # 텍스트 정제
        cleaned_text = clean_caption(caption)
        post["cleaned_caption"] = cleaned_text
        
        # 통계 업데이트
        if cleaned_text:
            cleaned_count += 1
        else:
            empty_count += 1
            
        # 진행 상황 표시
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(posts)} 게시글 정제 완료")
    
    print(f"✅ 텍스트 정제 완료")
    print(f"   정제 성공: {cleaned_count}개")
    print(f"   빈 텍스트: {empty_count}개")
    
    return posts


def filter_meaningful_posts(posts: List[Dict], min_length: int = 10) -> List[Dict]:
    """
    의미 있는 텍스트를 가진 게시글만 필터링
    :param posts: 정제된 게시글 리스트
    :param min_length: 최소 텍스트 길이
    :return: 필터링된 게시글 리스트
    """
    meaningful_posts = []
    
    for post in posts:
        cleaned_text = post.get("cleaned_caption", "")
        
        # 최소 길이 확인
        if len(cleaned_text) >= min_length:
            # 의미 있는 단어 포함 여부 확인
            words = cleaned_text.split()
            if len(words) >= 3:  # 최소 3개 단어
                meaningful_posts.append(post)
    
    print(f"📊 의미 있는 게시글 필터링: {len(meaningful_posts)}/{len(posts)}개")
    return meaningful_posts


def save_cleaned_posts(posts: List[Dict], save_path: str) -> None:
    """
    정제된 게시글을 파일로 저장
    :param posts: 정제된 게시글 리스트
    :param save_path: 저장할 파일 경로
    """
    try:
        # 디렉토리 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 저장할 데이터 구성
        save_data = {
            "posts": posts,
            "summary": {
                "total_posts": len(posts),
                "cleaned_posts": len([p for p in posts if p.get("cleaned_caption", "")]),
                "empty_posts": len([p for p in posts if not p.get("cleaned_caption", "")])
            }
        }
        
        # JSON 파일로 저장
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 정제된 데이터 저장 완료: {save_path}")
        
    except Exception as e:
        print(f"❌ 파일 저장 실패: {e}")


def load_posts(file_path: str) -> List[Dict]:
    """
    게시글 데이터 로드
    :param file_path: 파일 경로
    :return: 게시글 리스트
    """
    try:
        if not os.path.exists(file_path):
            print(f"❌ 파일이 존재하지 않습니다: {file_path}")
            return []
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 데이터 구조 확인
        if isinstance(data, list):
            posts = data
        elif isinstance(data, dict):
            posts = data.get("posts", data.get("data", []))
        else:
            print("❌ 알 수 없는 데이터 구조")
            return []
        
        print(f"✅ {len(posts)}개 게시글 로드 완료")
        return posts
        
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return []


def main():
    """메인 실행 함수"""
    print("🧹 텍스트 정제 프로세스 시작")
    
    # 입력 파일 경로
    input_file = "../data/여행_posts.json"
    
    # 게시글 로드
    posts = load_posts(input_file)
    if not posts:
        print("❌ 처리할 데이터가 없습니다.")
        return
    
    # 텍스트 정제
    cleaned_posts = clean_captions(posts)
    
    # 의미 있는 게시글만 필터링
    meaningful_posts = filter_meaningful_posts(cleaned_posts, min_length=10)
    
    # 정제 결과 저장
    output_file = "../data/여행_posts_cleaned.json"
    save_cleaned_posts(meaningful_posts, output_file)
    
    # 샘플 결과 출력
    print("\n📋 정제 결과 샘플:")
    for i, post in enumerate(meaningful_posts[:3]):
        print(f"\n{i+1}. 원본: {post.get('caption', '')[:50]}...")
        print(f"   정제: {post.get('cleaned_caption', '')[:50]}...")
    
    print("\n✅ 텍스트 정제 완료!")


if __name__ == "__main__":
    main()