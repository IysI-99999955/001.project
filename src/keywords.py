# src/keywords.py
import json
import os
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from collections import Counter

def load_keyword_model(model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS") -> Optional[KeyBERT]:
    """
    KeyBERT 키워드 추출 모델 로딩
    :param model_name: SBERT 임베딩 모델 이름
    :return: KeyBERT 모델 또는 None (실패 시)
    """
    try:
        print(f"한국어 SBERT 모델 로딩 중: {model_name}")
        print("⚠️  최초 실행 시 모델 다운로드에 시간이 소요될 수 있습니다...")
        
        sbert_model = SentenceTransformer(model_name)
        kw_model = KeyBERT(sbert_model)
        
        print("✅ 키워드 추출 모델 로딩 완료")
        return kw_model
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        print("💡 다음 명령어로 필요한 라이브러리를 설치하세요:")
        print("   pip install keybert sentence-transformers")
        return None

def extract_keywords(posts: List[Dict], model: KeyBERT, top_n: int = 3) -> List[Dict]:
    """
    각 게시글에서 상위 키워드 추출
    :param posts: 정제 및 감정 분석된 게시글 리스트
    :param model: KeyBERT 모델
    :param top_n: 추출할 키워드 수
    :return: keywords 필드가 추가된 게시글 리스트
    """
    if not model:
        print("❌ 모델이 로드되지 않았습니다.")
        return posts
    
    print(f"🔍 {len(posts)}개 게시글에서 키워드 추출 시작...")
    
    for i, post in enumerate(posts):
        try:
            # cleaned_caption 필드가 있는지 확인
            text = post.get("cleaned_caption", post.get("caption", ""))
            
            if not text or len(text.strip()) < 10:
                post["keywords"] = []
                continue
            
            # 키워드 추출
            keywords = model.extract_keywords(
                text, 
                top_n=top_n,
                keyphrase_ngram_range=(1, 2),  # 1-2단어 키워드
                stop_words=None,
                use_maxsum=True,  # 다양성 증가
                nr_candidates=20,  # 후보 키워드 수
                diversity=0.5  # 다양성 파라미터
            )
            
            # 키워드만 추출 (점수 제외)
            post["keywords"] = [kw[0] for kw in keywords]
            
            # 진행 상황 표시
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(posts)} 게시글 처리 완료")
                
        except Exception as e:
            print(f"  게시글 {i+1} 키워드 추출 실패: {e}")
            post["keywords"] = []
    
    print("✅ 키워드 추출 완료")
    return posts

def get_keyword_frequency(posts: List[Dict], min_length: int = 2) -> Counter:
    """
    전체 게시글에서 등장한 키워드 빈도 계산
    :param posts: keywords 필드 포함된 게시글 리스트
    :param min_length: 최소 키워드 길이
    :return: 키워드 빈도 Counter 객체
    """
    all_keywords = []
    
    for post in posts:
        keywords = post.get("keywords", [])
        # 최소 길이 필터링
        filtered_keywords = [kw for kw in keywords if len(kw) >= min_length]
        all_keywords.extend(filtered_keywords)
    
    print(f"📊 총 {len(all_keywords)}개의 키워드 발견")
    return Counter(all_keywords)

def save_keyword_analysis(posts: List[Dict], keyword_freq: Counter, save_path: str) -> None:
    """
    키워드 분석 결과 저장
    :param posts: 키워드가 추가된 게시글 리스트
    :param keyword_freq: 키워드 빈도 Counter
    :param save_path: 저장할 파일 경로
    """
    try:
        # 분석 결과 구성
        analysis_result = {
            "posts": posts,
            "keyword_analysis": {
                "total_keywords": len(keyword_freq),
                "top_keywords": keyword_freq.most_common(20),
                "keyword_frequency": dict(keyword_freq)
            },
            "analysis_date": json.dumps({"timestamp": "2024-01-01 00:00:00"})  # 현재 시간으로 대체 가능
        }
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 파일 저장
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 키워드 분석 결과 저장: {save_path}")
        
    except Exception as e:
        print(f"❌ 저장 실패: {e}")

def load_analyzed_posts(file_path: str) -> List[Dict]:
    """
    분석된 게시글 데이터 로드
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
        elif isinstance(data, dict) and "posts" in data:
            posts = data["posts"]
        else:
            print("❌ 알 수 없는 데이터 구조입니다.")
            return []
        
        print(f"✅ {len(posts)}개의 게시글 로드 완료")
        return posts
        
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return []

def main():
    """메인 실행 함수"""
    print("🔍 키워드 추출 분석 시작")
    
    # 모델 로딩
    kw_model = load_keyword_model()
    if not kw_model:
        print("❌ 모델 로딩 실패로 종료합니다.")
        return
    
    # 분석할 데이터 파일 경로
    input_file = "../data/여행_posts_analyzed.json"
    
    # 분석된 게시글 데이터 불러오기
    posts = load_analyzed_posts(input_file)
    if not posts:
        print("❌ 분석할 데이터가 없습니다.")
        return
    
    # 키워드 추출
    posts_with_keywords = extract_keywords(posts, kw_model, top_n=5)
    
    # 키워드 빈도 분석
    keyword_freq = get_keyword_frequency(posts_with_keywords)
    
    # 결과 출력
    print("\n📊 상위 키워드 TOP 10:")
    for keyword, count in keyword_freq.most_common(10):
        print(f"  {keyword}: {count}회")
    
    # 결과 저장
    output_file = "../data/여행_posts_keywords.json"
    save_keyword_analysis(posts_with_keywords, keyword_freq, output_file)
    
    print("\n✅ 키워드 분석 완료!")

if __name__ == "__main__":
    main()