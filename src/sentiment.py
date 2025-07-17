# src/sentiment.py

from transformers import pipeline
from typing import List, Dict
from tqdm import tqdm # tqdm 라이브러리 임포트
import os # os 모듈 임포트 (파일 경로 확인용)

def load_sentiment_model(model_name: str = "snunlp/KR-FinBert"):
    """
    감정 분석용 모델 로드
    :param model_name: Hugging Face 모델 이름
    :return: 파이프라인 모델 객체
    """
    print(f"🔄 감정 분석 모델 '{model_name}' 로드 중... (이 과정은 시간이 다소 소요될 수 있습니다)")
    return pipeline("sentiment-analysis", model=model_name)


def analyze_sentiment(posts: List[Dict], model) -> List[Dict]:
    """
    cleaned_caption을 대상으로 감정 분석 수행
    모델의 출력 레이블을 '긍정', '중립', '부정'으로 매핑합니다.
    :param posts: 게시글 리스트 (cleaned_caption 필드 포함)
    :param model: Transformers pipeline 감정 분석 모델
    :return: 감정 분석 결과 추가된 게시글 리스트
    """
    # 모델의 출력 레이블을 한글로 매핑하는 딕셔너리
    # 사용하는 모델의 실제 출력 레이블에 따라 이 매핑을 조정해야 합니다.
    # 예: 'positive' -> '긍정', 'negative' -> '부정', 'neutral' -> '중립'
    label_map = {
        "positive": "긍정",
        "negative": "부정",
        "neutral": "중립",
        "LABEL_0": "부정", # 모델이 LABEL_0, LABEL_1 등으로 출력할 경우를 대비 (예시)
        "LABEL_1": "긍정",
        # 추가적인 레이블이 있다면 여기에 매핑을 추가하세요.
        # snunlp/KR-FinBert는 주로 'positive', 'negative'를 출력합니다.
    }

    for post in tqdm(posts, desc="📝 감정 분석 진행 중"):
        try:
            # cleaned_caption이 너무 길 경우 모델 입력 제한에 맞춰 자릅니다.
            # 캡션이 없거나 유효하지 않은 경우 건너뜁니다.
            if not post.get("cleaned_caption") or not isinstance(post["cleaned_caption"], str) or len(post["cleaned_caption"].strip()) == 0:
                post["sentiment"] = "처리불가"
                post["score"] = 0.0
                continue

            result = model(post["cleaned_caption"][:512])[0]
            original_label = result["label"]
            score = round(result["score"], 4)

            # 모델의 출력 레이블을 한글로 매핑
            mapped_label = label_map.get(original_label.lower(), "중립") # 기본값은 '중립'
            
            post["sentiment"] = mapped_label
            post["score"] = score
        except Exception as e:
            # 오류 발생 시 'ERROR'로 처리하고 점수를 0.0으로 설정합니다.
            # 실제 사용 시에는 오류 로깅을 추가하는 것이 좋습니다.
            print(f"감정 분석 중 오류 발생: {e} (캡션: {post.get('cleaned_caption', '')[:50]}...)")
            post["sentiment"] = "ERROR"
            post["score"] = 0.0
    return posts


if __name__ == "__main__":
    import json

    # 1. 감정 분석 모델 로드
    model = load_sentiment_model()

    # 2. 입력 파일 로드
    print("\n📂 입력 파일 로드 중...")
    try:
        # 현재 스크립트의 절대 경로를 기준으로 프로젝트 루트를 찾습니다.
        current_script_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_script_path)
        project_root_dir = os.path.join(current_dir, "..", "..") # src 폴더에서 두 단계 위로

        input_file_path = os.path.join(project_root_dir, "data", "여행_posts_cleaned.json")

        with open(input_file_path, "r", encoding="utf-8") as f:
            posts = json.load(f)
        print(f"✅ 총 {len(posts)}개의 게시글 로드 완료.")
    except FileNotFoundError:
        print(f"❌ 오류: '{input_file_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        exit()
    except json.JSONDecodeError:
        print(f"❌ 오류: '{input_file_path}' 파일이 유효한 JSON 형식이 아닙니다.")
        exit()
    except Exception as e:
        print(f"❌ 오류: 파일 로드 중 예상치 못한 오류 발생: {e}")
        exit()

    # 3. 감정 분석 수행
    print("\n📝 감정 분석 시작...")
    analyzed_posts = analyze_sentiment(posts, model)
    print("✅ 감정 분석 완료.")

    # 4. 결과 출력 (상위 5개만)
    print("\n📊 감정 분석 결과 (상위 5개):")
    for i, post in enumerate(analyzed_posts[:5]):
        print(f"  - 캡션: {post['cleaned_caption'][:50]}...")
        print(f"    감정: {post.get('sentiment', 'N/A')}, 점수: {post.get('score', 'N/A')}")
    
    # 5. 감정 분포 요약
    sentiment_counts = Counter([p.get('sentiment') for p in analyzed_posts if p.get('sentiment') != 'ERROR'])
    print("\n📈 전체 감정 분포:")
    for sentiment, count in sentiment_counts.items():
        print(f"  - {sentiment}: {count}개")

    # 6. 결과 저장 (선택 사항)
    output_file_path = os.path.join(project_root_dir, "data", "여행_posts_analyzed.json")
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(analyzed_posts, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 분석 결과 저장 완료: {output_file_path}")
    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")

