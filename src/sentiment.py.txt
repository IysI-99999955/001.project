# src/sentiment.py

from transformers import pipeline
from typing import List, Dict
from tqdm import tqdm # tqdm 라이브러리 임포트

def load_sentiment_model(model_name: str = "nlp04/korean-sentiment-classification"):
    """
    감정 분석용 모델 로드
    :param model_name: Hugging Face 모델 이름
    :return: 파이프라인 모델 객체
    """
    # 모델 로드가 시작됨을 사용자에게 알립니다.
    # transformers pipeline은 모델 다운로드 시 자체 진행률 표시를 제공할 수 있습니다.
    print(f"🔄 감정 분석 모델 '{model_name}' 로드 중... (이 과정은 시간이 다소 소요될 수 있습니다)")
    return pipeline("sentiment-analysis", model=model_name)


def analyze_sentiment(posts: List[Dict], model) -> List[Dict]:
    """
    cleaned_caption을 대상으로 감정 분석 수행
    :param posts: 게시글 리스트 (cleaned_caption 필드 포함)
    :param model: Transformers pipeline 감정 분석 모델
    :return: 감정 분석 결과 추가된 게시글 리스트
    """
    # tqdm을 사용하여 게시글 처리 진행 상황을 표시합니다.
    # 'desc' 매개변수로 진행률 바 앞에 표시될 설명을 지정합니다.
    for post in tqdm(posts, desc="📝 감정 분석 진행 중"):
        try:
            # cleaned_caption이 너무 길 경우 모델 입력 제한에 맞춰 자릅니다.
            result = model(post["cleaned_caption"][:512])[0]
            post["sentiment"] = result["label"]
            post["score"] = round(result["score"], 4)
        except Exception as e:
            # 오류 발생 시 'ERROR'로 처리하고 점수를 0.0으로 설정합니다.
            # 실제 사용 시에는 오류 로깅을 추가하는 것이 좋습니다.
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
        with open("../data/여행_posts_cleaned.json", "r", encoding="utf-8") as f:
            posts = json.load(f)
        print(f"✅ 총 {len(posts)}개의 게시글 로드 완료.")
    except FileNotFoundError:
        print("❌ 오류: '여행_posts_cleaned.json' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        exit()
    except json.JSONDecodeError:
        print("❌ 오류: '여행_posts_cleaned.json' 파일이 유효한 JSON 형식이 아닙니다.")
        exit()
    except Exception as e:
        print(f"❌ 오류: 파일 로드 중 예상치 못한 오류 발생: {e}")
        exit()

    # 3. 감정 분석 수행
    analyzed = analyze_sentiment(posts, model)

    # 4. 분석 결과 저장
    print("\n💾 감정 분석 결과 저장 중...")
    try:
        with open("../data/여행_posts_analyzed.json", "w", encoding="utf-8") as f:
            json.dump(analyzed, f, ensure_ascii=False, indent=2)
        print("✨ 감정 분석 완료. 'sentiment', 'score' 필드가 추가되었습니다.")
    except Exception as e:
        print(f"❌ 오류: 결과 저장 중 예상치 못한 오류 발생: {e}")

