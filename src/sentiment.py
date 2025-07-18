# src/sentiment.py
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
from typing import List, Dict, Optional, Union
import torch
import logging
from functools import lru_cache
import re

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """감정 분석을 위한 클래스"""
    
    def __init__(self, model_name: str = "alsgyu/sentiment-analysis-fine-tuned-model", batch_size: int = 8, force_cpu: bool = False):
        self.model_name = model_name
        self.batch_size = batch_size
        self.force_cpu = force_cpu  # CPU 강제 사용 옵션
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        
        # 감정 임계값 설정 (중립 감정 판단용)
        self.neutral_threshold = 0.6
        
        # 감정 레이블 매핑
        self.label_mapping = {
            'LABEL_0': '부정',
            'LABEL_1': '긍정',
            'NEGATIVE': '부정',
            'POSITIVE': '긍정'
        }
    
    @lru_cache(maxsize=1)
    def load_model(self) -> bool:
        """
        감정 분석 모델을 로드합니다.
        캐싱을 사용하여 중복 로드를 방지합니다.
        """
        try:
            # GPU 사용 가능 여부 확인 (더 안전한 방식)
            device = -1  # 기본적으로 CPU 사용
            device_name = "CPU"
            
            # force_cpu 옵션이 True가 아닐 때만 GPU 시도
            if not self.force_cpu:
                try:
                    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                        # GPU 메모리 체크
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory
                        if gpu_memory > 1e9:  # 1GB 이상
                            device = 0
                            device_name = f"GPU (Memory: {gpu_memory/1e9:.1f}GB)"
                            logger.info(f"GPU detected and will be used: {device_name}")
                        else:
                            logger.info("GPU memory insufficient, using CPU")
                    else:
                        logger.info("CUDA not available, using CPU")
                except Exception as gpu_error:
                    logger.warning(f"GPU 감지 중 오류 발생, CPU 사용: {gpu_error}")
                    device = -1
                    device_name = "CPU (GPU 오류로 인한 대체)"
            else:
                logger.info("Force CPU mode enabled")
            
            # 모델과 토크나이저 직접 로드 (더 나은 제어)
            logger.info(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            logger.info(f"Loading model: {self.model_name}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # GPU 사용 시 모델을 GPU로 이동
            if device == 0:
                try:
                    self.model = self.model.cuda()
                    logger.info("Model moved to GPU successfully")
                except Exception as cuda_error:
                    logger.warning(f"GPU로 모델 이동 실패, CPU 사용: {cuda_error}")
                    device = -1
                    device_name = "CPU (GPU 이동 실패)"
            
            # 파이프라인 생성
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device,
                return_all_scores=True,  # 모든 스코어 반환
                truncation=True,
                max_length=512
            )
            
            st.success(f"✅ 감정 분석 모델 '{self.model_name}' 로드 완료! (Device: {device_name})")
            logger.info(f"Model loaded successfully: {self.model_name} on {device_name}")
            return True
            
        except Exception as e:
            error_msg = f"❌ 감정 분석 모델 로드 중 오류 발생: {e}"
            st.error(error_msg)
            logger.error(error_msg)
            
            # CPU로 재시도
            try:
                logger.info("GPU 로드 실패, CPU로 재시도...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=-1,  # 강제로 CPU 사용
                    return_all_scores=True,
                    truncation=True,
                    max_length=512
                )
                
                st.warning("⚠️ GPU 로드 실패로 CPU 사용 중입니다.")
                logger.info("Successfully loaded model on CPU as fallback")
                return True
                
            except Exception as cpu_error:
                error_msg = f"❌ CPU 로드도 실패: {cpu_error}"
                st.error(error_msg)
                logger.error(error_msg)
                return False
    
    def preprocess_text(self, text: str) -> str:
        """
        텍스트 전처리를 수행합니다.
        """
        if not text:
            return ""
        
        # 기본 정리
        text = text.strip()
        
        # 과도한 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 과도한 특수문자 반복 제거 (예: !!!!!! -> !)
        text = re.sub(r'([!?.])\1{2,}', r'\1', text)
        
        # 해시태그에서 # 제거 (선택사항)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        return text
    
    def classify_sentiment(self, scores: List[Dict]) -> tuple[str, float]:
        """
        스코어를 기반으로 감정을 분류합니다.
        중립 감정 판단 로직 포함.
        """
        if not scores:
            return '알 수 없음', 0.0
        
        # 최고 스코어 찾기
        best_score = max(scores, key=lambda x: x['score'])
        label = best_score['label']
        score = best_score['score']
        
        # 중립 감정 판단
        if score < self.neutral_threshold:
            return '중립', score
        
        # 레이블 매핑
        korean_sentiment = self.label_mapping.get(label, '알 수 없음')
        return korean_sentiment, score
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        배치 단위로 감정 분석을 수행합니다.
        """
        if not self.pipeline:
            return [{'sentiment': '처리불가', 'score': 0.0} for _ in texts]
        
        try:
            # 배치 처리
            results = self.pipeline(texts)
            
            processed_results = []
            for result in results:
                sentiment, score = self.classify_sentiment(result)
                processed_results.append({
                    'sentiment': sentiment,
                    'score': score
                })
            
            return processed_results
            
        except Exception as e:
            logger.error(f"배치 감정 분석 중 오류: {e}")
            return [{'sentiment': '오류', 'score': 0.0} for _ in texts]
    
    def analyze_sentiment(self, posts: List[Dict]) -> List[Dict]:
        """
        주어진 릴스 리스트에 대해 감정 분석을 수행합니다.
        배치 처리를 통해 성능을 최적화합니다.
        """
        if not self.pipeline:
            st.error("감정 분석 모델이 로드되지 않아 감정 분석을 건너뜁니다.")
            return [{'sentiment': '처리불가', 'score': 0.0, **post} for post in posts]
        
        processed_posts = []
        valid_posts = []
        valid_texts = []
        
        # 유효한 텍스트 필터링 및 전처리
        for i, post in enumerate(posts):
            caption = post.get("cleaned_caption", "")
            
            if not caption or caption.strip() == "":
                processed_posts.append({
                    **post, 
                    'sentiment': '처리불가', 
                    'score': 0.0,
                    'index': i
                })
            else:
                preprocessed_caption = self.preprocess_text(caption)
                valid_posts.append({**post, 'index': i})
                valid_texts.append(preprocessed_caption)
        
        # 배치 단위로 처리
        if valid_texts:
            try:
                with st.spinner(f"감정 분석 중... ({len(valid_texts)}개 텍스트 처리)"):
                    # 배치 크기로 나누어 처리
                    for i in range(0, len(valid_texts), self.batch_size):
                        batch_texts = valid_texts[i:i + self.batch_size]
                        batch_posts = valid_posts[i:i + self.batch_size]
                        
                        batch_results = self.analyze_batch(batch_texts)
                        
                        for post, result in zip(batch_posts, batch_results):
                            processed_posts.append({
                                **post,
                                'sentiment': result['sentiment'],
                                'score': result['score']
                            })
                            
            except Exception as e:
                st.error(f"배치 처리 중 오류 발생: {e}")
                logger.error(f"Batch processing error: {e}")
                
                # 개별 처리로 대체
                for post, text in zip(valid_posts, valid_texts):
                    try:
                        result = self.pipeline(text)[0]
                        sentiment, score = self.classify_sentiment(result)
                        processed_posts.append({
                            **post,
                            'sentiment': sentiment,
                            'score': score
                        })
                    except Exception as individual_error:
                        logger.warning(f"개별 처리 오류: {individual_error}")
                        processed_posts.append({
                            **post,
                            'sentiment': '오류',
                            'score': 0.0
                        })
        
        # 원래 순서대로 정렬
        processed_posts.sort(key=lambda x: x['index'])
        
        # index 필드 제거
        for post in processed_posts:
            post.pop('index', None)
        
        return processed_posts

# 전역 인스턴스 (싱글톤 패턴)
_sentiment_analyzer = None

def get_sentiment_analyzer(model_name: str = "alsgyu/sentiment-analysis-fine-tuned-model", force_cpu: bool = False) -> SentimentAnalyzer:
    """
    감정 분석기 인스턴스를 반환합니다.
    싱글톤 패턴으로 구현되어 메모리 효율성을 높입니다.
    """
    global _sentiment_analyzer
    if _sentiment_analyzer is None or _sentiment_analyzer.model_name != model_name:
        _sentiment_analyzer = SentimentAnalyzer(model_name, force_cpu=force_cpu)
        if not _sentiment_analyzer.load_model():
            _sentiment_analyzer = None
    return _sentiment_analyzer

# 기존 함수들과의 호환성을 위한 래퍼 함수들
def load_sentiment_model():
    """기존 코드와의 호환성을 위한 래퍼 함수"""
    analyzer = get_sentiment_analyzer()
    return analyzer.pipeline if analyzer else None

def analyze_sentiment(posts: List[Dict], sentiment_pipeline=None) -> List[Dict]:
    """기존 코드와의 호환성을 위한 래퍼 함수"""
    analyzer = get_sentiment_analyzer()
    if analyzer:
        return analyzer.analyze_sentiment(posts)
    else:
        return [{'sentiment': '처리불가', 'score': 0.0, **post} for post in posts]

# 사용 예시
if __name__ == "__main__":
    # 클래스 방식 사용
    analyzer = SentimentAnalyzer()
    if analyzer.load_model():
        test_posts = [
            {"cleaned_caption": "오늘 정말 좋은 날이다!"},
            {"cleaned_caption": "너무 슬프다..."},
            {"cleaned_caption": "그냥 평범한 하루"},
            {"cleaned_caption": ""}
        ]
        
        results = analyzer.analyze_sentiment(test_posts)
        for result in results:
            print(f"감정: {result['sentiment']}, 점수: {result['score']:.3f}")

# 추천 모델 리스트 (성능 순서)
RECOMMENDED_MODELS = [
    "sangrimlee/bert-base-multilingual-cased-nsmc",  # NSMC 데이터셋 기반 (not good)
    "beomi/kcbert-base",                    # 한국어 댓글 데이터로 학습된 BERT(not good)
    "alsgyu/sentiment-analysis-fine-tuned-model",  # 감정 분석 파인튜닝 모델 (현재 사용)
    "tabularisai/multilingual-sentiment-analysis",   # 다국어 감정 분석 모델
]