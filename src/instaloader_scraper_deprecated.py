# src/scraper.py
import instaloader
import json
import os
import time
import random
from datetime import datetime
from typing import List, Dict, Optional # Optional 임포트 추가

class SafeInstagramScraper:
    def __init__(self, min_delay: int = 30, max_delay: int = 60, max_retries: int = 3):
        """
        안전한 Instagram 스크래퍼 초기화
        :param min_delay: 최소 대기 시간(초)
        :param max_delay: 최대 대기 시간(초)
        :param max_retries: 최대 재시도 횟수
        """
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.loader = None
        self._init_loader()
    
    def _init_loader(self):
        """Instaloader 초기화"""
        try:
            self.loader = instaloader.Instaloader(
                download_pictures=False,
                download_videos=False,
                download_video_thumbnails=False,
                download_comments=False,
                save_metadata=False,
                quiet=True,
                # 더 안전한 설정
                sleep=True,  # 자동 대기 기능 활성화
                max_connection_attempts=3
            )
            print("Instaloader 초기화 완료")
        except Exception as e:
            print(f"Instaloader 초기화 실패: {e}")
            raise

    def _safe_delay(self):
        """랜덤 대기 시간으로 더 자연스러운 요청 패턴 생성"""
        delay = random.randint(self.min_delay, self.max_delay)
        print(f"안전을 위해 {delay}초 대기 중...")
        time.sleep(delay)

    def fetch_hashtag_posts(self, hashtag: str, max_count: int = 50, progress_callback: Optional[callable] = None) -> List[Dict]: # progress_callback 인자 추가
        """
        재시도 로직이 포함된 안전한 해시태그 스크랩
        :param hashtag: 스크랩할 해시태그
        :param max_count: 수집할 최대 게시글 수
        :param progress_callback: (current_count, max_count)를 인자로 받는 콜백 함수 (진행률 업데이트용)
        """
        for attempt in range(self.max_retries):
            try:
                print(f"'{hashtag}' 해시태그 스크랩 시도 {attempt + 1}/{self.max_retries}")
                
                hashtag_obj = instaloader.Hashtag.from_name(self.loader.context, hashtag)
                posts = hashtag_obj.get_posts()
                
                results = []
                collected = 0
                
                # 콜백이 있다면, 0%로 초기화
                if progress_callback:
                    progress_callback(0, max_count)

                for post in posts:
                    if collected >= max_count:
                        break
                    
                    try:
                        if not post.caption:
                            continue
                            
                        results.append({
                            "caption": post.caption,
                            "date": post.date_utc.strftime("%Y-%m-%d %H:%M:%S"),
                            "shortcode": post.shortcode,
                            "url": f"https://www.instagram.com/p/{post.shortcode}/",
                            "hashtag": hashtag,
                            "likes": post.likes if hasattr(post, 'likes') else 0,
                            "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        collected += 1
                        
                        # 진행률 콜백 호출
                        if progress_callback:
                            progress_callback(collected, max_count)
                        
                        # 게시글 수집 간에도 짧은 대기
                        if collected % 10 == 0:
                            print(f"  {collected}개 게시글 수집 완료...")
                            time.sleep(2)
                            
                    except Exception as e:
                        print(f"  게시글 처리 중 오류: {e}")
                        continue
                
                print(f"'{hashtag}' 스크랩 성공: {len(results)}개 게시글 수집")
                return results
                
            except instaloader.exceptions.InstaloaderException as e:
                print(f"Instagram API 오류 (시도 {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 60  # 점진적 대기 시간 증가
                    print(f"  {wait_time}초 후 재시도...")
                    time.sleep(wait_time)
                else:
                    print(f"  최대 재시도 횟수 초과. '{hashtag}' 스크랩 실패")
                    if progress_callback:
                        progress_callback(0, max_count) # 오류 시 0%로 리셋
                    return []
                    
            except Exception as e:
                print(f"예상치 못한 오류 (시도 {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(30)
                else:
                    if progress_callback:
                        progress_callback(0, max_count) # 오류 시 0%로 리셋
                    return []
        
        return [] # 모든 재시도 실패 시 빈 리스트 반환
