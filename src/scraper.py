# src/scraper.py
import instaloader
import json
import os
import time
import random
from datetime import datetime
from typing import List, Dict, Optional

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

    def fetch_hashtag_posts(self, hashtag: str, max_count: int = 50) -> List[Dict]:
        """
        재시도 로직이 포함된 안전한 해시태그 스크랩
        """
        for attempt in range(self.max_retries):
            try:
                print(f"'{hashtag}' 해시태그 스크랩 시도 {attempt + 1}/{self.max_retries}")
                
                hashtag_obj = instaloader.Hashtag.from_name(self.loader.context, hashtag)
                posts = hashtag_obj.get_posts()
                
                results = []
                collected = 0
                
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
                    return []
                    
            except Exception as e:
                print(f"예상치 못한 오류 (시도 {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(30)
                else:
                    return []
        
        return []

    def scrape_multiple_hashtags(self, hashtags: List[str], max_count: int = 30) -> Dict[str, List[Dict]]:
        """
        매우 안전한 다중 해시태그 스크랩
        """
        if len(hashtags) > 5:
            print("⚠️  한 번에 5개 이상의 해시태그는 권장하지 않습니다.")
            print("   IP 차단 위험이 있습니다. 계속하시겠습니까? (y/n)")
            if input().lower() != 'y':
                return {}
        
        all_results = {}
        
        for i, tag in enumerate(hashtags):
            print(f"\n[{i+1}/{len(hashtags)}] '{tag}' 해시태그 스크랩 시작")
            print(f"예상 소요 시간: {(len(hashtags) - i) * (self.max_delay + 30) // 60}분")
            
            results = self.fetch_hashtag_posts(tag, max_count)
            all_results[tag] = results
            
            # 마지막 해시태그가 아니라면 안전한 대기
            if i < len(hashtags) - 1:
                self._safe_delay()
        
        return all_results

    def save_posts_to_json(self, posts: List[Dict], save_path: str) -> None:
        """JSON 파일 저장"""
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(posts, f, ensure_ascii=False, indent=2)
                
            print(f"✅ 파일 저장 완료: {save_path}")
            
        except Exception as e:
            print(f"❌ 파일 저장 실패: {e}")

def main():
    """메인 실행 함수"""
    print("🔒 안전한 Instagram 스크래핑 시작")
    print("⚠️  주의: Instagram 정책을 준수하여 사용하세요.")
    print("⚠️  상업적 사용 시 Instagram 이용약관을 확인하세요.\n")
    
    # 더 안전한 설정으로 초기화
    scraper = SafeInstagramScraper(
        min_delay=45,  # 최소 45초 대기
        max_delay=90,  # 최대 90초 대기
        max_retries=3
    )
    
    # 권장: 단일 해시태그만 스크랩
    single_tag = "여행"
    print(f"단일 해시태그 '{single_tag}' 스크랩 (권장 방식):")
    result = scraper.fetch_hashtag_posts(single_tag, max_count=30)
    
    if result:
        scraper.save_posts_to_json(result, f"../data/{single_tag}_posts.json")
        print(f"✅ {len(result)}개의 게시글이 저장되었습니다.")
    
    # 다중 해시태그 (신중하게 사용)
    use_multiple = input("\n다중 해시태그 스크랩을 진행하시겠습니까? (y/n): ")
    if use_multiple.lower() == 'y':
        multiple_tags = ["카페", "맛집"]  # 최대 2-3개 권장
        print(f"⚠️  다중 해시태그 스크랩 시작: {multiple_tags}")
        
        all_results = scraper.scrape_multiple_hashtags(multiple_tags, max_count=20)
        
        # 각 해시태그별 저장
        for tag, posts in all_results.items():
            if posts:
                scraper.save_posts_to_json(posts, f"../data/{tag}_posts.json")
        
        print(f"\n✅ 다중 해시태그 스크랩 완료")

if __name__ == "__main__":
    main()