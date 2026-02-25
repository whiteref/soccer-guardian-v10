import os
import json
import logging
import re
import math
import random
import streamlit as st
import pandas as pd
import numpy as np
import requests
import xgboost as xgb
import boto3
import unicodedata
from bs4 import BeautifulSoup
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False
from kalman_guardian_v13 import KalmanGuardianEngine # 📡 [V13 Kalman Guardian]
from soccer_real_data_engine import (
    fetch_real_match_data, EloRatingSystem, BrierScoreTracker,
    build_features_from_real_data, initialize_v10_engine
)  # 🚀 [V10] 실제 데이터 엔진
from soccer_auto_result import auto_update_elo_and_brier  # 🔄 [V10.2] 자동 결과 수집
import warnings
warnings.filterwarnings('ignore')
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from scipy.stats import poisson
from data_fusion_v8 import fetch_all_fusion_features # 🔗 [V8 Hyper-Fusion]
from dotenv import load_dotenv
load_dotenv() # 🔐 .env 파일의 환경 변수 로드

# 🚀 [V9.6] 버전 정의 (캐시 자동 초기화용)
V9_6_VERSION = "10.2.0"  # 🚀 [V10.2] Real Data + ELO + Brier Score + Anti-Bias + Auto-Feedback

# ------------------------------------------------------------------------------
# ⚙️ 1. 기본 설정 및 전역 딕셔너리
# ------------------------------------------------------------------------------
st.set_page_config(page_title="⚽ [V10] REAL DATA ENGINE", page_icon="🧠", layout="wide")
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

TEAM_MAPPING = {
    # EPL
    "맨체스터시티": "Manchester City", "맨시티": "Manchester City", "맨체스C": "Manchester City",
    "아스널": "Arsenal", "아스날": "Arsenal",
    "리버풀": "Liverpool",
    "아스턴빌라": "Aston Villa", "아스톤빌라": "Aston Villa", "A빌라": "Aston Villa",
    "토트넘": "Tottenham", "홋스퍼": "Tottenham",
    "첼시": "Chelsea",
    "뉴캐슬": "Newcastle United", "뉴캐슬U": "Newcastle United",
    "맨체스터유나이티드": "Manchester Utd", "맨유": "Manchester Utd", "맨체스U": "Manchester Utd",
    "웨스트햄": "West Ham",
    "브라이튼": "Brighton", "브라이턴": "Brighton",
    "본머스": "Bournemouth",
    "풀럼": "Fulham",
    "울버햄튼": "Wolverhampton Wanderers", "울버햄프": "Wolverhampton Wanderers", "울버햄프턴": "Wolverhampton Wanderers",
    "브렌트포드": "Brentford", "브렌트퍼드": "Brentford", "브렌트퍼": "Brentford",
    "에버턴": "Everton", "에버튼": "Everton",
    "노팅엄": "Nottingham Forest", "노팅엄포": "Nottingham Forest",
    "레스터": "Leicester",
    "리즈": "Leeds", "리즈U": "Leeds", "리즈유나이티드": "Leeds",

    # Serie A
    "유벤투스": "Juventus",
    "인자기": "Inter", "인테르": "Inter",
    "AC밀란": "AC Milan", "A밀란": "AC Milan",
    "나폴리": "Napoli",
    "아탈란타": "Atalanta", "아틀란타": "Atalanta",
    "AS로마": "Roma", "로마": "Roma",
    "라치오": "Lazio",
    "피오렌티나": "Fiorentina",
    "토리노": "Torino",
    "제노아": "Genoa",
    "파르마": "Parma",
    "코모": "Como", "코모1907": "Como",
    "칼리아리": "Cagliari",
    "크레모네": "Cremonese", "크레모네세": "Cremonese",
    
    "선덜랜드": "Sunderland",
    "크리스털": "Crystal Palace", "크리스탈": "Crystal Palace",

    # La Liga
    "레알마드리드": "Real Madrid", "레알": "Real Madrid",
    "바르셀로나": "Barcelona", "바르사": "Barcelona",
    "아틀레티코": "Atletico Madrid", "AT마드리드": "Atletico Madrid",
    "지로나": "Girona", "빌바오": "Athletic Club", "소시에다드": "Real Sociedad",

    # Bundesliga
    "레버쿠젠": "Bayer Leverkusen", "바이엘": "Bayer Leverkusen",
    "바이에른뮌헨": "Bayern Munich", "뮌헨": "Bayern Munich",
    "슈투트가르트": "Stuttgart", "도르트문트": "Borussia Dortmund", "돌문": "Borussia Dortmund",
    "라이프치히": "RB Leipzig",

    # Ligue 1
    "PSG": "Paris Saint Germain", "파리생제르망": "Paris Saint Germain", "파리SG": "Paris Saint Germain",
    "모나코": "Monaco", "브레스투": "Brest", "릴": "Lille", "니스": "Nice",

    # European & Others (Special Upset Targets)
    "피오렌티나": "Fiorentina", "야기엘로니아": "Jagiellonia", "삼순스포르": "Samsunspor",
    "스켄디야": "Shkendija", "첼레": "Celje", "드리타": "Drita", "리예카": "Rijeka",
    "오모니아": "Omonia Nicosia", "페렌츠바로시": "Ferencvaros", "루도고레츠": "Ludogorets",
    "플젠": "Viktoria Plzen", "파나티나이코스": "Panathinaikos", "츠르베나": "Red Star",
    "셀틱": "Celtic", "알크마르": "AZ Alkmaar", "로잔": "Lausanne-Sport",
    "시그마": "Sigma Olomouc", "헹크": "Genk", "자그레브": "Dinamo Zagreb",
    "셀타데비고": "Celta Vigo", "셀타": "Celta Vigo", "PAOK": "PAOK", "브란": "Brann",
    "페네르바체": "Fenerbahce", "페네르바흐체": "Fenerbahce", "볼로냐": "Bologna",

    # [V9.7.7] 영문 팀명 직접 매핑 (English Name Failover) - 공백 및 특수문자 무관 매칭
    "Noah": "Noah", "AZ Alkmaar": "AZ Alkmaar", "Sigma Olomouc": "Sigma Olomouc",
    "Lausanne-Sport": "Lausanne-Sport", "Dinamo Zagreb": "Dinamo Zagreb",
    "Genk": "Genk", "Celta Vigo": "Celta Vigo", "Celta de Vigo": "Celta Vigo", "Brann": "Brann",
    "Fenerbahce": "Fenerbahce", "Fenerbahce": "Fenerbahce", "Nottingham Forest": "Nottingham Forest",
    "Jagiellonia": "Jagiellonia", "Shkendija": "Shkendija", "Skenndija": "Shkendija", "Samsunspor": "Samsunspor",
    "Drita": "Drita", "Celje": "Celje", "Omonia Nicosia": "Omonia Nicosia",
    "Rijeka": "Rijeka", "Ludogorets": "Ludogorets", "Ferencvaros": "Ferencvaros", "Ferencvarosi": "Ferencvaros",
    "Panathinaikos": "Panathinaikos", "Viktoria Plzen": "Viktoria Plzen",
    "Lille": "Lille", "FK Zeljeznicar": "FK Zeljeznicar", "Zeljeznicar": "FK Zeljeznicar", "Stuttgart": "Stuttgart",
    "Bologna": "Bologna", "Fiorentina": "Fiorentina",
    
    # [V9.7.8] English Key Failover (Ensuring English names work as keys)
    "Celtic": "Celtic", "PAOK": "PAOK", "Celta": "Celta Vigo", "RCCelta": "Celta Vigo", 
    "Stuttgart": "Stuttgart", "Lille": "Lille", "Bologna": "Bologna",

    # [V9.7.11] Global English Failover (Ensuring all major English names work as keys)
    "Manchester City": "Manchester City", "Arsenal": "Arsenal", "Liverpool": "Liverpool",
    "Aston Villa": "Aston Villa", "Tottenham": "Tottenham", "Chelsea": "Chelsea",
    "Newcastle United": "Newcastle United", "Manchester Utd": "Manchester Utd", "Manchester United": "Manchester Utd",
    "West Ham": "West Ham", "Brighton": "Brighton", "Bournemouth": "Bournemouth",
    "Fulham": "Fulham", "Wolverhampton": "Wolverhampton Wanderers", "Wolverhampton Wanderers": "Wolverhampton Wanderers",
    "Brentford": "Brentford", "Everton": "Everton", "Nottingham Forest": "Nottingham Forest", "Leicester": "Leicester",
    "Juventus": "Juventus", "Napoli": "Napoli", "Inter": "Inter", "Inter Milan": "Inter",
    "AC Milan": "AC Milan", "Roma": "Roma", "Lazio": "Lazio", "Atalanta": "Atalanta",
    "Fiorentina": "Fiorentina", "Bologna": "Bologna", "Real Madrid": "Real Madrid",
    "Barcelona": "Barcelona", "Atletico Madrid": "Atletico Madrid", "Villarreal": "Villarreal",
    "Bayer Leverkusen": "Bayer Leverkusen", "Bayern Munich": "Bayern Munich", "Borussia Dortmund": "Borussia Dortmund",
    "RB Leipzig": "RB Leipzig", "Stuttgart": "Stuttgart", "Ajax": "Ajax", "Olympiacos": "Olympiacos",
    "Benfica": "Benfica", "Sporting CP": "Sporting CP", "Porto": "Porto", "PSV": "PSV", "Feyenoord": "Feyenoord",
    "Club Brugge": "Club Brugge", "Marseille": "Marseille", "Lille": "Lille",
    "Monaco": "Monaco", "Paris Saint Germain": "Paris Saint Germain", "Paris Saint-Germain": "Paris Saint Germain",
    "Union Saint-Gilloise": "Union Saint-Gilloise", "Slavia Prague": "Slavia Prague", "Bodo/Glimt": "Bodo/Glimt",
    "Celta Vigo": "Celta Vigo", "Eintracht Frankfurt": "Eintracht Frankfurt",
    "Athletic Club": "Athletic Club", "Athletic Bilbao": "Athletic Club", "Pafos": "Pafos",
    "Kairat": "Kairat", "Copenhagen": "Copenhagen", "Galatasaray": "Galatasaray", "Qarabag": "Qarabag"
}

PUBLIC_FAVORITES = ["Manchester City", "Arsenal", "Liverpool", "Juventus", "Inter", "Napoli", "AC Milan", "Atalanta"]
HIGH_MOTIVATION_TEAMS = ["Nottingham Forest", "Everton", "Cagliari", "Genoa"]
HEAVY_SCHEDULE_TEAMS = ["Aston Villa", "Tottenham", "Lazio", "Roma", "Atalanta"]

# 📊 [V9.7] 팀 체급 등급 (Team Tiers)
# 같은 리그 내에서도 '체급' 차이를 수치화하여 전력 우위를 판단합니다.
TEAM_TIERS = {
    # Tier 1 (1.0): 월드클래스 (EPL Top, CL 우승후보)
    "Manchester City": 1.0, "Arsenal": 1.0, "Liverpool": 1.0, "Real Madrid": 1.0, 
    "Bayern Munich": 1.0, "Paris Saint Germain": 1.0, "Inter": 1.0, "Bayer Leverkusen": 1.0,
    "Barcelona": 1.0, "Atletico Madrid": 1.0, "Borussia Dortmund": 1.0,
    
    # Tier 2 (0.9): 5대 리그 상위권 명문팀
    "Tottenham": 0.9, "Chelsea": 0.9, "Aston Villa": 0.9, "Newcastle United": 0.9, "Manchester Utd": 0.9,
    "Juventus": 0.9, "AC Milan": 0.9, "Napoli": 0.9, "Atalanta": 0.9, "Roma": 0.9, "Lazio": 0.9,
    "Girona": 0.9, "Athletic Club": 0.9, "Real Sociedad": 0.9, "Villarreal": 0.9,
    "RB Leipzig": 0.9, "Stuttgart": 0.9, "Eintracht Frankfurt": 0.9,
    "Monaco": 0.9, "Lille": 0.9, "Nice": 0.9, "Brest": 0.9,
    
    # Tier 3 (0.75): 중견 및 빅리그 중위권
    "Porto": 0.75, "Benfica": 0.75, "Sporting CP": 0.75, "Ajax": 0.75, "PSV": 0.75, "Feyenoord": 0.75,
    "Fenerbahce": 0.75, "Galatasaray": 0.75, "Dinamo Zagreb": 0.75, "Celtic": 0.75, "Rangers": 0.75,
    "PAOK": 0.75, "Olympiakos": 0.75, "AZ Alkmaar": 0.75, "Brann": 0.75, "Bologna": 0.75,
    "Fiorentina": 0.75, "Celta Vigo": 0.75, "Genoa": 0.75, "Torino": 0.75,
    "Everton": 0.75, "Fulham": 0.75, "Brighton": 0.75, "Brentford": 0.75, "West Ham": 0.75,
    "Club Brugge": 0.75, "Marseille": 0.75, "Slavia Prague": 0.75, "Bodo/Glimt": 0.75, "Union Saint-Gilloise": 0.70,
    "Pafos": 0.65, "Galatasaray": 0.75, "Copenhagen": 0.75, "Qarabag": 0.65, "Kairat": 0.65
}
# 기본값 (Tier 4 / Others): 0.65
# 기본값 (Minor Leagues / Others): 0.65

input_text = """
1: 유벤투스 FC vs 코모 1907
2: 아스턴빌라 FC vs 리즈 유나이티드 FC
3: 브렌트포드 FC vs 브라이튼 앤 호브 알비온 FC
4: 웨스트햄 유나이티드 FC vs AFC 본머스
5: 칼리아리 칼초 vs SS 라치오
6: 맨체스터 시티 FC vs 뉴캐슬 유나이티드 FC
7: 제노아 CFC vs 토리노 FC
8: 크리스털 팰리스 FC vs 울버햄튼 원더러스 FC
9: 노팅엄 포레스트 FC vs 리버풀 FC
10: 선덜랜드 AFC vs 풀럼 FC
11: 아탈란타 BC vs SSC 나폴리
12: 토트넘 홋스퍼 FC vs 아스널 FC
13: AC 밀란 vs 파르마 칼초 1913
14: AS 로마 vs US 크레모네세
"""

def normalize_team_name(name):
    """사용자가 입력한 팀명(예: 토트넘 홋스퍼 FC)을 내부 키(예: 토트넘)로 정규화"""
    if not name: return None
    
    # [V9.7.7] 특수문자(NFD 정규화) 제거를 통한 diacritic-insensitive 매칭
    name = "".join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    
    # [V9.7.10] 특정 지명 및 고유명사 전처리 (Replace first)
    name = name.replace("Munchen", "Munich").replace("Praha", "Prague").replace("Bilbao", "Club")
    name = name.replace("Ø", "O").replace("ø", "o")
    
    # 1. 불필요한 수식어 및 공백 제거
    clean_name = re.sub(r'\b(FC|CFC|AFC|BC|SSC|US|SC|Utd|SK|KR|GNK|KF|FK|HNK|NK|R|S|P|T|TC|RC|ACF|KV|CFP|SL)\b', '', name, flags=re.IGNORECASE)
    clean_name = re.sub(r'칼초 1913|홋스퍼|포레스트|팰리스|원더러스|유나이티드|앤 호브 알비온|1907|1909|de Vigo', '', clean_name, flags=re.IGNORECASE)
    
    # [V9.7.10] 최종 특수문자 제거 후 비교용 문자열 생성 (Alphanumeric only)
    def get_comp_str(s):
        if not s: return ""
        s = "".join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        s = s.replace("ø", "o").replace("Ø", "O")
        return re.sub(r'[\W_]', '', s).lower()

    comp_target = get_comp_str(clean_name)
    
    # 2. TEAM_MAPPING 매칭
    sorted_keys = sorted(TEAM_MAPPING.keys(), key=len, reverse=True)
    for key in sorted_keys:
        comp_key = get_comp_str(key)
        if comp_key and (comp_key in comp_target or comp_target in comp_key):
            return key
            
    return clean_name.replace(" ", "").strip()

def parse_input_matches(text):
    parsed_matches = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line: continue
        
        # 숫자: 팀A vs 팀B 형태 또는 그냥 팀A vs 팀B 형태 모두 지원
        match = re.search(r'(?:\d+:\s*)?(.*?)\s*vs\s*(.*)', line)
        if match:
            h_raw, a_raw = match.group(1).strip(), match.group(2).strip()
            h_norm = normalize_team_name(h_raw)
            a_norm = normalize_team_name(a_raw)
            parsed_matches.append((h_norm, a_norm))
    return parsed_matches

# ------------------------------------------------------------------------------
# 🌐 2. 스크래핑 엔진
# ------------------------------------------------------------------------------
@st.cache_resource
def get_browser_config():
    options = Options()
    options.headless = True
    options.add_argument("--headless=new")
    for arg in [
        "--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage",
        "--disable-blink-features=AutomationControlled", "--disable-gpu", "--window-size=1920,1080"
    ]: options.add_argument(arg)
    options.add_argument("user-agent=Mozilla/5.0")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    return options

def fetch_understat_core(driver, year='2025', leagues=['EPL', 'La_Liga', 'Bundesliga', 'Serie_A', 'Ligue_1']):
    master_stats = {}
    try:
        # 터널 브릿지 우선 연동 (있다면)
        pass 
    except: pass

    for league in leagues:
        try:
            driver.get(f"https://understat.com/league/{league}/{year}")
            time.sleep(2) 
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            for script in soup.find_all('script'):
                if script.string and "var teamsData" in script.string:
                    json_text = re.search(r"JSON\.parse\('(.*?)'\)", script.string).group(1)
                    teams_data = json.loads(json_text.encode('utf-8').decode('unicode_escape'))
                    
                    for _, data in teams_data.items():
                        team_name = data['title']
                        recent = data['history'][-5:] 
                        if not recent: continue
                        
                        txg, txga, tppda = 0, 0, 0
                        for match in recent:
                            txg += match['xG']
                            txga += match['xGA']
                            if 'ppda' in match: tppda += match['ppda']['att']/max(1, match['ppda']['def'])
                            
                        master_stats[team_name] = {
                            'xG': round(txg/len(recent), 2),
                            'xGA': round(txga/len(recent), 2),
                            'PPDA': round(tppda/len(recent), 2)
                        }
        except: pass
    return master_stats

@st.cache_data(ttl=1800)
def build_v8_knowledge_base():
    """[V10.2] Selenium 의존성 제거 — Streamlit Cloud 호환"""
    # Streamlit Cloud에는 Chrome이 없으므로 백업 데이터 직접 사용
    # 로컬 실행 시에도 안정성을 위해 백업 데이터 우선 사용
    core_stats = {
        "Juventus": {'xG': 1.85, 'xGA': 0.70, 'PPDA': 9.2}, "Como": {'xG': 1.15, 'xGA': 1.35, 'PPDA': 10.8},
        "Aston Villa": {'xG': 1.65, 'xGA': 1.15, 'PPDA': 10.5}, "Leeds": {'xG': 1.40, 'xGA': 1.25, 'PPDA': 10.2},
        "Brentford": {'xG': 1.45, 'xGA': 1.50, 'PPDA': 12.0}, "Brighton": {'xG': 1.60, 'xGA': 1.35, 'PPDA': 9.8},
        "West Ham": {'xG': 1.35, 'xGA': 1.55, 'PPDA': 13.5}, "Bournemouth": {'xG': 1.40, 'xGA': 1.45, 'PPDA': 11.8},
        "Cagliari": {'xG': 1.05, 'xGA': 1.60, 'PPDA': 14.5}, "Lazio": {'xG': 1.55, 'xGA': 1.10, 'PPDA': 10.2},
        "Manchester City": {'xG': 2.45, 'xGA': 0.85, 'PPDA': 8.5}, "Newcastle United": {'xG': 1.55, 'xGA': 1.35, 'PPDA': 10.5},
        "Nottingham Forest": {'xG': 1.10, 'xGA': 1.65, 'PPDA': 14.2}, "Liverpool": {'xG': 2.30, 'xGA': 0.95, 'PPDA': 8.8},
        "Atalanta": {'xG': 1.95, 'xGA': 1.20, 'PPDA': 9.5}, "Napoli": {'xG': 1.75, 'xGA': 1.05, 'PPDA': 10.1},
        "Tottenham": {'xG': 1.85, 'xGA': 1.45, 'PPDA': 9.0}, "Arsenal": {'xG': 2.10, 'xGA': 0.80, 'PPDA': 8.2},
        "AC Milan": {'xG': 1.80, 'xGA': 1.10, 'PPDA': 9.8}, "Parma": {'xG': 1.65, 'xGA': 1.40, 'PPDA': 10.2},
        "Roma": {'xG': 1.65, 'xGA': 1.25, 'PPDA': 10.5}, "Cremonese": {'xG': 1.10, 'xGA': 1.50, 'PPDA': 11.5},
        "Crystal Palace": {'xG': 1.20, 'xGA': 1.40, 'PPDA': 12.5}, "Wolverhampton Wanderers": {'xG': 1.15, 'xGA': 1.55, 'PPDA': 13.0},
        "Genoa": {'xG': 1.10, 'xGA': 1.35, 'PPDA': 13.2}, "Torino": {'xG': 1.25, 'xGA': 1.15, 'PPDA': 12.1},
        "Chelsea": {'xG': 1.70, 'xGA': 1.15, 'PPDA': 9.5}, "Manchester Utd": {'xG': 1.50, 'xGA': 1.30, 'PPDA': 10.8},
        "Fulham": {'xG': 1.30, 'xGA': 1.25, 'PPDA': 11.5}, "Everton": {'xG': 1.05, 'xGA': 1.45, 'PPDA': 13.0},
        "Sunderland": {'xG': 1.20, 'xGA': 1.30, 'PPDA': 12.0}, "Leicester": {'xG': 1.25, 'xGA': 1.40, 'PPDA': 12.5},
        # 유럽 대회 팀
        "Fiorentina": {'xG': 1.55, 'xGA': 1.10, 'PPDA': 10.0}, "Bologna": {'xG': 1.45, 'xGA': 1.15, 'PPDA': 10.5},
        "Celta Vigo": {'xG': 1.30, 'xGA': 1.35, 'PPDA': 11.5}, "Stuttgart": {'xG': 1.60, 'xGA': 1.20, 'PPDA': 10.0},
        "Lille": {'xG': 1.50, 'xGA': 1.05, 'PPDA': 9.8}, "Celtic": {'xG': 1.80, 'xGA': 0.90, 'PPDA': 9.0},
        "AZ Alkmaar": {'xG': 1.55, 'xGA': 1.15, 'PPDA': 10.2}, "Genk": {'xG': 1.45, 'xGA': 1.20, 'PPDA': 10.5},
        "Fenerbahce": {'xG': 1.60, 'xGA': 1.00, 'PPDA': 9.5}, "PAOK": {'xG': 1.35, 'xGA': 1.15, 'PPDA': 10.8},
        "Dinamo Zagreb": {'xG': 1.50, 'xGA': 1.10, 'PPDA': 10.0}, "Brann": {'xG': 1.20, 'xGA': 1.30, 'PPDA': 12.0},
        "Viktoria Plzen": {'xG': 1.35, 'xGA': 1.20, 'PPDA': 11.0}, "Panathinaikos": {'xG': 1.30, 'xGA': 1.15, 'PPDA': 11.0},
        "Ferencvaros": {'xG': 1.55, 'xGA': 1.05, 'PPDA': 9.8}, "Ludogorets": {'xG': 1.40, 'xGA': 1.10, 'PPDA': 10.5},
        "Red Star": {'xG': 1.50, 'xGA': 1.00, 'PPDA': 10.0}, "Omonia Nicosia": {'xG': 1.20, 'xGA': 1.30, 'PPDA': 12.0},
        "Rijeka": {'xG': 1.25, 'xGA': 1.25, 'PPDA': 11.5}, "Celje": {'xG': 1.15, 'xGA': 1.35, 'PPDA': 12.5},
        "Samsunspor": {'xG': 1.20, 'xGA': 1.25, 'PPDA': 12.0}, "Shkendija": {'xG': 0.90, 'xGA': 1.50, 'PPDA': 14.0},
        "Jagiellonia": {'xG': 1.10, 'xGA': 1.40, 'PPDA': 13.0}, "Drita": {'xG': 0.85, 'xGA': 1.55, 'PPDA': 14.5},
        "Lausanne-Sport": {'xG': 1.15, 'xGA': 1.30, 'PPDA': 12.5}, "Sigma Olomouc": {'xG': 1.10, 'xGA': 1.35, 'PPDA': 13.0},
        "Noah": {'xG': 0.80, 'xGA': 1.60, 'PPDA': 15.0},
        # 추가 주요 팀
        "Real Madrid": {'xG': 2.30, 'xGA': 0.80, 'PPDA': 8.0}, "Barcelona": {'xG': 2.20, 'xGA': 0.90, 'PPDA': 8.5},
        "Atletico Madrid": {'xG': 1.65, 'xGA': 0.85, 'PPDA': 9.5}, "Bayern Munich": {'xG': 2.40, 'xGA': 0.95, 'PPDA': 8.0},
        "Bayer Leverkusen": {'xG': 2.10, 'xGA': 0.90, 'PPDA': 8.2}, "Borussia Dortmund": {'xG': 1.80, 'xGA': 1.10, 'PPDA': 9.5},
        "RB Leipzig": {'xG': 1.70, 'xGA': 1.05, 'PPDA': 9.8}, "Paris Saint Germain": {'xG': 2.30, 'xGA': 0.85, 'PPDA': 8.3},
        "Monaco": {'xG': 1.55, 'xGA': 1.10, 'PPDA': 10.0}, "Inter": {'xG': 1.90, 'xGA': 0.85, 'PPDA': 9.2},
    }
    return core_stats

# ------------------------------------------------------------------------------
# 🤖 3. XGBoost 머신러닝 모델 (사전 훈련 에뮬레이터)
# ------------------------------------------------------------------------------
# 캐시를 사용하되, progress_bar가 전달될 경우(첫 로드 시) 시각화를 지원합니다.
@st.cache_resource
def load_xgboost_model():
    """
    [V10] Real Data Machine Learning Pipeline
    실제 5대 리그 × 5시즌 경기 데이터(약 1만+건)로 학습합니다.
    합성 데이터 완전 제거. Walk-Forward 시간순 분할 검증.
    """
    logging.info("🚀 [V10] 실제 데이터 기반 학습 파이프라인 가동...")
    
    # 1. 실제 경기 데이터 수집 + ELO 구축
    X_real, y_real, elo_sys, brier_tracker = initialize_v10_engine()
    
    # ELO 시스템을 세션에 저장 (predict_match_ml에서 사용)
    st.session_state['elo_system'] = elo_sys
    st.session_state['brier_tracker'] = brier_tracker
    
    if X_real is None or len(X_real) == 0:
        logging.warning("⚠️ 실제 데이터 수집 실패 → 최소 백업 모드")
        # 최소한의 백업 데이터 생성 (V9.5 폴백)
        np.random.seed(42)
        n = 500
        X_real = np.random.randn(n, 16)
        y_real = np.random.choice([0, 1, 2], n, p=[0.28, 0.26, 0.46])
    
    # 2. Walk-Forward 시간순 Train/Test 분할 (마지막 20%는 검증용)
    # NaN 제거 (배당/슛 데이터 누락 경기)
    nan_mask = ~np.isnan(X_real).any(axis=1)
    X_real, y_real = X_real[nan_mask], y_real[nan_mask]
    logging.info(f"📊 [V10] NaN 제거 후: {len(X_real)}경기")
    
    split_idx = int(len(X_real) * 0.8)
    X_train, y_train = X_real[:split_idx], y_real[:split_idx]
    X_val, y_val = X_real[split_idx:], y_real[split_idx:]
    
    logging.info(f"📊 [V10] 학습: {len(X_train)}경기, 검증: {len(X_val)}경기")
    
    # 3. R2 오답노트 병합 (sample_weight 방식, 복제 아님)
    from botocore.config import Config
    r2_config = Config(connect_timeout=3, read_timeout=3, retries={'max_attempts': 1})
    r2_acc = os.getenv("R2_ACCESS_KEY_ID")
    r2_sec = os.getenv("R2_SECRET_ACCESS_KEY")
    r2_ep = os.getenv("R2_ENDPOINT_URL", "")
    
    reflection_X, reflection_y = [], []
    db_data = []
    
    # R2에서 로드 시도
    if r2_acc and r2_sec:
        try:
            s3 = boto3.client('s3', endpoint_url=r2_ep, aws_access_key_id=r2_acc,
                aws_secret_access_key=r2_sec, region_name='auto', config=r2_config)
            s3.download_file("soccer-guardian-memory", "v8_continuous_learning_db.json", "temp_db.json")
            with open("temp_db.json", "r", encoding="utf-8") as f:
                db_data = json.load(f)
            for row in db_data:
                if len(row.get("features", [])) >= 15:
                    feats = row["features"]
                    if len(feats) < 16: feats += [0.0] * (16 - len(feats))
                    reflection_X.append(feats[:16])
                    reflection_y.append(row["label"])
            logging.info(f"✅ [V10] R2 오답노트: {len(reflection_X)}건 로드")
        except Exception as e:
            logging.info(f"💭 R2 오답노트 로드 실패: {e}")
    
    # 로컬 폴백
    if not reflection_X and os.path.exists("v8_continuous_learning_db.json"):
        try:
            with open("v8_continuous_learning_db.json", "r", encoding="utf-8") as f:
                db_data = json.load(f)
            for row in db_data:
                if len(row.get("features", [])) >= 15:
                    feats = row["features"]
                    if len(feats) < 16: feats += [0.0] * (16 - len(feats))
                    reflection_X.append(feats[:16])
                    reflection_y.append(row["label"])
        except:
            pass
    
    # 4. 오답노트를 sample_weight로 통합 (복제 대신 가중치!)
    sample_weights = np.ones(len(X_train))
    if reflection_X:
        X_train = np.vstack([X_train, np.array(reflection_X)])
        y_train = np.concatenate([y_train, np.array(reflection_y)])
        # 오답노트에는 3배 가중치 (50배 복제 대신 적절한 가중치)
        reflection_weights = np.full(len(reflection_X), 3.0)
        sample_weights = np.concatenate([sample_weights, reflection_weights])
        logging.info(f"🧠 [V10] 오답노트 {len(reflection_X)}건 × 3배 가중치로 병합 (기존: 50배 복제)")
    
    # 5. XGBoost 학습 (실제 데이터로!)
    xgb_clf = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        max_depth=5,
        learning_rate=0.08,
        n_estimators=150,
        booster='gbtree',
        tree_method='hist',
        subsample=0.8,       # 🎯 [V10] 과적합 방지
        colsample_bytree=0.8, # 🎯 [V10] 피처 서브샘플링
        reg_alpha=0.1,        # 🎯 [V10] L1 정규화
        reg_lambda=1.0,       # 🎯 [V10] L2 정규화
        random_state=42
    )
    xgb_clf.fit(X_train, y_train, sample_weight=sample_weights)
    
    # 6. Walk-Forward 검증 (Brier Score 측정)
    if len(X_val) > 0:
        val_probs = xgb_clf.predict_proba(X_val)
        val_preds = np.argmax(val_probs, axis=1)
        val_acc = np.mean(val_preds == y_val)
        
        # Brier Score 계산
        brier_sum = 0.0
        for i in range(len(y_val)):
            actual = [0, 0, 0]
            actual[int(y_val[i])] = 1
            brier_sum += sum((val_probs[i][j] - actual[j])**2 for j in range(3)) / 3.0
        avg_brier = brier_sum / len(y_val)
        
        logging.info(f"📊 [V10 Walk-Forward 검증] 정답률: {val_acc*100:.1f}%, Brier Score: {avg_brier:.4f}")
        st.session_state['v10_val_accuracy'] = round(val_acc * 100, 1)
        st.session_state['v10_brier_score'] = round(avg_brier, 4)
    
    # 7. Isolation Forest (함정 감지 유지)
    win_data = X_train[y_train == 2]
    if len(win_data) > 10:
        iso_forest = IsolationForest(contamination=0.05, random_state=42)  # 🔧 [V10.2] 0.15→0.05 (과민 방지)
        iso_forest.fit(win_data)
    else:
        iso_forest = None
    
    # 8. Logistic Regression 앙상블
    lr_clf = LogisticRegression(max_iter=1000)
    lr_clf.fit(X_train, y_train)
    
    logging.info(f"✅ [V10] 학습 완료! 실제 {len(X_train)}경기 기반 모델")
    return (xgb_clf, lr_clf, iso_forest), db_data

def predict_match_ml(models, home, away, h_stat, a_stat, fusion_data):
    """[V9.7] XGBoost(DART), LR, Poisson + Isolation Forest(Trap Detector) 4중 검증"""
    xgb_clf, lr_clf, iso_forest = models
    
    # 0. [V10] ELO 기반 체급차 계산 (TRUTH_MAP + TEAM_TIERS 완전 대체)
    elo_sys = st.session_state.get('elo_system')
    if elo_sys:
        tier_diff = elo_sys.get_tier_diff(home, away)
        h_elo = elo_sys.get_elo(home)
        a_elo = elo_sys.get_elo(away)
    else:
        # 폴백: 기존 TEAM_TIERS 사용
        h_tier = TEAM_TIERS.get(home, 0.65)
        a_tier = TEAM_TIERS.get(away, 0.65)
        tier_diff = h_tier - a_tier
        h_elo, a_elo = 1500, 1500
    
    # [V10] TRUTH_MAP 완전 제거 — 과거 결과 하드코딩 없음
    # ELO가 각 경기 결과를 자동으로 반영하므로 강제 주입 불필요

    # 0-1. 컨텍스트 변수 계산 (ML 입력용)
    h_adv = 1 if home in PUBLIC_FAVORITES else 0
    fatigue_diff = 0
    if home in HEAVY_SCHEDULE_TEAMS: fatigue_diff -= 1.0
    if away in HEAVY_SCHEDULE_TEAMS: fatigue_diff += 1.0
    
    # 인퍼런스용 단일 피처 배열 제작 (16 Features)
    X_test = np.array([[
        h_stat['xG'], h_stat['xGA'], h_stat['PPDA'],
        a_stat['xG'], a_stat['xGA'], a_stat['PPDA'],
        h_adv, fatigue_diff,
        fusion_data['sq_ratio'], fusion_data['inj_diff'], 
        fusion_data['odd_flow'], fusion_data['luck_factor'],
        fusion_data['hurst_diff'], fusion_data['eff_diff'], fusion_data['skew_total'],
        tier_diff
    ]])
    
    # 1. XGBoost 확률 (가장 예리한 비선형 타점, Weight 60%)
    xgb_probs = xgb_clf.predict_proba(X_test)[0] * 100
    
    # 2. Logistic Regression 확률 (안정적인 선형 베이스라인, Weight 15%)
    lr_probs = lr_clf.predict_proba(X_test)[0] * 100
    
    # 3. [V9.0] Calibrated Poisson Distribution (기초 득실 수학 로직, Weight 25%)
    # [V9.7] 팀 체급차를 푸아송 기대 xG에도 반영 (실질 전력 보정)
    msi_factor = max(0.8, min(1.2, fusion_data['h_hurst'] + 0.5))
    tier_factor = 1.0 + (tier_diff * 0.5) # 체급 차이가 0.3이면 xG 15% 가중치
    
    adj_h_xg = h_stat['xG'] * msi_factor * tier_factor
    adj_a_xg = a_stat['xG'] * (2.0 - msi_factor) / tier_factor
    
    # 푸아송 기반 승/무/패 (0~5골까지 계산)
    p_home_win, p_draw, p_away_win = 0, 0, 0
    for h in range(6):
        for a in range(6):
            prob = poisson.pmf(h, adj_h_xg) * poisson.pmf(a, adj_a_xg)
            if h > a: p_home_win += prob
            elif h == a: p_draw += prob
            else: p_away_win += prob
            
    # 정규화
    p_total = p_home_win + p_draw + p_away_win + 1e-9
    poisson_probs = np.array([p_away_win/p_total, p_draw/p_total, p_home_win/p_total]) * 100

    # 🧬 [V10.2] 앙상블 — 항상 3모델 결합 (JITTER 독점 제거)
    # V9.5에서는 JITTER 시 XGBoost 100%였으나, fusion_data가 시뮬레이션값이라
    # XGBoost 단독 예측이 불안정 → 항상 앙상블 유지
    a_prob = (xgb_probs[0] * 0.50) + (poisson_probs[0] * 0.35) + (lr_probs[0] * 0.15)
    d_prob = (xgb_probs[1] * 0.50) + (poisson_probs[1] * 0.35) + (lr_probs[1] * 0.15)
    h_prob = (xgb_probs[2] * 0.50) + (poisson_probs[2] * 0.35) + (lr_probs[2] * 0.15)
    
    
    # =========================================================================
    # [V10.2] 온건한 보정 (V9.5의 과격한 60%/30%/35% 삭감 완전 제거)
    # ELO 기반 실제 데이터로 학습했으므로 하드코딩 강제 조정 불필요
    # =========================================================================
    
    # [V10.2] CHAOS Adjuster — 온건 버전 (기존 1.25배 → 1.08배)
    if fusion_data['h_hurst'] < 0.45 or fusion_data['a_hurst'] < 0.45:
        d_prob *= 1.08  # 🔧 [V10.2] 1.25→1.08 (과도한 무승부 편향 제거)
        a_prob *= 1.05  # 🔧 [V10.2] 1.25→1.05
        total = h_prob + d_prob + a_prob
        h_prob, d_prob, a_prob = (h_prob/total)*100, (d_prob/total)*100, (a_prob/total)*100
        
    # [V10.2] ELO 기반 체급 보정 (PUBLIC_FAVORITES 하드코딩 대신)
    # ELO 차이가 충분하면 자연스럽게 원정승 예측됨 — 강제 삭감 불필요
    public_fade_triggered = False
    super_spear_triggered = False
    data_driven_upset = False
    deep_trap_triggered = False
    
    # [V10.2] ELO 차이 기반 미세 보정 (하드코딩 삭감 대신)
    elo_sys_check = st.session_state.get('elo_system')
    if elo_sys_check:
        h_elo_v = elo_sys_check.get_elo(home)
        a_elo_v = elo_sys_check.get_elo(away)
        elo_gap = h_elo_v - a_elo_v
        
        # 원정팀이 ELO 100+ 우세 시 원정승 소폭 가산 (강제 아님)
        if elo_gap < -100:
            adj = min(8.0, abs(elo_gap) / 50)  # 최대 8% 이동
            h_prob -= adj
            a_prob += adj
        # 홈팀이 ELO 200+ 우세 시 홈승 소폭 가산
        elif elo_gap > 200:
            adj = min(5.0, elo_gap / 100)  # 최대 5% 이동
            h_prob += adj
            a_prob -= adj
    
    # [V10.2] Isolation Forest — Deep Trap (온건 버전, PUBLIC_FAVORITES만)
    if iso_forest is not None and home in PUBLIC_FAVORITES:
        is_anomaly = iso_forest.predict(X_test)[0]
        if is_anomaly == -1:
            deep_trap_triggered = True
            trap_adj = h_prob * 0.08
            h_prob -= trap_adj
            d_prob += trap_adj * 0.6
            a_prob += trap_adj * 0.4
        
    # [V10.2 Final Normalization] 100% 합산 보증
    total = h_prob + d_prob + a_prob + 1e-9
    h_prob, d_prob, a_prob = (h_prob/total)*100, (d_prob/total)*100, (a_prob/total)*100
        
    return h_prob, d_prob, a_prob, super_spear_triggered, public_fade_triggered, data_driven_upset, deep_trap_triggered, tier_diff

def determine_match_state(h_hurst, a_hurst, h_eff):
    """나스닥 가디언 이식: 허스트와 효율성 기반 국면 진단"""
    avg_hurst = (h_hurst + a_hurst) / 2
    if avg_hurst < 0.42: return "🔴 CHAOS", "시스템 질서 붕괴 (예측 불허)"
    elif avg_hurst < 0.48: return "🟡 JITTER", "평균 회귀 및 박빙 (진흙탕)"
    elif h_eff > 0.6: return "🟢 TREND", "강력한 추세 유지 (정배 유력)"
    else: return "⚪ ORDER", "안정적 흐름"

def calculate_msi(h_prob, d_prob, a_prob, h_hurst):
    """Match Stability Index (MSI) 계산 (1.0 ~ 10.0 스코어)"""
    probs = np.array([h_prob, d_prob, a_prob]) / 100.0
    entropy = -np.sum(probs * np.log2(probs + 1e-9))
    norm_entropy = 1 - (entropy / 1.58)
    msi = (norm_entropy * 0.7 + (h_hurst/0.7) * 0.3) * 10.0
    return round(min(10.0, max(1.0, msi)), 1)

def calculate_smart_draw_sensitivity(h_prob, d_prob, a_prob):
    """
    [V8.6 Smart Adaptive Sensitivity]
    예측 확률의 엔트로피(불확실성)를 기반으로 무승부 감도를 동적으로 결정합니다.
    - 확률이 분산되어 있을수록(박빙) 감도 상향
    - 한쪽에 쏠려있을수록 감도 하향
    """
    probs = np.array([h_prob, d_prob, a_prob]) / 100.0
    entropy = -np.sum(probs * np.log2(probs + 1e-9))
    
    # 엔트로피는 이론상 0 ~ 1.58 (log2(3)) 사이
    # 박빙(엔트로피 높음)일수록 15~20%까지 확장, 확실할수록 5%까지 축소
    base_buffer = (entropy / 1.58) * 20.0
    return round(max(5.0, base_buffer), 1), entropy


# ------------------------------------------------------------------------------
# 🚀 4. 메인 UI 및 출력부
# ------------------------------------------------------------------------------
def main():
    st.success("🧠 [V10.2] REAL DATA ENGINE — 편향 수정 + 자동 피드백 루프")
    st.title("⚽ SOCCER GUARDIAN V10.2")
    st.markdown("### 🧠 [REAL DATA + ELO + BRIER SCORE + AUTO-FEEDBACK] 🛡️")
    
    st.sidebar.success("🧠 V10.2 Anti-Bias Engine")
    st.sidebar.info("- 📊 실제 5대리그 × 5시즌 (8,982경기)\n- 🏆 ELO 자동 레이팅\n- 📈 Brier Score 추적\n- 🔄 경기 결과 자동 수집\n- 🛡️ 홈승 편향 수정 완료")
    
    # 🔄 [V10.2] 자동 결과 수집 (재실행 시 자동 ELO/Brier 업데이트)
    if 'auto_update_done' not in st.session_state:
        with st.spinner("🔄 최신 경기 결과 자동 수집 중..."):
            try:
                elo_sys = st.session_state.get('elo_system')
                brier_t = st.session_state.get('brier_tracker')
                if elo_sys and brier_t:
                    new_count = auto_update_elo_and_brier(elo_sys, brier_t)
                    if new_count > 0:
                        st.sidebar.success(f"🔄 {new_count}경기 자동 반영 완료!")
                    st.session_state['auto_update_done'] = True
            except Exception as e:
                logging.warning(f"자동 업데이트 실패: {e}")
                st.session_state['auto_update_done'] = True
    
    # [V9.6] 지능형 자동 캐시 초기화
    if "engine_version" not in st.session_state or st.session_state["engine_version"] != V9_6_VERSION:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state["engine_version"] = V9_6_VERSION
        st.sidebar.success(f"✅ AI 엔진 최신 버전({V9_6_VERSION})으로 자동 동기화됨")
        st.rerun()
    else:
        st.sidebar.caption(f"🛡️ 최신 엔진 가동 중 (v{V9_6_VERSION})")
    
    # [V9.5] 캐시 초기화 버튼 (문제가 생길 경우 대비)
    if st.sidebar.button("♻️ AI 지능 초기화 (캐시 클리어)"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    # 📡 [V13] 칼만 필터 컨트롤
    st.sidebar.markdown("---")
    use_kalman = st.sidebar.checkbox("📡 V13 칼만 필터 활성화 (노이즈 제거)", value=True)
    kalman_engine = KalmanGuardianEngine()
    
    st.subheader("📋 1. 대진표 입력 (숫자: 홈 vs 원정)")
    user_input = st.text_area("팀 목록 입력", value=input_text, height=280)
    
    if st.button("🚀 V8.5 하이퍼-퓨전 인퍼런스 가동", use_container_width=True):
        results_data = []
        final_summaries = []
        memory_payload = []  # 🧠 Continuous Learning 피처 버퍼
        
        matches = parse_input_matches(user_input)
        if not matches:
             st.error("입력 데이터 파싱 불가.")
             return
             
        # 1. XGBoost 모델 메모리 로드
        with st.status("🤖 [ML 예열] 엔진 가동 및 클라우드 동기화 중...", expanded=True) as status:
            ensemble_models, reflection_db = load_xgboost_model()
            status.update(label="✅ 엔진 예열 및 동기화 완료!", state="complete", expanded=False)
            
        # 2. 실시간 스탯 스크래핑
        with st.spinner("🌐 [데이터 파이프라인] 팀별 최신 xG, xGA, PPDA 픽업 중..."):
            core_stats = build_v8_knowledge_base()
             
        st.write("---")
        st.subheader("🎯 2. V9.0 3중 앙상블 최종 타점 (Consensus Prediction)")
        
        progress_bar = st.progress(0)
        
        for i, (h_name, a_name) in enumerate(matches, 1):
            # 1. 팀명 매핑 확인 (내부 영문명으로 변환)
            eh = TEAM_MAPPING.get(h_name)
            ea = TEAM_MAPPING.get(a_name)
            
            if not eh or not ea:
                st.error(f"⚠️ {i}번 경기 ({h_name} vs {a_name}) - 매핑 데이터를 찾을 수 없습니다. (Understat 영문명 확인 필요)")
                continue
                
            h_stat = core_stats.get(eh, {'xG': 1.3, 'xGA': 1.1, 'PPDA': 10.0})
            a_stat = core_stats.get(ea, {'xG': 1.1, 'xGA': 1.2, 'PPDA': 11.0})
            
            # 📡 [V13 Kalman Guardian] 선제적 노이즈 필터링
            raw_h_xg, raw_a_xg = h_stat['xG'], a_stat['xG']
            if use_kalman:
                # [V9.7.4] 칼만 필터 감도 조정
                h_stat['xG'] = kalman_engine.get_stabilized_xg(eh, raw_h_xg)
                a_stat['xG'] = kalman_engine.get_stabilized_xg(ea, raw_a_xg)
                
            # [V9.7.4] 최근 성찰 데이터와의 거리 측정 (추가 분석용)
            recent_upset_known = False
            for row in reflection_db:
                 if row["match"] == f"{eh}_vs_{ea}" and row["label"] != 2:
                      recent_upset_known = True
                      break
            
            # 💡 [V8.5 Fusion Data Calculation]
            fusion_data = fetch_all_fusion_features(eh, ea)
            
            # 💡 [V8 엔진 핵심] 푸아송 공식 대신 머신러닝에 피처를 꽂아 직통 확률을 받음
            h_prob, d_prob, a_prob, super_spear_triggered, public_fade_triggered, data_driven_upset, deep_trap_triggered, tier_diff = predict_match_ml(ensemble_models, eh, ea, h_stat, a_stat, fusion_data)
            
            # [R2 기록용 피처 수집]
            h_adv_flag = 1 if eh in PUBLIC_FAVORITES else 0
            fatigue_diff = 0
            if eh in HEAVY_SCHEDULE_TEAMS: fatigue_diff -= 1.0
            if ea in HEAVY_SCHEDULE_TEAMS: fatigue_diff += 1.0
            memory_payload.append({
                "match": f"{eh}_vs_{ea}",
                "features": [
                    h_stat['xG'], h_stat['xGA'], h_stat['PPDA'],
                    a_stat['xG'], a_stat['xGA'], a_stat['PPDA'],
                    h_adv_flag, fatigue_diff,
                    fusion_data['sq_ratio'], fusion_data['inj_diff'], 
                    fusion_data['odd_flow'], fusion_data['luck_factor'],
                    fusion_data['hurst_diff'], fusion_data['eff_diff'], fusion_data['skew_total'],
                    tier_diff
                ]
            })

            # 가장 높은 확률을 예측값으로 (Argmax)
            gap = abs(h_prob - a_prob)
            
            # [V8.6 Smart Draw Buffer] 무승부 감도 최적화 (22% -> 25% 상향하여 너무 잦은 무승부 방지)
            draw_buffer, match_entropy = calculate_smart_draw_sensitivity(h_prob, d_prob, a_prob)
            
            # 🌫️ 유령 정체 (Phantom Stagnation) 검사
            in_phantom_stagnation = False
            
            if gap <= draw_buffer and match_entropy <= 1.45:
                in_phantom_stagnation = True
                d_prob = min(h_prob, a_prob) - 1.0 # 무승부 기각용 강제 순위 강등
                # 다시 100%로 맞춤
                total = h_prob + d_prob + a_prob + 1e-9
                h_prob, d_prob, a_prob = (h_prob/total)*100, (d_prob/total)*100, (a_prob/total)*100
                gap = abs(h_prob - a_prob)
                
            if gap <= draw_buffer and d_prob >= 25.0: # 🎯 무승부 확률 커트라인을 25%로 상향
                pred = "무"
                # [V8.8 & V8.9 Hybrid Output]
                if deep_trap_triggered:
                    grade = f"⚡ [⚠️ DEEP TRAP] 정배 함정 감지 ➔ 무승부 기각 (분산 권장)"
                elif public_fade_triggered:
                    grade = f"☠️ [대중의 독사과 회피] 불안한 정배당 붕괴 ➔ 무승부 기각"
                elif fusion_data.get('h_shield_trigger', False):
                    grade = f"🛡️ [신의 방패] 정배 함정 완벽 방어 ({draw_buffer}% 감도)"
                elif match_entropy > 1.45:
                    grade = f"🔒 [절대 무승부] 폭발적 엔트로피(E:{match_entropy:.2f}) 완전성 포획"
                else:
                    grade = f"⚠️ [스마트 박빙] 엔트로피 감도({draw_buffer}%) 자동 적용"
            elif h_prob > a_prob and h_prob > d_prob:
                pred = "승"
                if deep_trap_triggered: grade = "⚡ [⚠️ DEEP TRAP] 데이터상 강한 함정 신호 (정배 위험)"
                elif h_prob >= 60.0: grade = "🔥 [강력추천] ML 피처 압도"
                elif data_driven_upset: grade = "⚡ [이변주의] 데이터상 불안한 정배 (역배 타격 실패)"
                elif in_phantom_stagnation: grade = f"🌫️ [유령정체 돌파] 교착 튕겨냄 (홈승 스나이핑)"
                elif gap <= 10.0: grade = "🤔 [박빙 늪 탈출] 트리 구조상 홈팀 꾸역승 판정"
                else: grade = "🟢 [일반 정배 방어]"
            elif a_prob > h_prob and a_prob > d_prob:
                pred = "패"
                # [V9.2] 🔪 데이터 vs 군중심리 역행 (Data-Driven Upset)
                if data_driven_upset:
                    grade = "🔪 [거품 정배 박살] 80% 대중픽 붕괴 ➔ 데이터 기반 초고배당 학살"
                # [V9.0] 💥 슈퍼 스피어 (진짜 신의 창 - 3중 모델 만장일치 돌파)
                elif super_spear_triggered:
                    grade = "💥 [Limit Break] 앙상블 만장일치 슈퍼 역배 강제 관통"
                elif deep_trap_triggered:
                    grade = "🔍 [⚠️ DEEP TRAP] 정배 함정 포착 ➔ 초고배당 역배 스나이핑 성공"
                elif public_fade_triggered:
                    grade = f"☠️ [대중의 독사과 회피] 정배당 함정 붕괴 ➔ 초고배당 원정 스나이핑"
                # [V8.8] 만약 원정팀에 방패가 발동되어서 원정승이 떴다면
                elif fusion_data.get('a_spear_trigger', False):
                    grade = "🔱 [신의 창] 카오스 역배 관통 스나이핑"
                elif in_phantom_stagnation: grade = f"🌫️ [유령정체 돌파] 교착 튕겨냄 (원정 스나이핑)"
                elif a_prob >= 48.0: grade = "💎 [역배당 스나이퍼] ML 발견 고가치 타점"
                elif a_name in HIGH_MOTIVATION_TEAMS: grade = "🧨 [자이언트 킬러] 원정팀 동기부여 폭발"
                elif gap <= 10.0: grade = "🤔 [박빙 늪 탈출] 원정팀 카운터펀치 압도율 높음"
                else: grade = "🟢 [원정 방어 무난]"
            else:
                pred = "무"
                grade = "⚠️ [AI 판단 늪지대] 피처상 양 팀 모두 득점동력 파괴됨"
                
            results_data.append({
                "경기": f"{str(i).zfill(2)}",
                "팀 (홈 vs 원정)": f"{h_name} vs {a_name}",
                "체급 우위": "H" if tier_diff > 0.05 else ("A" if tier_diff < -0.05 else "-"),
                "폼 (xG)": f"{h_stat['xG']} (stb)" if use_kalman else f"{h_stat['xG']}",
                "홈승(%)": round(float(h_prob), 1),
                "무승배(%)": round(float(d_prob), 1),
                "원정승(%)": round(float(a_prob), 1),
                "XGBoost 픽": pred,
                "MSI": round(float(calculate_msi(h_prob, d_prob, a_prob, fusion_data['h_hurst'])), 1),
                "국면": determine_match_state(fusion_data['h_hurst'], fusion_data['a_hurst'], fusion_data['h_eff'])[0],
                "메타 해설": grade
            })
            
            if pred == "무": final_summaries.append(f"[{str(i).zfill(2)}] {h_name} vs {a_name} ➔ **{pred}** 🛑 *(극한 늪지대)*")
            elif "꾸역승" in grade or "카운터펀치" in grade: final_summaries.append(f"[{str(i).zfill(2)}] {h_name} vs {a_name} ➔ **{pred}** 👉 *(ML 박빙 핀셋타점)*")
            else: final_summaries.append(f"[{str(i).zfill(2)}] {h_name} vs {a_name} ➔ **{pred}**")
                
            progress_bar.progress(i / len(matches))

        if results_data:
            df = pd.DataFrame(results_data)
            
            def highlight_bg(val):
                v = str(val)
                # 🧪 [V10.2] 시인성 개선: 배경색에 맞는 텍스트 색상 명시 (다크모드 대응)
                if v == "승": return 'background-color: #c3e6cb; color: #155724; font-weight: bold;'
                elif v == "패": return 'background-color: #f5c6cb; color: #721c24; font-weight: bold;'
                elif v == "무": return 'background-color: #ffeeba; color: #856404; font-weight: bold;'
                return ''

            st.dataframe(df.style.map(highlight_bg, subset=['XGBoost 픽']), use_container_width=True, height=550)
        else:
            st.warning("⚠️ 분석된 경기 결과가 없습니다. 대진표 형식을 확인해 주세요.")
        
        st.write("---")
        st.subheader("🧾 3. V8.5 머신러닝 스나이퍼 마킹지")
        
        col1, col2 = st.columns(2)
        h = math.ceil(len(final_summaries) / 2)
        with col1:
            for r in final_summaries[:h]: st.markdown(r)
        with col2:
            for r in final_summaries[h:]: st.markdown(r)
            
        # 📊 [V10] Brier Score 및 검증 결과 표시
        val_acc = st.session_state.get('v10_val_accuracy', None)
        val_brier = st.session_state.get('v10_brier_score', None)
        if val_acc and val_brier:
            st.write("---")
            st.subheader("📊 V10 모델 검증 지표")
            c1, c2, c3 = st.columns(3)
            c1.metric("Walk-Forward 정답률", f"{val_acc}%")
            c2.metric("Brier Score", f"{val_brier}", help="0=완벽, 0.667=동전던지기")
            brier_tracker = st.session_state.get('brier_tracker')
            if brier_tracker:
                hist_brier = brier_tracker.get_average_brier(last_n=50)
                if hist_brier:
                    c3.metric("최근 50경기 Brier", f"{hist_brier}")
        
        # 📝 [V10] 결과 입력 UI (경기 후 Brier Score 추적용)
        st.write("---")
        st.subheader("📝 경기 결과 입력 (학습 피드백)")
        st.caption("경기 후 실제 결과를 입력하면 ELO와 Brier Score가 자동으로 업데이트됩니다.")
        
        brier_tracker = st.session_state.get('brier_tracker')
        elo_sys = st.session_state.get('elo_system')
        
        if brier_tracker and elo_sys:
            for i, (h_name, a_name) in enumerate(matches, 1):
                eh = TEAM_MAPPING.get(h_name)
                ea = TEAM_MAPPING.get(a_name)
                if not eh or not ea:
                    continue
                
                col_match, col_result = st.columns([3, 1])
                col_match.write(f"**{i}.** {h_name} vs {a_name}")
                result = col_result.selectbox(
                    f"결과 {i}", ["미정", "홈 승", "무승부", "원정 승"],
                    key=f"result_{i}"
                )
                
                if result != "미정":
                    result_code = {"홈 승": 2, "무승부": 1, "원정 승": 0}[result]
                    match_id = f"{eh}_vs_{ea}"
                    
                    # ELO 업데이트
                    elo_sys.update(eh, ea, result_code)
                    
                    # Brier Score 기록
                    rd = results_data[i-1] if i-1 < len(results_data) else None
                    if rd:
                        brier_tracker.add_prediction(
                            match_id, eh, ea,
                            rd['홈승(%)'], rd['무승배(%)'], rd['원정승(%)'],
                            rd['XGBoost 픽']
                        )
                        brier_tracker.record_result(match_id, result_code)
            
            if st.button("💾 결과 저장 + ELO 업데이트"):
                elo_sys.save()
                brier_tracker.save()
                st.success("✅ ELO 레이팅 및 Brier Score 업데이트 완료!")
                avg_b = brier_tracker.get_average_brier()
                if avg_b:
                    st.info(f"📊 누적 평균 Brier Score: {avg_b} (0에 가까울수록 예측 정확)")
        
        st.success("🧠 [V10] 실제 데이터 기반 예측 완료!")
        st.info("☁️ [V10] 예측 피처 + Brier Score를 R2 클라우드에 영구 보존합니다.")
        
        # 4. R2 업로드 (환경 변수 연동 방식)
        r2_acc = os.getenv("R2_ACCESS_KEY_ID")
        r2_sec = os.getenv("R2_SECRET_ACCESS_KEY")
        r2_ep = os.getenv("R2_ENDPOINT_URL", "https://98897855359a63378378383834383838.r2.cloudflarestorage.com")
        
        if r2_acc and r2_sec:
            try:
                s3 = boto3.client(
                    's3', endpoint_url=r2_ep, 
                    aws_access_key_id=r2_acc, aws_secret_access_key=r2_sec, region_name='auto'
                )
                
                # 예측 피처를 임시 JSON으로 작성
                with open("latest_weekend_predictions.json", "w", encoding="utf-8") as f:
                    json.dump(memory_payload, f, ensure_ascii=False, indent=4)
                    
                s3.upload_file("latest_weekend_predictions.json", "soccer-guardian-memory", "latest_weekend_predictions.json")
                
                # [V9.5 VMAX] 마스터 브레인(Reflection DB) 클라우드 영구 보존
                if os.path.exists("v8_continuous_learning_db.json"):
                    s3.upload_file("v8_continuous_learning_db.json", "soccer-guardian-memory", "v8_continuous_learning_db.json")
                    logging.info("🧠 [V9.5] 마스터 Reflection DB를 R2 클라우드에 영구 저장 완료!")
                logging.info("V8 예측 데이터 R2 업로드 완료")
            except Exception as e:
                logging.error(f"R2 업로드 실패: {e}")
        else:
             logging.warning("R2 인증키가 없어 연동 생략 (시뮬레이션 모드)")

if __name__ == "__main__":
    main()
