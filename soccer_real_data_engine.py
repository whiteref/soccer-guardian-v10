"""
⚽ [V10] Soccer Real Data Engine
- football-data.co.uk에서 실제 5대 리그 경기 데이터 수집
- ELO 레이팅 시스템 (TEAM_TIERS / TRUTH_MAP 대체)
- Walk-Forward Validation 파이프라인
- Brier Score 추적
"""
import os
import json
import logging
import numpy as np
import pandas as pd
import requests
import boto3
from datetime import datetime
from botocore.config import Config

# ==============================================================================
# 1. 실제 경기 데이터 수집 (football-data.co.uk)
# ==============================================================================

# CSV 컬럼 매핑: https://www.football-data.co.uk/notes.txt
# Div, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR (Full Time Result: H/D/A)
# HS, AS (Shots), HST, AST (Shots on Target), B365H, B365D, B365A (Bet365 odds)

LEAGUE_URLS = {
    "EPL": "E0",
    "La_Liga": "SP1",
    "Bundesliga": "D1",
    "Serie_A": "I1",
    "Ligue_1": "F1",
}

# 최근 5시즌 (2020~2025)
SEASONS = ["2021", "2122", "2223", "2324", "2425"]

# 팀명 정규화 맵 (football-data.co.uk → 내부 영문명)
FDATA_TEAM_MAP = {
    "Man City": "Manchester City", "Man United": "Manchester Utd",
    "Newcastle": "Newcastle United", "Nott'm Forest": "Nottingham Forest",
    "Wolves": "Wolverhampton Wanderers", "Spurs": "Tottenham",
    "West Brom": "West Bromwich", "Sheffield United": "Sheffield Utd",
    "Leverkusen": "Bayer Leverkusen", "Bayern Munich": "Bayern Munich",
    "Dortmund": "Borussia Dortmund", "M'gladbach": "Borussia Monchengladbach",
    "Ein Frankfurt": "Eintracht Frankfurt", "FC Koln": "FC Koln",
    "Mainz": "Mainz 05", "Hertha": "Hertha Berlin",
    "Betis": "Real Betis", "Ath Madrid": "Atletico Madrid",
    "Ath Bilbao": "Athletic Club", "Sociedad": "Real Sociedad",
    "Vallecano": "Rayo Vallecano", "Celta": "Celta Vigo",
    "La Coruna": "Deportivo", "Espanol": "Espanyol",
    "Paris SG": "Paris Saint Germain", "St Etienne": "Saint-Etienne",
    "Clermont": "Clermont Foot",
}

def _normalize_fdata_team(name):
    """football-data.co.uk 팀명을 내부 표준 영문명으로 변환"""
    return FDATA_TEAM_MAP.get(name, name)


def fetch_real_match_data(use_cache=True):
    """
    football-data.co.uk에서 실제 5대 리그 × 5시즌 경기 데이터를 수집합니다.
    캐시 파일이 있으면 재사용, 없으면 HTTP 요청으로 수집.
    
    Returns: pd.DataFrame with columns:
        home, away, h_goals, a_goals, result (0=away win, 1=draw, 2=home win),
        h_shots, a_shots, h_sot, a_sot, b365_h, b365_d, b365_a, league, season
    """
    cache_path = "real_match_data_cache.csv"
    
    if use_cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        logging.info(f"📦 캐시에서 {len(df)}경기 로드 완료")
        return df
    
    all_rows = []
    
    for league_name, league_code in LEAGUE_URLS.items():
        for season in SEASONS:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"
            try:
                ts = int(datetime.now().timestamp() * 1000)
                resp = requests.get(f"{url}?t={ts}", timeout=10)
                if resp.status_code != 200:
                    continue
                
                # CSV 파싱 (인코딩 이슈 대응)
                from io import StringIO
                raw_text = resp.content.decode('utf-8', errors='replace')
                df_raw = pd.read_csv(StringIO(raw_text), on_bad_lines='skip')
                
                required_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
                if not all(c in df_raw.columns for c in required_cols):
                    continue
                
                for _, row in df_raw.iterrows():
                    try:
                        home = _normalize_fdata_team(str(row['HomeTeam']).strip())
                        away = _normalize_fdata_team(str(row['AwayTeam']).strip())
                        h_goals = int(row['FTHG'])
                        a_goals = int(row['FTAG'])
                        ftr = str(row['FTR']).strip()
                        
                        if ftr == 'H': result = 2
                        elif ftr == 'D': result = 1
                        else: result = 0
                        
                        # 선택적 컬럼
                        h_shots = float(row.get('HS', 0) or 0)
                        a_shots = float(row.get('AS', 0) or 0)
                        h_sot = float(row.get('HST', 0) or 0)
                        a_sot = float(row.get('AST', 0) or 0)
                        b365_h = float(row.get('B365H', 0) or 0)
                        b365_d = float(row.get('B365D', 0) or 0)
                        b365_a = float(row.get('B365A', 0) or 0)
                        
                        all_rows.append({
                            'home': home, 'away': away,
                            'h_goals': h_goals, 'a_goals': a_goals,
                            'result': result,
                            'h_shots': h_shots, 'a_shots': a_shots,
                            'h_sot': h_sot, 'a_sot': a_sot,
                            'b365_h': b365_h, 'b365_d': b365_d, 'b365_a': b365_a,
                            'league': league_name, 'season': season
                        })
                    except:
                        continue
                        
                logging.info(f"✅ {league_name}/{season}: {len(df_raw)}경기 수집")
            except Exception as e:
                logging.warning(f"⚠️ {league_name}/{season} 수집 실패: {e}")
    
    if not all_rows:
        logging.error("❌ 실제 데이터 수집 실패. 백업 모드 사용.")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_rows)
    df.to_csv(cache_path, index=False)
    logging.info(f"✅ 총 {len(df)}경기 수집 → 캐시 저장 완료")
    return df


# ==============================================================================
# 2. ELO 레이팅 시스템 (TRUTH_MAP / TEAM_TIERS 완전 대체)
# ==============================================================================

class EloRatingSystem:
    """
    경기 결과에 따라 팀 실력을 자동으로 업데이트하는 ELO 레이팅.
    - 초기값 1500, K-factor 32
    - 홈 어드밴티지 보정 +65
    - R2 클라우드 영구 보존
    """
    
    DEFAULT_ELO = 1500
    HOME_ADVANTAGE = 65  # ELO 포인트 (약 55% 홈승 기대)
    
    def __init__(self, k_factor=32):
        self.k = k_factor
        self.ratings = {}
        self._load()
    
    def _get_r2_client(self):
        r2_acc = os.getenv("R2_ACCESS_KEY_ID")
        r2_sec = os.getenv("R2_SECRET_ACCESS_KEY")
        r2_ep = os.getenv("R2_ENDPOINT_URL")
        if r2_acc and r2_sec:
            r2_config = Config(connect_timeout=3, read_timeout=3, retries={'max_attempts': 1})
            return boto3.client('s3', endpoint_url=r2_ep, aws_access_key_id=r2_acc,
                              aws_secret_access_key=r2_sec, region_name='auto', config=r2_config)
        return None
    
    def _load(self):
        """R2 → 로컬 순으로 ELO 데이터 로드"""
        local_path = "elo_ratings.json"
        
        s3 = self._get_r2_client()
        if s3:
            try:
                s3.download_file("soccer-guardian-memory", "elo_ratings.json", local_path)
            except:
                pass
        
        if os.path.exists(local_path):
            try:
                with open(local_path, 'r') as f:
                    self.ratings = json.load(f)
                logging.info(f"📊 ELO 로드: {len(self.ratings)}팀")
            except:
                self.ratings = {}
    
    def save(self):
        """로컬 + R2 동시 저장"""
        local_path = "elo_ratings.json"
        with open(local_path, 'w') as f:
            json.dump(self.ratings, f, indent=2, ensure_ascii=False)
        
        s3 = self._get_r2_client()
        if s3:
            try:
                s3.upload_file(local_path, "soccer-guardian-memory", "elo_ratings.json")
            except:
                pass
    
    def get_elo(self, team):
        return self.ratings.get(team, self.DEFAULT_ELO)
    
    def get_tier_diff(self, home, away):
        """ELO 기반 체급차 계산 (기존 TEAM_TIERS 대체)"""
        h_elo = self.get_elo(home)
        a_elo = self.get_elo(away)
        # ELO 차이를 0~0.4 스케일로 변환 (기존 tier_diff 호환)
        diff = (h_elo - a_elo) / 500.0  # 200 포인트 차이 = 0.4
        return max(-0.4, min(0.4, diff))
    
    def expected_score(self, home, away, include_home_adv=True):
        """ELO 기반 기대 승률 계산 (승/무/패)"""
        h_elo = self.get_elo(home)
        a_elo = self.get_elo(away)
        
        if include_home_adv:
            h_elo += self.HOME_ADVANTAGE
        
        # 기대 승점 (0~1)
        exp_h = 1.0 / (1.0 + 10 ** ((a_elo - h_elo) / 400.0))
        exp_a = 1.0 - exp_h
        
        # 승/무/패 분배 (Dixon-Coles 방식 근사)
        draw_prob = 0.28 * (1.0 - abs(exp_h - 0.5) * 2)  # 박빙일수록 무승부 확률 높음
        h_win = exp_h * (1.0 - draw_prob)
        a_win = exp_a * (1.0 - draw_prob)
        
        total = h_win + draw_prob + a_win + 1e-9
        return h_win/total, draw_prob/total, a_win/total
    
    def update(self, home, away, result):
        """경기 결과에 따라 ELO 업데이트. result: 2=홈승, 1=무, 0=원정승"""
        h_elo = self.get_elo(home)
        a_elo = self.get_elo(away)
        
        # 기대값 (홈 어드밴티지 포함)
        exp_h = 1.0 / (1.0 + 10 ** ((a_elo - (h_elo + self.HOME_ADVANTAGE)) / 400.0))
        
        # 실제 결과
        if result == 2: actual_h = 1.0    # 홈 승
        elif result == 1: actual_h = 0.5  # 무승부
        else: actual_h = 0.0              # 원정 승
        
        # ELO 업데이트
        delta = self.k * (actual_h - exp_h)
        self.ratings[home] = h_elo + delta
        self.ratings[away] = a_elo - delta
    
    def batch_update_from_df(self, df):
        """DataFrame의 모든 경기로 ELO 일괄 업데이트"""
        count = 0
        for _, row in df.iterrows():
            self.update(row['home'], row['away'], row['result'])
            count += 1
        self.save()
        logging.info(f"✅ ELO 일괄 업데이트: {count}경기 처리, {len(self.ratings)}팀")
        return count


# ==============================================================================
# 3. 피처 엔지니어링 (실제 데이터 → ML 입력)
# ==============================================================================

def build_features_from_real_data(df, elo_system):
    """
    실제 경기 DataFrame에서 머신러닝 피처를 추출합니다.
    각 경기에 대해 해당 경기 이전 직전 5경기의 평균 통계를 사용.
    
    Features (16개, V9.5 호환):
        0: home_avg_goals (≈xG 대체)
        1: home_avg_conceded (≈xGA 대체)
        2: home_shots_ratio (≈PPDA 대체)
        3: away_avg_goals
        4: away_avg_conceded
        5: away_shots_ratio
        6: home_advantage (1/0)
        7: odds_implied_diff (배당 내재 차이)
        8: elo_strength_ratio (ELO 비율)
        9: home_form (최근 5경기 승률)
        10: away_form
        11: home_scoring_consistency (득점 표준편차 역수)
        12: elo_diff_normalized
        13: goal_diff_trend
        14: draw_tendency (두 팀 무승부 빈도)
        15: upset_potential (ELO 약체가 이길 확률)
    """
    X, y = [], []
    teams_history = {}  # {team: deque of recent results}
    
    for idx, row in df.iterrows():
        home, away = row['home'], row['away']
        
        # 각 팀의 최근 5경기 히스토리 수집
        h_hist = teams_history.get(home, [])
        a_hist = teams_history.get(away, [])
        
        if len(h_hist) >= 3 and len(a_hist) >= 3:
            # 홈팀 최근 통계 (최대 5경기)
            h_recent = h_hist[-5:]
            a_recent = a_hist[-5:]
            
            h_avg_goals = np.mean([g['goals_for'] for g in h_recent])
            h_avg_conceded = np.mean([g['goals_against'] for g in h_recent])
            h_shots_ratio = np.mean([g['shots_ratio'] for g in h_recent])
            h_form = np.mean([g['points'] for g in h_recent]) / 3.0
            h_consistency = 1.0 / (np.std([g['goals_for'] for g in h_recent]) + 0.5)
            h_gd_trend = np.mean([g['goals_for'] - g['goals_against'] for g in h_recent[-3:]])
            
            a_avg_goals = np.mean([g['goals_for'] for g in a_recent])
            a_avg_conceded = np.mean([g['goals_against'] for g in a_recent])
            a_shots_ratio = np.mean([g['shots_ratio'] for g in a_recent])
            a_form = np.mean([g['points'] for g in a_recent]) / 3.0
            
            # ELO 기반 피처
            h_elo = elo_system.get_elo(home)
            a_elo = elo_system.get_elo(away)
            elo_ratio = h_elo / max(a_elo, 1000)
            elo_diff_norm = (h_elo - a_elo) / 400.0
            
            # 배당 기반 피처
            b365_h = max(row.get('b365_h', 0), 1.01)
            b365_a = max(row.get('b365_a', 0), 1.01)
            odds_diff = (1/b365_a) - (1/b365_h)  # 양수면 홈 유리
            
            # 무승부 경향
            h_draws = sum(1 for g in h_recent if g['points'] == 1) / len(h_recent)
            a_draws = sum(1 for g in a_recent if g['points'] == 1) / len(a_recent)
            draw_tendency = (h_draws + a_draws) / 2.0
            
            # 이변 가능성 (약팀이 강팀을 이길 확률)
            upset_pot = max(0, (a_elo - h_elo) / 400.0) if h_elo > a_elo else max(0, (h_elo - a_elo) / 400.0)
            
            features = [
                h_avg_goals, h_avg_conceded, h_shots_ratio,
                a_avg_goals, a_avg_conceded, a_shots_ratio,
                1.0,  # home_advantage (항상 홈 기준)
                odds_diff, elo_ratio, h_form, a_form,
                h_consistency, elo_diff_norm, h_gd_trend,
                draw_tendency, upset_pot
            ]
            
            X.append(features)
            y.append(row['result'])
        
        # 히스토리 업데이트
        h_goals, a_goals = row['h_goals'], row['a_goals']
        h_shots = max(row.get('h_shots', 1), 1)
        a_shots = max(row.get('a_shots', 1), 1)
        
        if row['result'] == 2: h_pts, a_pts = 3, 0
        elif row['result'] == 1: h_pts, a_pts = 1, 1
        else: h_pts, a_pts = 0, 3
        
        if home not in teams_history:
            teams_history[home] = []
        teams_history[home].append({
            'goals_for': h_goals, 'goals_against': a_goals,
            'shots_ratio': h_shots / (h_shots + a_shots),
            'points': h_pts
        })
        
        if away not in teams_history:
            teams_history[away] = []
        teams_history[away].append({
            'goals_for': a_goals, 'goals_against': h_goals,
            'shots_ratio': a_shots / (h_shots + a_shots),
            'points': a_pts
        })
        
        # ELO 업데이트 (시간순)
        elo_system.update(home, away, row['result'])
    
    return np.array(X), np.array(y)


# ==============================================================================
# 4. Brier Score 추적 시스템
# ==============================================================================

class BrierScoreTracker:
    """
    예측 확률과 실제 결과를 비교하여 Brier Score를 계산·저장합니다.
    0 = 완벽한 예측, 0.667 = 동전 던지기 수준 (3-way)
    """
    
    def __init__(self):
        self.predictions = []
        self._load()
    
    def _load(self):
        local_path = "brier_score_history.json"
        if os.path.exists(local_path):
            try:
                with open(local_path, 'r') as f:
                    self.predictions = json.load(f)
            except:
                self.predictions = []
    
    def save(self):
        with open("brier_score_history.json", 'w') as f:
            json.dump(self.predictions, f, indent=2, ensure_ascii=False)
        
        # R2 동기화
        try:
            r2_acc = os.getenv("R2_ACCESS_KEY_ID")
            r2_sec = os.getenv("R2_SECRET_ACCESS_KEY")
            r2_ep = os.getenv("R2_ENDPOINT_URL")
            if r2_acc and r2_sec:
                s3 = boto3.client('s3', endpoint_url=r2_ep,
                    aws_access_key_id=r2_acc, aws_secret_access_key=r2_sec, region_name='auto')
                s3.upload_file("brier_score_history.json", "soccer-guardian-memory", "brier_score_history.json")
        except:
            pass
    
    def add_prediction(self, match_id, home, away, h_prob, d_prob, a_prob, prediction):
        """예측 결과를 기록 (경기 전)"""
        self.predictions.append({
            'match_id': match_id,
            'home': home, 'away': away,
            'h_prob': round(h_prob/100, 4),
            'd_prob': round(d_prob/100, 4),
            'a_prob': round(a_prob/100, 4),
            'prediction': prediction,
            'actual_result': None,
            'brier_score': None,
            'date': datetime.now().isoformat()
        })
        self.save()
    
    def record_result(self, match_id, actual_result):
        """실제 결과 기록 + Brier Score 계산"""
        for pred in self.predictions:
            if pred['match_id'] == match_id and pred['actual_result'] is None:
                pred['actual_result'] = actual_result
                
                # Brier Score 계산 (3-way)
                actual_vec = [0, 0, 0]
                actual_vec[actual_result] = 1  # 0=away, 1=draw, 2=home
                
                pred_vec = [pred['a_prob'], pred['d_prob'], pred['h_prob']]
                brier = sum((p - a) ** 2 for p, a in zip(pred_vec, actual_vec)) / 3.0
                pred['brier_score'] = round(brier, 4)
                break
        self.save()
    
    def get_average_brier(self, last_n=None):
        """최근 N경기의 평균 Brier Score"""
        scored = [p for p in self.predictions if p['brier_score'] is not None]
        if not scored:
            return None
        if last_n:
            scored = scored[-last_n:]
        return round(np.mean([p['brier_score'] for p in scored]), 4)
    
    def get_accuracy(self, last_n=None):
        """최근 N경기의 정답률 (argmax 기준)"""
        completed = [p for p in self.predictions if p['actual_result'] is not None]
        if not completed:
            return None
        if last_n:
            completed = completed[-last_n:]
        
        correct = 0
        for p in completed:
            probs = [p['a_prob'], p['d_prob'], p['h_prob']]
            predicted = np.argmax(probs)
            if predicted == p['actual_result']:
                correct += 1
        return round(correct / len(completed), 4)
    
    def get_pending_matches(self):
        """아직 결과가 입력되지 않은 예측 목록"""
        return [p for p in self.predictions if p['actual_result'] is None]


# ==============================================================================
# 5. V10 통합 팩토리 함수
# ==============================================================================

def initialize_v10_engine():
    """
    V10 엔진 초기화: 실제 데이터 수집 → ELO 구축 → XGBoost 학습
    Returns: (X_train, y_train, elo_system, brier_tracker)
    """
    logging.info("🚀 [V10] 실제 데이터 기반 학습 엔진 초기화 중...")
    
    # 1. 실제 데이터 수집
    df = fetch_real_match_data()
    
    if df.empty:
        logging.error("❌ 데이터 수집 실패")
        return None, None, EloRatingSystem(), BrierScoreTracker()
    
    # 2. ELO 시스템 초기화 + 경기 데이터로 ELO 구축
    elo = EloRatingSystem()
    
    if not elo.ratings:
        logging.info("📊 ELO 초기 구축 중 (과거 데이터 기반)...")
    
    # 3. 피처 엔지니어링 (ELO도 시간순으로 업데이트됨)
    X, y = build_features_from_real_data(df, elo)
    elo.save()
    
    logging.info(f"✅ [V10] 학습 데이터: {len(X)}경기, ELO: {len(elo.ratings)}팀")
    
    # 4. Brier Score 트래커
    brier = BrierScoreTracker()
    
    return X, y, elo, brier
