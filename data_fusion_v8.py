import logging
import random
import numpy as np

# [V8.5 Hyper-Fusion] 데이터 통합 엔진
# 이 모듈은 여러 웹사이트의 파싱 결과를 V8 엔진이 이해할 수 있는 피처(Feature)로 변환합니다.

def get_squad_value_data():
    """Transfermarkt 기반 스쿼드 가치 (백만 유로)"""
    return {
        "Manchester City": 1260, "Arsenal": 1170, "Liverpool": 920, "Tottenham": 770,
        "Chelsea": 950, "Aston Villa": 630, "Manchester Utd": 850, "Newcastle United": 650,
        "Brighton": 520, "West Ham": 480, "Bournemouth": 350, "Fulham": 340,
        "Wolverhampton Wanderers": 360, "Brentford": 410, "Everton": 350, "Nottingham Forest": 370,
        "Leicester": 300, "Leeds": 250, "Crystal Palace": 420, "Sunderland": 80,
        "Juventus": 590, "Inter": 670, "Napoli": 550, "AC Milan": 600, "Roma": 450,
        "Atalanta": 440, "Lazio": 320, "Fiorentina": 280, "Torino": 180, "Genoa": 150,
        "Parma": 130, "Como": 140, "Cagliari": 100, "Cremonese": 60
    }

def get_injury_impact_data():
    """핵심 선수 부상 타격도 (0~1.0)"""
    return {
        "Liverpool": 0.45,  # 살라 부상 타격 큼
        "Arsenal": 0.20,
        "Manchester City": 0.15,
        "Tottenham": 0.35,
        "Juventus": 0.10,
        "Real Madrid": 0.50, # 주전 대거 부상 시나리오
    }

def get_odds_flow_data():
    """Betman/OddsPortal 실시간 배당 흐름 (Drop Ratio %)"""
    # 실제 수집 데이터 연동 전까지 시뮬레이션
    return {
        "Manchester City": -3.5, 
        "Tottenham": -7.2, 
        "Liverpool": 2.1, 
        "Arsenal": -1.5
    }

def get_luck_factor_data():
    """성적 vs xG 괴리율 (운 지수) - Flashscore 대조"""
    # +값이면 실력보다 운이 좋아 승점을 많이 딴 상태 (거품 가능성)
    return {
        "Liverpool": 0.15,
        "Bayer Leverkusen": 0.25,
        "Manchester City": -0.05,
        "Chelsea": -0.20 # 실력만큼 승점이 안 나오는 상태 (반등 가능성)
    }

def calculate_fractal_indicators(team_name):
    """
    [V8.7 Fractal Engine]
    팀의 과거 xG 히스토리를 분석하여 허스트 지수와 효율성을 계산합니다.
    (실제 시계열 데이터가 없는 경우를 위해 시드 기반 시뮬레이션 활용)
    """
    # 팀명 기반 시드 고정 (일관성 있는 지표 생성)
    seed = sum(ord(c) for c in team_name)
    random.seed(seed)
    np.random.seed(seed)
    
    # 최근 10경기 xG 흐름 시뮬레이션
    history = np.random.normal(1.5, 0.5, 10)
    
    # 1. 허스트 지수 (Hurst Exponent) 근사치
    # 0.5: Random Walk, >0.5: Persistence(상승세 유지), <0.5: Mean Reversion(조정/반등 임박)
    if team_name in ["Manchester City", "Arsenal", "Liverpool"]:
        hurst = 0.65 + random.uniform(-0.05, 0.1) # 강팀은 추세 유지 성향
    elif team_name in ["Chelsea", "Manchester Utd"]:
        hurst = 0.35 + random.uniform(-0.1, 0.05) # 기복이 큰 팀은 평균 회귀 성향
    else:
        hurst = 0.50 + random.uniform(-0.1, 0.1)
        
    # 2. 효율성 (Efficiency Index)
    # 실효 변동성 대비 추세의 강도
    efficiency = abs(np.diff(history).mean()) / (np.std(history) + 1e-6)
    
    # 3. 스큐 (Skewness)
    # 하방 리스크 (이변 가능성) - Skew가 높을수록 '터질' 확률이 높음
    skew = np.mean(((history - np.mean(history)) / np.std(history))**3)
    
    return round(hurst, 3), round(efficiency, 3), round(skew, 3)

def fetch_all_fusion_features(home_eng, away_eng):
    """모든 외부 소스를 퓨전하여 단일 딕셔너리로 반환"""
    sq_values = get_squad_value_data()
    injuries = get_injury_impact_data()
    odds = get_odds_flow_data()
    luck = get_luck_factor_data()
    
    h_val = sq_values.get(home_eng, 200)
    a_val = sq_values.get(away_eng, 200)
    sq_ratio = h_val / a_val
    
    h_inj = injuries.get(home_eng, 0.0)
    a_inj = injuries.get(away_eng, 0.0)
    inj_diff = h_inj - a_inj
    
    # 배당 흐름 (홈팀 기준 점수화)
    # 홈 배당이 떨어지면(-), 원정이 올라가면(+) -> 홈에 유리한 흐름
    h_odd = odds.get(home_eng, 0.0)
    a_odd = odds.get(away_eng, 0.0)
    odd_flow = a_odd - h_odd # 양수일수록 홈팀에 돈이 쏠림
    
    # 운 지수
    h_luck = luck.get(home_eng, 0.0)
    a_luck = luck.get(away_eng, 0.0)
    luck_fact = h_luck - a_luck
    
    # [V8.7 Fractal Indicators & V8.8 Extreme TTTr]
    h_hurst, h_eff, h_skew = calculate_fractal_indicators(home_eng)
    a_hurst, a_eff, a_skew = calculate_fractal_indicators(away_eng)
    
    # 🛡️ 신의 방패 (Shield) 트리거: 정배당 강팀의 엔트로피 붕괴 상태 (Extreme Negative Skew + Low Hurst)
    h_shield_trigger = True if h_hurst < 0.40 and h_skew < -0.8 else False
    
    # 🔱 신의 창 (Spear) 트리거: 역배당 언더독의 역습 효율성 폭발 상태 (High Efficiency + Positive Skew)
    a_spear_trigger = True if a_eff > 0.65 and a_skew > 0.5 else False
    
    return {
        "sq_ratio": round(sq_ratio, 3),
        "inj_diff": round(inj_diff, 3),
        "odd_flow": round(odd_flow, 3),
        "luck_factor": round(luck_fact, 3),
        "hurst_diff": round(h_hurst - a_hurst, 3),
        "eff_diff": round(h_eff - a_eff, 3),
        "skew_total": round(h_skew + a_skew, 3),
        "h_hurst": h_hurst,
        "h_eff": h_eff,
        "h_skew": h_skew,
        "a_hurst": a_hurst,
        "a_eff": a_eff,
        "a_skew": a_skew,
        "h_shield_trigger": h_shield_trigger,
        "a_spear_trigger": a_spear_trigger
    }
