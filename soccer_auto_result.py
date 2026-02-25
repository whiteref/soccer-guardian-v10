"""
⚽ [V10.2] Auto Result Fetcher
- 경기 결과를 자동으로 수집하여 ELO + Brier Score 자동 업데이트
- API-Football (무료 Tier: api-sports.io) 또는 football-data.co.uk 최근 결과 사용
- 수동 입력 없이 재실행 시 자동 반영
"""
import os, json, logging, requests
from datetime import datetime, timedelta

def fetch_recent_results_fdata():
    """
    football-data.co.uk 최신 시즌 CSV에서 최근 결과를 가져옵니다.
    이미 캐시된 데이터와 비교하여 새로운 경기만 추출.
    """
    import pandas as pd
    from io import StringIO
    
    LEAGUES = {"E0": "EPL", "SP1": "La_Liga", "D1": "Bundesliga", "I1": "Serie_A", "F1": "Ligue_1"}
    new_results = []
    
    for code, name in LEAGUES.items():
        try:
            ts = int(datetime.now().timestamp() * 1000)
            url = f"https://www.football-data.co.uk/mmz4281/2425/{code}.csv?t={ts}"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            
            raw = resp.content.decode('utf-8', errors='replace')
            df = pd.read_csv(StringIO(raw), on_bad_lines='skip')
            
            required = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'Date']
            if not all(c in df.columns for c in required):
                continue
            
            for _, row in df.iterrows():
                try:
                    home = str(row['HomeTeam']).strip()
                    away = str(row['AwayTeam']).strip()
                    h_goals = int(row['FTHG'])
                    a_goals = int(row['FTAG'])
                    ftr = str(row['FTR']).strip()
                    date_str = str(row['Date']).strip()
                    
                    if ftr == 'H': result = 2
                    elif ftr == 'D': result = 1
                    else: result = 0
                    
                    new_results.append({
                        'home': home, 'away': away,
                        'h_goals': h_goals, 'a_goals': a_goals,
                        'result': result, 'date': date_str,
                        'league': name
                    })
                except:
                    continue
        except Exception as e:
            logging.warning(f"⚠️ {name} 결과 수집 실패: {e}")
    
    return new_results


def fetch_recent_results_api_football(days=7):
    """
    API-Football (api-sports.io) 무료 Tier로 최근 N일 경기 결과 수집.
    환경변수: API_FOOTBALL_KEY
    """
    api_key = os.getenv("API_FOOTBALL_KEY", "")
    if not api_key:
        return []
    
    results = []
    today = datetime.now()
    
    for delta in range(days):
        date = (today - timedelta(days=delta)).strftime("%Y-%m-%d")
        try:
            ts = int(datetime.now().timestamp() * 1000)
            resp = requests.get(
                f"https://v3.football.api-sports.io/fixtures?date={date}&t={ts}",
                headers={"x-apisports-key": api_key},
                timeout=10
            )
            data = resp.json()
            
            for fixture in data.get('response', []):
                status = fixture.get('fixture', {}).get('status', {}).get('short', '')
                if status != 'FT':  # Full Time만
                    continue
                
                home = fixture['teams']['home']['name']
                away = fixture['teams']['away']['name']
                h_goals = fixture['goals']['home']
                a_goals = fixture['goals']['away']
                
                if h_goals > a_goals: result = 2
                elif h_goals == a_goals: result = 1
                else: result = 0
                
                results.append({
                    'home': home, 'away': away,
                    'h_goals': h_goals, 'a_goals': a_goals,
                    'result': result, 'date': date,
                    'league': fixture['league']['name']
                })
        except Exception as e:
            logging.warning(f"⚠️ API-Football {date} 실패: {e}")
    
    return results


def auto_update_elo_and_brier(elo_system, brier_tracker):
    """
    자동 결과 수집 → ELO + Brier Score 업데이트.
    이미 처리된 경기는 건너뜀 (중복 방지).
    
    반환: 새로 처리된 경기 수
    """
    # 이미 처리된 경기 ID 로드
    processed_path = "auto_processed_matches.json"
    if os.path.exists(processed_path):
        with open(processed_path, 'r') as f:
            processed = set(json.load(f))
    else:
        processed = set()
    
    # 결과 수집 (두 소스 병합)
    results = fetch_recent_results_fdata()
    api_results = fetch_recent_results_api_football()
    results.extend(api_results)
    
    new_count = 0
    for r in results:
        match_id = f"{r['home']}_vs_{r['away']}_{r['date']}"
        
        if match_id in processed:
            continue
        
        # ELO 업데이트
        elo_system.update(r['home'], r['away'], r['result'])
        
        # Brier Score에 기존 예측이 있으면 기록
        for pred in brier_tracker.predictions:
            if pred['actual_result'] is None:
                # 팀명 매칭 (부분 일치)
                if (r['home'].lower() in pred['home'].lower() or 
                    pred['home'].lower() in r['home'].lower()):
                    if (r['away'].lower() in pred['away'].lower() or 
                        pred['away'].lower() in r['away'].lower()):
                        brier_tracker.record_result(pred['match_id'], r['result'])
        
        processed.add(match_id)
        new_count += 1
    
    # 처리 완료 기록 저장
    with open(processed_path, 'w') as f:
        json.dump(list(processed), f)
    
    if new_count > 0:
        elo_system.save()
        brier_tracker.save()
        logging.info(f"✅ [Auto Update] {new_count}경기 자동 반영 완료! ELO + Brier 업데이트됨")
    
    return new_count
