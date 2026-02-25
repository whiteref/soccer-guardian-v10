import os
import json

KALMAN_STORE = "v13_kalman_states.json"

class KalmanGuardianEngine:
    """[V13 Kalman Guardian] Streamlit Cloud 호환 버전"""
    def __init__(self, q=0.02, r=0.15):
        self.q = q
        self.r = r
        self.states = {}
        # 로컬 파일 로드 시도
        try:
            if os.path.exists(KALMAN_STORE):
                with open(KALMAN_STORE, 'r') as f:
                    self.states = json.load(f)
        except:
            self.states = {}

    def _save_states(self):
        """저장 시도 — 실패해도 무시 (Streamlit Cloud 읽기전용)"""
        try:
            with open(KALMAN_STORE, 'w') as f:
                json.dump(self.states, f, indent=4)
        except:
            pass

    def get_stabilized_xg(self, team_name, raw_xg):
        """Kalman Filter로 xG 안정화"""
        if team_name not in self.states:
            self.states[team_name] = [raw_xg, 1.0]
            self._save_states()
            return raw_xg

        prev_estimate, prev_p = self.states[team_name]
        p_prior = prev_p + self.q
        k = p_prior / (p_prior + self.r)
        new_estimate = prev_estimate + k * (raw_xg - prev_estimate)
        new_p = (1 - k) * p_prior
        self.states[team_name] = [new_estimate, new_p]
        self._save_states()
        return round(new_estimate, 3)

    def get_all_estimates(self):
        return {k: v[0] for k, v in self.states.items()}
