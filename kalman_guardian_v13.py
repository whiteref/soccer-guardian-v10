import os
import json
import boto3
from datetime import datetime

KALMAN_STORE = "/home/white/.gemini/antigravity/brain/v13_kalman_states.json"

class KalmanGuardianEngine:
    """
    [V13 Kalman Guardian: State Estimation & Noise Reduction]
    Separates 'True Team Strength' from 'Statistical Noise (xG)' using Kalman Filtering.
    Ensures that a single lucky or unlucky match doesn't over-influence the model.
    """
    def __init__(self, q=0.02, r=0.15):
        # q: Process noise (How much true strength fluctuates over time)
        # r: Measurement noise (How much noise is in the raw xG stat)
        self.q = q
        self.r = r
        self.states = self._load_states()
        self._sync_from_r2()

    def _load_states(self):
        if os.path.exists(KALMAN_STORE):
            try:
                with open(KALMAN_STORE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {} # {team_name: [estimate, error_covariance]}

    def _save_states(self):
        with open(KALMAN_STORE, 'w') as f:
            json.dump(self.states, f, indent=4)
        self._sync_to_r2()

    def _get_r2_client(self):
        r2_acc = os.getenv("R2_ACCESS_KEY_ID")
        r2_sec = os.getenv("R2_SECRET_ACCESS_KEY")
        r2_ep = os.getenv("R2_ENDPOINT_URL")
        if r2_acc and r2_sec:
            return boto3.client(
                's3', endpoint_url=r2_ep, 
                aws_access_key_id=r2_acc, aws_secret_access_key=r2_sec, region_name='auto'
            )
        return None

    def _sync_to_r2(self):
        s3 = self._get_r2_client()
        if s3:
            try:
                s3.upload_file(KALMAN_STORE, "guardian-memory", "v13_kalman_states.json")
            except:
                pass

    def _sync_from_r2(self):
        s3 = self._get_r2_client()
        if s3:
            try:
                s3.download_file("guardian-memory", "v13_kalman_states.json", KALMAN_STORE)
                self.states = self._load_states()
            except:
                pass

    def get_stabilized_xg(self, team_name, raw_xg):
        """
        [Kalman Filter Implementation]
        Calculates the stabilized xG by filtering the raw input through the state model.
        """
        if team_name not in self.states:
            # First time seeing this team: High uncertainty
            self.states[team_name] = [raw_xg, 1.0]
            self._save_states()
            return raw_xg

        prev_estimate, prev_p = self.states[team_name]

        # 1. Prediction Step (Time Update)
        # We assume the team's strength is roughly constant plus some noise (q)
        p_prior = prev_p + self.q

        # 2. Measurement Update Step (Correction)
        # k: Kalman Gain (Weight assigned to the new measurement)
        k = p_prior / (p_prior + self.r)
        
        # New estimate is a weighted average of prediction and new data
        new_estimate = prev_estimate + k * (raw_xg - prev_estimate)
        
        # Update error covariance
        new_p = (1 - k) * p_prior

        self.states[team_name] = [new_estimate, new_p]
        self._save_states()
        
        return round(new_estimate, 3)

    def get_all_estimates(self):
        return {k: v[0] for k, v in self.states.items()}
