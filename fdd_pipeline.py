"""
fdd_pipeline.py
===============
Loads the trained model bank (7 faults x 2 datasets) and runs the
three-layer hybrid diagnosis described in thesis Chapter 3.9:

  Layer 1: Gordon-Ng physics residual
  Layer 2: 12-feature diagnostic vector (feature_engineering.py)
  Layer 3: RBF-SVM bank with Platt-calibrated probabilities

One-versus-normal bank: each fault gets its own binary classifier, and the
decision threshold is a fixed alpha = 0.50 per the thesis's reported
evaluation methodology.
"""

import os
import joblib
import numpy as np

from feature_engineering import RawReading, feature_vector

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

FAULTS = ["nc", "oc", "rl", "cf", "eo", "fwe", "fwc"]
FAULT_LABELS = {
    "nc": "Non-Condensable Gas (NC)",
    "oc": "Refrigerant Overcharge (OC)",
    "rl": "Refrigerant Leakage (RL)",
    "cf": "Condenser Fouling (CF)",
    "eo": "Excess Oil (EO)",
    "fwe": "Reduced Evaporator Water Flow (FWE)",
    "fwc": "Reduced Condenser Water Flow (FWC)",
}

DATASETS = ["sql", "rp1043"]
DATASET_LABELS = {
    "sql": "EnergyPlus / OpenStudio (338-Ton commercial simulation)",
    "rp1043": "ASHRAE RP-1043 (90-Ton laboratory chiller)",
}

# Healthy-baseline constants quoted in thesis Chapter 4 (Table 4.1) - used
# only by the GUI's simulated-stream generator, not by feature_engineering
# itself (which needs no baseline constants - see the real source script).
BASELINE = {
    "sql": {"cop_nominal": 4.042, "power_nominal_kw": 278.8},
    "rp1043": {"cop_nominal": 3.643, "power_nominal_kw": 53.7},
}

ALARM_THRESHOLD = 0.50


class FDDModelBank:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.physics = {}
        self.scalers = {}
        self.svms = {}
        self._load(models_dir)

    def _load(self, models_dir: str):
        missing = []
        for fault in FAULTS:
            phys_path = os.path.join(models_dir, f"physics_{fault}_sql.pkl")
            if os.path.exists(phys_path):
                self.physics[fault] = joblib.load(phys_path)
            else:
                missing.append(phys_path)

            for ds in DATASETS:
                scaler_path = os.path.join(models_dir, f"scaler_{fault}_{ds}.pkl")
                svm_path = os.path.join(models_dir, f"svm_{fault}_{ds}.pkl")
                if os.path.exists(scaler_path):
                    self.scalers[(fault, ds)] = joblib.load(scaler_path)
                else:
                    missing.append(scaler_path)
                if os.path.exists(svm_path):
                    self.svms[(fault, ds)] = joblib.load(svm_path)
                else:
                    missing.append(svm_path)

        self.missing = missing
        self.available_faults = [f for f in FAULTS if f in self.physics]

    def diagnose(self, dataset: str, reading: RawReading) -> dict:
        """Runs every available fault classifier for the given dataset
        against one raw sensor reading. Returns a dict keyed by fault code:
            {fault: {"probability": float, "alarm": bool, "residual": float}}
        """
        results = {}
        for fault in self.available_faults:
            key = (fault, dataset)
            if key not in self.scalers or key not in self.svms:
                continue
            physics_model = self.physics[fault]
            feats = feature_vector(reading, physics_model)
            scaler = self.scalers[key]
            svm = self.svms[key]

            x = np.array(feats).reshape(1, -1)
            x_scaled = scaler.transform(x)
            proba = svm.predict_proba(x_scaled)[0]
            # classes_ is [0, 1] -> probability of class 1 = fault present
            fault_idx = list(svm.classes_).index(1) if 1 in svm.classes_ else 1
            p_fault = float(proba[fault_idx])

            results[fault] = {
                "probability": p_fault,
                "alarm": p_fault >= ALARM_THRESHOLD,
                "power_residual": feats[0],  # Power_Residual is index 0 in FEATURE_ORDER
            }
        return results

    def primary_diagnosis(self, results: dict):
        """Largest eligible probability above threshold, per thesis
        Chapter 3.9 ('prioritises the largest eligible probability for the
        primary diagnosis'); otherwise reports Normal."""
        alarmed = {f: r for f, r in results.items() if r["alarm"]}
        if not alarmed:
            return None, 0.0
        best = max(alarmed, key=lambda f: alarmed[f]["probability"])
        return best, alarmed[best]["probability"]
