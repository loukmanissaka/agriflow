"""
AgriFlow — Simulation Monte Carlo (Section 6)
=============================================
Génère les métriques de la Section 6 par simulation paramétrique.
Les paramètres sont issus de la littérature (Table 1 de l'article).

Usage:
    python simulation_monte_carlo.py

Sorties:
    - results/metrics_par_dag.csv         → Table 2 (fiabilité)
    - results/ablation_study.csv          → Table 4 (ablation)
    - results/quality_scores.csv          → Figure 2 (qualité)
    - results/sensitivity_validation.csv  → Section 6.5
    - results/scalability.csv             → Table 5
    - results/figures/                    → Figures LaTeX-ready
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import os
import json

# ── Reproductibilité ─────────────────────────────────────────
RNG = np.random.default_rng(seed=42)
N_SIM = 1000      # réplications Monte Carlo
N_DAYS = 365      # jours simulés par réplication

os.makedirs("results/figures", exist_ok=True)

# ============================================================
# PARAMÈTRES DE SIMULATION (Table 1 de l'article)
# Sources bibliographiques intégrées en commentaires
# ============================================================
PARAMS = {
    # Disponibilités API — sources : CDSE SLA 2024, littérature IoT
    "p_iot":          {"dist": "bernoulli", "p": 0.972,   "ref": "iot_review_2025"},
    "p_sentinel2":    {"dist": "bernoulli", "p": 0.918,   "ref": "cdse_sla_2024"},
    "p_weather":      {"dist": "bernoulli", "p": 0.991,   "ref": "openmeteo_uptime"},
    "p_fmis":         {"dist": "bernoulli", "p": 0.987,   "ref": "pipeline_survey_2024"},
    "p_align":        {"dist": "bernoulli", "p": 0.954,   "ref": "pipeline_survey_2024"},
    "p_quality_gate": {"dist": "bernoulli", "p": 0.986,   "ref": "pipeline_survey_2024"},

    # Latences (minutes) — source : pipeline_survey_2024, iot_agronomy_2023
    "lat_iot":        {"dist": "gamma",     "k": 2,   "theta": 0.7,  "ref": "iot_agronomy_2023"},
    "lat_sentinel2":  {"dist": "gamma",     "k": 3,   "theta": 6.1,  "ref": "sentinel2_ard_2022"},
    "lat_weather":    {"dist": "gamma",     "k": 2,   "theta": 0.3,  "ref": "openmeteo_uptime"},
    "lat_align":      {"dist": "gamma",     "k": 4,   "theta": 6.2,  "ref": "pipeline_survey_2024"},

    # Recouvrement spatial Sentinel-2 — source : sentinel2_ard_2022
    "rho_S_summer":   {"dist": "uniform",   "low": 0.78, "high": 0.95, "ref": "sentinel2_ard_2022"},
    "rho_S_autumn":   {"dist": "uniform",   "low": 0.40, "high": 0.70, "ref": "sentinel2_ard_2022"},

    # Qualité initiale des données — source : iot_review_2025, iot_frontiers_2025
    "C_iot":          {"dist": "beta",      "a": 48,  "b": 2,    "ref": "iot_review_2025"},
    "R_iot":          {"dist": "beta",      "a": 60,  "b": 1.5,  "ref": "iot_frontiers_2025"},
    "C_sat":          {"dist": "beta",      "a": 28,  "b": 4,    "ref": "ndvi_sentinel2_2023"},
    "R_sat":          {"dist": "beta",      "a": 50,  "b": 2.5,  "ref": "sentinel2_ard_2022"},

    # Fréquence changements de schéma FMIS — source : pipeline_survey_2024
    "fmis_schema_changes_per_year": {"dist": "poisson", "lam": 3.0, "ref": "pipeline_survey_2024"},
}

# Poids modèle qualité (Équation 1 article)
ALPHA, BETA, GAMMA = 0.40, 0.40, 0.20
LAMBDA_DECAY = 0.10   # h⁻¹
Q_MIN = 0.85          # seuil porte qualité

# ============================================================
# FONCTIONS DU MODÈLE FORMEL (Section 3 de l'article)
# ============================================================

def score_qualite(C: float, R: float, F: float) -> float:
    """Équation (1) : q(D) = αC + βR + γF"""
    return ALPHA * C + BETA * R + GAMMA * F

def fraicheur(lag_heures: float) -> float:
    """Équation (4) : F = exp(-λ * lag_max)"""
    lag_heures = max(lag_heures, 0)
    return np.exp(-LAMBDA_DECAY * lag_heures)