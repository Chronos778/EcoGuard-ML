from __future__ import annotations
import pandas as pd
from typing import Dict, Any


CONTROLLABLE = [
    'human_activity',
    'hunting_pressure',
    'habitat_quality',
]


def run_scenario(predictor, baseline_row: pd.Series, adjustments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a trained predictor and a baseline row (population dataset schema),
    apply adjustments to controllable variables and compute predicted impacts.
    Returns deltas for population and risk probability breakdown.
    """
    df_base = pd.DataFrame([baseline_row.to_dict()])
    pop_base = predictor.predict_population(df_base)[0]
    risk_pred_base, risk_probs_base = predictor.predict_risk(df_base)

    df_adj = df_base.copy()
    for k, v in adjustments.items():
        if k in df_adj.columns:
            df_adj[k] = v

    pop_adj = predictor.predict_population(df_adj)[0]
    risk_pred_adj, risk_probs_adj = predictor.predict_risk(df_adj)

    return {
        'population_baseline': float(pop_base),
        'population_adjusted': float(pop_adj),
        'population_delta': float(pop_adj - pop_base),
        'risk_baseline': str(risk_pred_base[0]),
        'risk_adjusted': str(risk_pred_adj[0]),
        'risk_probs_baseline': risk_probs_base[0].tolist(),
        'risk_probs_adjusted': risk_probs_adj[0].tolist(),
        'adjustments': adjustments,
    }
