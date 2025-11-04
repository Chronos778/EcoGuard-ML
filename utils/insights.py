import numpy as np
import pandas as pd
from typing import List, Dict


RISK_SCORE = {"Low": 0, "Medium": 1, "High": 2}


def _latest_by_group(df: pd.DataFrame, group_cols: List[str], date_col: str) -> pd.DataFrame:
    """Return latest rows per group by date_col."""
    idx = df.groupby(group_cols)[date_col].transform(max) == df[date_col]
    return df[idx].copy()


def compute_risk_hotspots(pop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify hotspots by combining risk level and recent trend.
    Expects columns: ['date','location','species_name','population_count','risk_level','human_activity','habitat_quality','hunting_pressure']
    """
    df = pop_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(['species_name', 'location', 'date'], inplace=True)

    # Recent trend (last 6 records per group) using linear fit on index
    def _trend(group: pd.DataFrame) -> float:
        tail = group.tail(6)
        if len(tail) < 3:
            return 0.0
        # Ensure proper numeric arrays for polyfit
        x = np.arange(len(tail), dtype=float)
        y = pd.to_numeric(tail['population_count'], errors='coerce').to_numpy(dtype=float)
        # Handle potential NaNs after coercion
        mask = ~np.isnan(y)
        if mask.sum() < 3:
            return 0.0
        coef = np.polyfit(x[mask], y[mask], 1)[0]
        return float(coef)

    trend = (
        df.groupby(['species_name', 'location'])
          .apply(_trend)
          .reset_index(name='trend_slope')
    )

    latest = _latest_by_group(df, ['species_name', 'location'], 'date')
    latest = latest.merge(trend, on=['species_name', 'location'], how='left')
    latest['risk_numeric'] = latest['risk_level'].map(RISK_SCORE).fillna(0)

    # Priority score: weight risk and negative trend (decline)
    decline_component = (-latest['trend_slope']).clip(lower=0)
    latest['priority_score'] = latest['risk_numeric'] * 1.5 + (decline_component / (latest['population_count'] + 1e-6)) * 100

    cols = [
        'date','location','species_name','population_count','base_population','risk_level',
        'human_activity','habitat_quality','hunting_pressure','trend_slope','priority_score'
    ]
    for c in cols:
        if c not in latest.columns:
            latest[c] = np.nan
    result = latest[cols].sort_values('priority_score', ascending=False)
    return result


def recommend_actions(hotspots: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """
    Recommend actions for top hotspots with simple cost/effect estimates.
    Heuristic rules based on drivers: human_activity, habitat_quality, hunting_pressure, and risk level.
    """
    recs = []
    top = hotspots.head(top_k)
    for _, row in top.iterrows():
        actions: List[Dict] = []
        # Base weights
        risk = row.get('risk_level', 'Low')
        ha = float(row.get('human_activity', 0) or 0)
        hq = float(row.get('habitat_quality', 1) or 1)
        hp = float(row.get('hunting_pressure', 0) or 0)

        if hq < 0.6:
            actions.append({
                'action_type': 'habitat_restoration',
                'estimated_effect_pct': 20 if risk == 'High' else 12,
                'estimated_cost_usd': 50000,
                'rationale': 'Low habitat quality'
            })
        if ha > 0.5:
            actions.append({
                'action_type': 'public_education',
                'estimated_effect_pct': 8 if risk == 'High' else 5,
                'estimated_cost_usd': 15000,
                'rationale': 'High human disturbance'
            })
        if hp > 0.2:
            actions.append({
                'action_type': 'hunting_regulation',
                'estimated_effect_pct': 10,
                'estimated_cost_usd': 5000,
                'rationale': 'Elevated hunting pressure'
            })
        # Fallback
        if not actions:
            actions.append({
                'action_type': 'monitoring_program',
                'estimated_effect_pct': 3,
                'estimated_cost_usd': 5000,
                'rationale': 'Maintain oversight'
            })

        for a in actions:
            recs.append({
                'date': row['date'],
                'location': row['location'],
                'species_name': row['species_name'],
                'risk_level': row['risk_level'],
                'priority_score': row['priority_score'],
                **a,
                'estimated_roi': a['estimated_effect_pct'] / max(1.0, a['estimated_cost_usd'] / 1000.0)
            })

    recs_df = pd.DataFrame(recs)
    if not recs_df.empty:
        recs_df.sort_values(['priority_score', 'estimated_roi'], ascending=[False, False], inplace=True)
    return recs_df
