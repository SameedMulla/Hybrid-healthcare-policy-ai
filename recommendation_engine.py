"""
Recommendation Engine for Healthcare Resource Allocation
---------------------------------------------------------
Hybrid Rule-based + ML system for generating state-wise
healthcare recommendations covering:
- Budget allocation
- Hospital & infrastructure needs
- Doctor deployment
- Vaccine distribution
- Supply chain optimization
"""

import pandas as pd
import numpy as np
import json
import os


class HealthcareRecommendationEngine:
    """
    Generates evidence-based healthcare recommendations for Indian states.
    Uses WHO benchmarks, national targets, and ML predictions.
    """

    # WHO and National Benchmarks
    BENCHMARKS = {
        'doctor_per_1000': 1.0,           # WHO minimum
        'hospital_beds_per_1000': 3.0,     # WHO recommendation
        'nurses_per_1000': 3.0,            # WHO recommendation
        'icu_beds_per_100k': 8.0,          # Target
        'vaccine_coverage_pct': 95.0,      # Herd immunity target
        'mmr_target': 70,                  # SDG target
        'imr_target': 15,                  # National target
        'budget_per_capita_min': 3000,     # Rs per person per year
        'phc_per_30k': 1,                  # 1 PHC per 30,000 rural
        'chc_per_120k': 1,                 # 1 CHC per 120,000
    }

    # Priority weights for scoring
    PRIORITY_WEIGHTS = {
        'doctor_gap': 0.20,
        'bed_gap': 0.15,
        'vaccine_gap': 0.15,
        'budget_gap': 0.15,
        'mmr_gap': 0.10,
        'imr_gap': 0.10,
        'infra_gap': 0.10,
        'icu_gap': 0.05,
    }

    def __init__(self, data_path="data/india_healthcare_data.csv",
                 predictions_path="data/predictions_2028_29.json"):
        """Initialize with historical data and predictions."""
        self.df = pd.read_csv(data_path)
        self.latest_year = self.df['year'].max()
        self.latest_data = self.df[self.df['year'] == self.latest_year].copy()

        # Load predictions if available
        self.predictions = None
        if os.path.exists(predictions_path):
            with open(predictions_path, 'r') as f:
                pred_list = json.load(f)
            self.predictions = pd.DataFrame(pred_list)

    def get_state_profile(self, state_name):
        """Get comprehensive profile for a state."""
        state_data = self.latest_data[
            self.latest_data['state'].str.lower() == state_name.lower()
        ]
        if state_data.empty:
            return None
        return state_data.iloc[0].to_dict()

    def calculate_gap_analysis(self, state_name):
        """Calculate gaps between current metrics and benchmarks."""
        profile = self.get_state_profile(state_name)
        if not profile:
            return None

        population_millions = profile['population_crore'] * 10
        population = profile['population_crore'] * 10000000  # actual population

        gaps = {
            'state': state_name,
            'population_crore': profile['population_crore'],

            # Doctor gap
            'current_doctors': profile['doctors_total'],
            'required_doctors': int(population / 1000 * self.BENCHMARKS['doctor_per_1000']),
            'doctor_gap': max(0, int(population / 1000 * self.BENCHMARKS['doctor_per_1000']) - profile['doctors_total']),
            'doctor_ratio': profile['doctor_per_1000'],
            'doctor_ratio_target': self.BENCHMARKS['doctor_per_1000'],

            # Hospital bed gap
            'current_beds_per_1000': profile['hospital_beds_per_1000'],
            'required_beds_per_1000': self.BENCHMARKS['hospital_beds_per_1000'],
            'beds_gap': max(0, int((self.BENCHMARKS['hospital_beds_per_1000'] - profile['hospital_beds_per_1000']) * population_millions * 1000 / 1000)),
            'additional_beds_needed': max(0, int((self.BENCHMARKS['hospital_beds_per_1000'] - profile['hospital_beds_per_1000']) * population / 1000)),

            # ICU gap
            'current_icu_beds': profile['icu_beds'],
            'required_icu_beds': int(population / 100000 * self.BENCHMARKS['icu_beds_per_100k']),
            'icu_gap': max(0, int(population / 100000 * self.BENCHMARKS['icu_beds_per_100k']) - profile['icu_beds']),

            # Vaccine coverage gap
            'current_vaccine_coverage': profile['vaccine_coverage_pct'],
            'target_vaccine_coverage': self.BENCHMARKS['vaccine_coverage_pct'],
            'vaccine_coverage_gap': max(0, self.BENCHMARKS['vaccine_coverage_pct'] - profile['vaccine_coverage_pct']),

            # Budget gap
            'current_budget_crore': profile['health_budget_crore'],
            'current_per_capita': profile['budget_per_capita_inr'],
            'target_per_capita': self.BENCHMARKS['budget_per_capita_min'],
            'budget_gap_crore': max(0, (self.BENCHMARKS['budget_per_capita_min'] - profile['budget_per_capita_inr']) * population / 10000000),

            # Maternal & child health
            'current_mmr': profile['maternal_mortality_ratio'],
            'target_mmr': self.BENCHMARKS['mmr_target'],
            'mmr_gap': max(0, profile['maternal_mortality_ratio'] - self.BENCHMARKS['mmr_target']),
            'current_imr': profile['infant_mortality_rate'],
            'target_imr': self.BENCHMARKS['imr_target'],
            'imr_gap': max(0, profile['infant_mortality_rate'] - self.BENCHMARKS['imr_target']),

            # Infrastructure
            'infra_gap_score': profile['infra_gap_score'],
            'hospitals_total': profile['hospitals_total'],
            'cold_chain_facilities': profile['cold_chain_facilities'],
        }

        return gaps

    def calculate_priority_score(self, state_name):
        """
        Calculate composite priority score (0-100) for resource allocation.
        Higher score = more urgent need.
        """
        gaps = self.calculate_gap_analysis(state_name)
        if not gaps:
            return None

        # Normalize each gap to 0-1 scale
        scores = {}

        # Doctor gap (relative to benchmark)
        scores['doctor_gap'] = min(1, max(0, 1 - gaps['doctor_ratio'] / self.BENCHMARKS['doctor_per_1000']))

        # Bed gap
        scores['bed_gap'] = min(1, max(0, 1 - gaps['current_beds_per_1000'] / self.BENCHMARKS['hospital_beds_per_1000']))

        # Vaccine gap
        scores['vaccine_gap'] = min(1, max(0, (self.BENCHMARKS['vaccine_coverage_pct'] - gaps['current_vaccine_coverage']) / 40))

        # Budget gap
        scores['budget_gap'] = min(1, max(0, 1 - gaps['current_per_capita'] / self.BENCHMARKS['budget_per_capita_min']))

        # MMR gap
        scores['mmr_gap'] = min(1, max(0, (gaps['current_mmr'] - self.BENCHMARKS['mmr_target']) / 150))

        # IMR gap
        scores['imr_gap'] = min(1, max(0, (gaps['current_imr'] - self.BENCHMARKS['imr_target']) / 40))

        # Infra gap (already 1-10 scale)
        scores['infra_gap'] = min(1, gaps['infra_gap_score'] / 10)

        # ICU gap
        scores['icu_gap'] = min(1, max(0, gaps['icu_gap'] / max(1, gaps['required_icu_beds'])))

        # Weighted composite
        composite = sum(
            scores[key] * self.PRIORITY_WEIGHTS[key]
            for key in self.PRIORITY_WEIGHTS
        )

        return {
            'state': state_name,
            'composite_score': round(composite * 100, 1),
            'category': self._categorize_priority(composite * 100),
            'component_scores': {k: round(v * 100, 1) for k, v in scores.items()},
            'gaps': gaps
        }

    def _categorize_priority(self, score):
        """Categorize priority level."""
        if score >= 60:
            return "CRITICAL"
        elif score >= 40:
            return "HIGH"
        elif score >= 25:
            return "MODERATE"
        else:
            return "LOW"

    def get_state_recommendations(self, state_name):
        """Generate comprehensive recommendations for a state."""
        priority = self.calculate_priority_score(state_name)
        if not priority:
            return None

        gaps = priority['gaps']
        profile = self.get_state_profile(state_name)
        recommendations = []

        # 1. Doctor Deployment
        if gaps['doctor_gap'] > 0:
            annual_target = min(gaps['doctor_gap'], int(gaps['doctor_gap'] * 0.15))  # 15% per year
            recommendations.append({
                'category': '👨‍⚕️ Doctor Deployment',
                'priority': 'HIGH' if gaps['doctor_ratio'] < 0.5 else 'MEDIUM',
                'current': f"{gaps['doctor_ratio']:.2f} per 1,000",
                'target': f"{self.BENCHMARKS['doctor_per_1000']} per 1,000",
                'gap': f"{gaps['doctor_gap']:,} doctors needed",
                'action': f"Recruit {annual_target:,} doctors/year. Deploy through rural service bonds. "
                         f"Establish telemedicine in {int(profile.get('phc_count', 0) * 0.5)} PHCs.",
                'estimated_cost_crore': round(annual_target * 12 / 10000, 0)  # Approx ₹12L per doctor per year
            })

        # 2. Hospital Infrastructure
        if gaps['additional_beds_needed'] > 0:
            new_hospitals = max(1, int(gaps['additional_beds_needed'] / 100))  # ~100 beds per hospital
            recommendations.append({
                'category': '🏥 Hospital Infrastructure',
                'priority': 'HIGH' if gaps['current_beds_per_1000'] < 1.0 else 'MEDIUM',
                'current': f"{gaps['current_beds_per_1000']:.2f} beds per 1,000",
                'target': f"{self.BENCHMARKS['hospital_beds_per_1000']} beds per 1,000",
                'gap': f"{gaps['additional_beds_needed']:,} additional beds needed",
                'action': f"Build {new_hospitals} new hospitals. Upgrade existing CHCs. "
                         f"Add {gaps['icu_gap']:,} ICU beds in district hospitals.",
                'estimated_cost_crore': round(gaps['additional_beds_needed'] * 0.25, 0)  # ~₹25L per bed
            })

        # 3. Vaccine Distribution
        if gaps['vaccine_coverage_gap'] > 5:
            recommendations.append({
                'category': '💉 Vaccine Coverage',
                'priority': 'HIGH' if gaps['current_vaccine_coverage'] < 75 else 'MEDIUM',
                'current': f"{gaps['current_vaccine_coverage']:.1f}%",
                'target': f"{self.BENCHMARKS['vaccine_coverage_pct']}%",
                'gap': f"{gaps['vaccine_coverage_gap']:.1f} percentage points",
                'action': f"Intensify Mission Indradhanush. Deploy {int(gaps['population_crore'] * 20)} "
                         f"mobile vaccination units. Expand cold chain by "
                         f"{max(10, int(gaps['cold_chain_facilities'] * 0.15))} facilities.",
                'estimated_cost_crore': round(gaps['population_crore'] * 80, 0)
            })

        # 4. Budget Enhancement
        if gaps['budget_gap_crore'] > 0:
            recommendations.append({
                'category': '💰 Budget Allocation',
                'priority': 'HIGH' if gaps['current_per_capita'] < 1500 else 'MEDIUM',
                'current': f"₹{gaps['current_per_capita']:,.0f} per capita",
                'target': f"₹{self.BENCHMARKS['budget_per_capita_min']:,} per capita",
                'gap': f"₹{gaps['budget_gap_crore']:,.0f} crore additional needed",
                'action': f"Increase state health budget by {int(gaps['budget_gap_crore'] / gaps['current_budget_crore'] * 100)}%. "
                         f"Prioritize primary healthcare (40%), hospital infrastructure (25%), "
                         f"human resources (20%), immunization (15%).",
                'estimated_cost_crore': round(gaps['budget_gap_crore'], 0)
            })

        # 5. Maternal & Child Health
        if gaps['mmr_gap'] > 0 or gaps['imr_gap'] > 0:
            recommendations.append({
                'category': '🤱 Maternal & Child Health',
                'priority': 'HIGH' if gaps['current_mmr'] > 100 or gaps['current_imr'] > 25 else 'MEDIUM',
                'current': f"MMR: {gaps['current_mmr']}, IMR: {gaps['current_imr']}",
                'target': f"MMR: {self.BENCHMARKS['mmr_target']}, IMR: {self.BENCHMARKS['imr_target']}",
                'gap': f"MMR gap: {gaps['mmr_gap']}, IMR gap: {gaps['imr_gap']}",
                'action': "Expand institutional delivery. Deploy skilled birth attendants. "
                         "Strengthen JSSK/JSY benefits. Establish special newborn care units. "
                         "Nutrition-health convergence programs.",
                'estimated_cost_crore': round(gaps['population_crore'] * 60, 0)
            })

        # 6. Supply Chain
        recommendations.append({
            'category': '🔗 Supply Chain',
            'priority': 'MEDIUM',
            'current': f"{gaps['cold_chain_facilities']} cold chain facilities",
            'target': f"{int(gaps['cold_chain_facilities'] * 1.3)} (30% expansion)",
            'gap': f"{int(gaps['cold_chain_facilities'] * 0.3)} additional needed",
            'action': "Deploy IoT-enabled temperature monitoring. Establish regional drug warehouses. "
                     "Implement digital inventory with auto-reorder. Pre-monsoon emergency stocking.",
            'estimated_cost_crore': round(gaps['population_crore'] * 20, 0)
        })

        return {
            'state': state_name,
            'priority_score': priority['composite_score'],
            'priority_category': priority['category'],
            'component_scores': priority['component_scores'],
            'recommendations': recommendations,
            'total_investment_needed_crore': sum(r.get('estimated_cost_crore', 0) for r in recommendations)
        }

    def get_predictions_for_state(self, state_name, year=2028):
        """Get ML predictions for a state."""
        if self.predictions is None:
            return None
        pred = self.predictions[
            (self.predictions['state'].str.lower() == state_name.lower()) &
            (self.predictions['year'] == year)
        ]
        if pred.empty:
            return None
        return pred.iloc[0].to_dict()

    def rank_all_states(self):
        """Rank all states by priority score."""
        rankings = []
        for state in self.latest_data['state'].unique():
            priority = self.calculate_priority_score(state)
            if priority:
                rankings.append({
                    'state': state,
                    'score': priority['composite_score'],
                    'category': priority['category']
                })
        rankings.sort(key=lambda x: x['score'], reverse=True)
        return rankings

    def get_national_summary(self):
        """Get national-level summary statistics."""
        data = self.latest_data
        return {
            'total_population_crore': round(data['population_crore'].sum(), 2),
            'total_hospitals': int(data['hospitals_total'].sum()),
            'total_doctors': int(data['doctors_total'].sum()),
            'total_nurses': int(data['nurses_total'].sum()),
            'total_budget_crore': int(data['health_budget_crore'].sum()),
            'avg_beds_per_1000': round(data['hospital_beds_per_1000'].mean(), 2),
            'avg_doctor_per_1000': round(data['doctor_per_1000'].mean(), 2),
            'avg_vaccine_coverage': round(data['vaccine_coverage_pct'].mean(), 1),
            'avg_life_expectancy': round(data['life_expectancy'].mean(), 1),
            'states_critical': len([s for s in self.latest_data['state'].unique()
                                    if self.calculate_priority_score(s) and
                                    self.calculate_priority_score(s)['category'] == 'CRITICAL']),
            'states_high': len([s for s in self.latest_data['state'].unique()
                                if self.calculate_priority_score(s) and
                                self.calculate_priority_score(s)['category'] == 'HIGH']),
        }

    def compare_states(self, state1, state2):
        """Compare two states side-by-side."""
        p1 = self.get_state_profile(state1)
        p2 = self.get_state_profile(state2)
        if not p1 or not p2:
            return None

        metrics = [
            'population_crore', 'health_budget_crore', 'budget_per_capita_inr',
            'hospitals_total', 'hospital_beds_per_1000', 'doctors_total',
            'doctor_per_1000', 'nurses_total', 'vaccine_coverage_pct',
            'infra_gap_score', 'maternal_mortality_ratio', 'infant_mortality_rate',
            'life_expectancy', 'icu_beds', 'cold_chain_facilities'
        ]

        comparison = {}
        for metric in metrics:
            comparison[metric] = {
                state1: p1.get(metric, 'N/A'),
                state2: p2.get(metric, 'N/A'),
                'better': state1 if self._is_better(metric, p1.get(metric, 0), p2.get(metric, 0)) else state2
            }

        return comparison

    def _is_better(self, metric, val1, val2):
        """Determine which value is better for a given metric."""
        # Lower is better for these metrics
        lower_better = ['infra_gap_score', 'maternal_mortality_ratio', 'infant_mortality_rate', 'disease_index']
        if metric in lower_better:
            return val1 < val2
        return val1 > val2

    def get_historical_trend(self, state_name, metric):
        """Get historical trend data for a state and metric."""
        state_data = self.df[self.df['state'].str.lower() == state_name.lower()]
        if state_data.empty:
            return None
        return state_data[['year', metric]].to_dict('records')


# -------------------------
# Test the engine
# -------------------------
if __name__ == "__main__":
    engine = HealthcareRecommendationEngine()

    print("=" * 60)
    print("Healthcare Recommendation Engine Test")
    print("=" * 60)

    # National summary
    summary = engine.get_national_summary()
    print(f"\n--- National Summary ({engine.latest_year}) ---")
    print(f"Total Population: {summary['total_population_crore']} crore")
    print(f"Total Hospitals: {summary['total_hospitals']:,}")
    print(f"Total Doctors: {summary['total_doctors']:,}")
    print(f"Total Budget: Rs {summary['total_budget_crore']:,} crore")
    print(f"Avg Vaccine Coverage: {summary['avg_vaccine_coverage']}%")

    # Rankings
    print(f"\n--- State Priority Rankings ---")
    rankings = engine.rank_all_states()
    for i, r in enumerate(rankings[:10]):
        print(f"   {i+1}. {r['state']}: {r['score']:.1f} ({r['category']})")

    # Sample recommendations
    print(f"\n--- Sample Recommendations: Bihar ---")
    recs = engine.get_state_recommendations("Bihar")
    if recs:
        print(f"Priority Score: {recs['priority_score']} ({recs['priority_category']})")
        for rec in recs['recommendations']:
            print(f"\n   {rec['category']} [{rec['priority']}]")
            print(f"   Current: {rec['current']}")
            print(f"   Target: {rec['target']}")
            print(f"   Gap: {rec['gap']}")
            print(f"   Est. Cost: Rs {rec.get('estimated_cost_crore', 0):,.0f} crore")

    print("\n" + "=" * 60)
    print("Recommendation Engine initialized successfully!")
