import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

class DataGenerator:
    """
    Generate realistic ecological datasets for ML training
    """
    
    def __init__(self):
        self.species_profiles = {
            'Red Fox': {
                'base_population': 450,
                'growth_rate': 0.15,
                'temp_sensitivity': 0.02,
                'habitat_need': 0.8,
                'predator_impact': 0.1,
                'type': 'predator',
                'is_native': True
            },
            'Gray Wolf': {
                'base_population': 120,
                'growth_rate': 0.08,
                'temp_sensitivity': 0.01,
                'habitat_need': 1.2,
                'predator_impact': 0.0,
                'type': 'apex_predator',
                'is_native': True
            },
            'White-tailed Deer': {
                'base_population': 800,
                'growth_rate': 0.25,
                'temp_sensitivity': 0.015,
                'habitat_need': 0.6,
                'predator_impact': 0.3,
                'type': 'herbivore',
                'is_native': True
            },
            'European Starling': {
                'base_population': 1200,
                'growth_rate': 0.35,
                'temp_sensitivity': 0.025,
                'habitat_need': 0.4,
                'predator_impact': 0.2,
                'type': 'invasive_bird',
                'is_native': False
            },
            'Feral Cat': {
                'base_population': 300,
                'growth_rate': 0.4,
                'temp_sensitivity': 0.01,
                'habitat_need': 0.3,
                'predator_impact': 0.05,
                'type': 'invasive_predator',
                'is_native': False
            }
        }
    
    def generate_environmental_data(self, years=5, locations=['Forest_A', 'Grassland_B', 'Wetland_C']):
        """
        Generate environmental time series data
        """
        data = []
        start_date = datetime(2019, 1, 1)
        
        for location in locations:
            for year in range(years):
                for month in range(1, 13):
                    date = start_date + timedelta(days=year*365 + month*30)
                    
                    # Seasonal temperature pattern
                    temp_base = 12 + 8 * np.sin((month - 3) * np.pi / 6)
                    temperature = temp_base + np.random.normal(0, 3)
                    
                    # Seasonal rainfall pattern
                    rain_base = 60 + 40 * np.sin((month - 1) * np.pi / 6)
                    rainfall = max(0, rain_base + np.random.normal(0, 20))
                    
                    # Other environmental factors
                    humidity = 60 + 20 * np.sin((month - 2) * np.pi / 6) + np.random.normal(0, 5)
                    wind_speed = 15 + np.random.normal(0, 5)
                    
                    # Human disturbance factors
                    human_activity = np.random.beta(2, 5)  # Skewed toward low activity
                    hunting_pressure = np.random.beta(1, 4) if month in [9, 10, 11] else 0
                    
                    # Habitat quality (degradation over time)
                    habitat_quality = max(0.3, 1.0 - 0.02 * year + np.random.normal(0, 0.1))
                    
                    data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'year': date.year,
                        'month': month,
                        'location': location,
                        'temperature': round(temperature, 2),
                        'rainfall': round(rainfall, 2),
                        'humidity': round(max(20, min(95, humidity)), 2),
                        'wind_speed': round(max(0, wind_speed), 2),
                        'human_activity': round(human_activity, 3),
                        'hunting_pressure': round(hunting_pressure, 3),
                        'habitat_quality': round(habitat_quality, 3),
                        'season': ['Winter', 'Winter', 'Spring', 'Spring', 'Spring',
                                 'Summer', 'Summer', 'Summer', 'Fall', 'Fall', 'Fall', 'Winter'][month-1]
                    })
        
        return pd.DataFrame(data)
    
    def generate_species_population_data(self, environmental_data):
        """
        Generate species population data based on environmental factors
        """
        population_data = []
        
        for _, env_row in environmental_data.iterrows():
            for species_name, profile in self.species_profiles.items():
                # Calculate population based on environmental factors
                base_pop = profile['base_population']
                
                # Environmental effects
                temp_effect = 1 + profile['temp_sensitivity'] * (env_row['temperature'] - 15)
                rain_effect = 1 + 0.005 * min(env_row['rainfall'], 100) - 0.002 * max(0, env_row['rainfall'] - 100)
                humidity_effect = 1 + 0.002 * (env_row['humidity'] - 60)
                habitat_effect = env_row['habitat_quality'] ** profile['habitat_need']
                
                # Human impact
                human_impact = 1 - 0.3 * env_row['human_activity']
                hunting_impact = 1 - 0.5 * env_row['hunting_pressure'] if profile['type'] in ['predator', 'herbivore'] else 1
                
                # Seasonal effects
                seasonal_multiplier = {
                    'Spring': 1.2 if profile['type'] == 'herbivore' else 1.1,
                    'Summer': 1.1,
                    'Fall': 1.0,
                    'Winter': 0.8 if profile['type'] != 'invasive_bird' else 0.7
                }[env_row['season']]
                
                # Calculate population
                population = base_pop * temp_effect * rain_effect * humidity_effect * habitat_effect * human_impact * hunting_impact * seasonal_multiplier
                
                # Add noise
                population = max(5, population + np.random.normal(0, population * 0.15))
                
                # Population change from previous month (simplified)
                pop_change = np.random.normal(profile['growth_rate'] / 12, 0.05)
                
                # Risk assessment
                if population < base_pop * 0.6:
                    risk_level = 'High'
                elif population < base_pop * 0.8:
                    risk_level = 'Medium'
                else:
                    risk_level = 'Low'
                
                # Conservation status
                if population < base_pop * 0.3:
                    conservation_status = 'Critical'
                elif population < base_pop * 0.5:
                    conservation_status = 'Endangered'
                elif population < base_pop * 0.7:
                    conservation_status = 'Vulnerable'
                else:
                    conservation_status = 'Stable'
                
                population_data.append({
                    'date': env_row['date'],
                    'year': env_row['year'],
                    'month': env_row['month'],
                    'location': env_row['location'],
                    'species_name': species_name,
                    'species_type': profile['type'],
                    'is_native': profile['is_native'],
                    'population_count': round(population, 0),
                    'population_change_rate': round(pop_change, 4),
                    'risk_level': risk_level,
                    'conservation_status': conservation_status,
                    'base_population': base_pop,
                    # Environmental factors
                    'temperature': env_row['temperature'],
                    'rainfall': env_row['rainfall'],
                    'humidity': env_row['humidity'],
                    'human_activity': env_row['human_activity'],
                    'hunting_pressure': env_row['hunting_pressure'],
                    'habitat_quality': env_row['habitat_quality'],
                    'season': env_row['season']
                })
        
        return pd.DataFrame(population_data)
    
    def generate_interaction_data(self, population_data):
        """
        Generate species interaction data (predator-prey, competition)
        """
        interactions = []
        
        # Group by date and location
        grouped = population_data.groupby(['date', 'location'])
        
        for (date, location), group in grouped:
            species_pops = group.set_index('species_name')['population_count'].to_dict()
            
            # Predator-prey interactions
            predator_prey_pairs = [
                ('Gray Wolf', 'White-tailed Deer'),
                ('Red Fox', 'European Starling'),
                ('Feral Cat', 'European Starling')
            ]
            
            for predator, prey in predator_prey_pairs:
                if predator in species_pops and prey in species_pops:
                    predator_pop = species_pops[predator]
                    prey_pop = species_pops[prey]
                    
                    # Calculate interaction strength
                    interaction_strength = min(1.0, predator_pop / (predator_pop + prey_pop))
                    prey_pressure = interaction_strength * 0.3  # Max 30% impact
                    
                    interactions.append({
                        'date': date,
                        'location': location,
                        'interaction_type': 'predation',
                        'species_1': predator,
                        'species_2': prey,
                        'interaction_strength': round(interaction_strength, 3),
                        'impact_on_species_2': round(prey_pressure, 3)
                    })
            
            # Competition interactions (same type species)
            competition_pairs = [
                ('Red Fox', 'Feral Cat'),  # Both predators
                ('European Starling', 'White-tailed Deer')  # Resource competition
            ]
            
            for species1, species2 in competition_pairs:
                if species1 in species_pops and species2 in species_pops:
                    pop1, pop2 = species_pops[species1], species_pops[species2]
                    total_pop = pop1 + pop2
                    
                    competition_strength = min(0.5, (pop1 + pop2) / 1000)  # Increases with density
                    
                    interactions.append({
                        'date': date,
                        'location': location,
                        'interaction_type': 'competition',
                        'species_1': species1,
                        'species_2': species2,
                        'interaction_strength': round(competition_strength, 3),
                        'impact_on_species_1': round(competition_strength * pop2 / total_pop, 3),
                        'impact_on_species_2': round(competition_strength * pop1 / total_pop, 3)
                    })
        
        return pd.DataFrame(interactions)
    
    def generate_conservation_actions_data(self, population_data):
        """
        Generate conservation actions and their outcomes
        """
        actions = []
        
        # Sample conservation actions based on species status
        action_types = {
            'habitat_restoration': {'cost': 50000, 'effectiveness': 0.3, 'duration': 24},
            'invasive_control': {'cost': 25000, 'effectiveness': 0.4, 'duration': 6},
            'breeding_program': {'cost': 75000, 'effectiveness': 0.5, 'duration': 36},
            'hunting_regulation': {'cost': 5000, 'effectiveness': 0.2, 'duration': 12},
            'public_education': {'cost': 15000, 'effectiveness': 0.15, 'duration': 12}
        }
        
        # Randomly assign actions to species with risk
        high_risk_species = population_data[population_data['risk_level'] == 'High']
        
        for _, row in high_risk_species.sample(n=min(100, len(high_risk_species))).iterrows():
            action_type = np.random.choice(list(action_types.keys()))
            action_info = action_types[action_type]
            
            # Calculate expected outcome
            base_effectiveness = action_info['effectiveness']
            actual_effectiveness = base_effectiveness * np.random.uniform(0.5, 1.5)
            
            actions.append({
                'date': row['date'],
                'location': row['location'],
                'species_name': row['species_name'],
                'action_type': action_type,
                'cost_usd': action_info['cost'],
                'planned_duration_months': action_info['duration'],
                'expected_effectiveness': round(base_effectiveness, 2),
                'actual_effectiveness': round(actual_effectiveness, 2),
                'population_before': row['population_count'],
                'population_after': round(row['population_count'] * (1 + actual_effectiveness), 0),
                'success': actual_effectiveness > base_effectiveness * 0.7
            })
        
        return pd.DataFrame(actions)
    
    def create_complete_dataset(self, years=5, save_to_csv=True):
        """
        Generate complete ecological monitoring dataset
        """
        print("Generating environmental data...")
        env_data = self.generate_environmental_data(years=years)
        
        print("Generating species population data...")
        pop_data = self.generate_species_population_data(env_data)
        
        print("Generating species interaction data...")
        interaction_data = self.generate_interaction_data(pop_data)
        
        print("Generating conservation actions data...")
        actions_data = self.generate_conservation_actions_data(pop_data)
        
        if save_to_csv:
            pop_data.to_csv('data/ecological_population_data.csv', index=False)
            env_data.to_csv('data/environmental_data.csv', index=False)
            interaction_data.to_csv('data/species_interactions.csv', index=False)
            actions_data.to_csv('data/conservation_actions.csv', index=False)
            
            print("Datasets saved to data/ directory")
        
        return {
            'population': pop_data,
            'environmental': env_data,
            'interactions': interaction_data,
            'actions': actions_data
        }

if __name__ == "__main__":
    generator = DataGenerator()
    datasets = generator.create_complete_dataset(years=5, save_to_csv=True)
    
    print("\nDataset Summary:")
    for name, df in datasets.items():
        print(f"{name.capitalize()}: {len(df)} records")
        
    print(f"\nSpecies included: {datasets['population']['species_name'].unique()}")
    print(f"Locations: {datasets['population']['location'].unique()}")
    print(f"Date range: {datasets['population']['date'].min()} to {datasets['population']['date'].max()}")
