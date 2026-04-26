import pandas as pd
import numpy as np
import os

def generate_data(num_samples=500000):
    print(f"Generating {num_samples} samples of synthetic ad data...")
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(18, 70, num_samples),
        'gender': np.random.randint(0, 2, num_samples), # 0: Female, 1: Male
        'time_spent': np.random.uniform(1, 60, num_samples), # minutes
        'device_type': np.random.randint(0, 3, num_samples), # 0: Mobile, 1: Desktop, 2: Tablet
        'ad_category': np.random.randint(0, 5, num_samples), # 0: Tech, 1: Fashion, 2: Sports, 3: Food, 4: Travel
        'ad_position': np.random.randint(0, 4, num_samples), # 0: Top, 1: Sidebar, 2: Bottom, 3: Popup
        'link_length': np.random.randint(10, 200, num_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate target 'clicked' based on some logic
    # Click probability is higher for younger people, more time spent, and mobile devices
    prob = (
        (70 - df['age']) * 0.01 + 
        df['time_spent'] * 0.01 + 
        (df['device_type'] == 0) * 0.2 + 
        (df['ad_position'] == 0) * 0.1
    )
    # Add some noise
    prob += np.random.normal(0, 0.1, num_samples)
    
    # Normalize and convert to binary
    df['clicked'] = (prob > prob.median()).astype(int)
    
    output_path = 'ad_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    generate_data()
