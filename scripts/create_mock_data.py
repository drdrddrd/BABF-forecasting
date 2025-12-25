import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration ---
OUTPUT_DIR = Path("data/preprocessed/final_nfi_ch2018_merged")
SCENARIOS = ["RCP26", "RCP45", "RCP85"]
NUM_PLOTS = 100   # Number of unique plots
SEED = 35         # Fixed seed

def generate_mock_files():
    # --- 1. Setup Directory ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"--- Generating Mock Data ---")
    print(f"Target Directory: {OUTPUT_DIR.resolve()}\n")

    np.random.seed(SEED)

    # --- 2. Generate Static Site Data ---
    clnrs = np.arange(1, NUM_PLOTS + 1)
    site_df = pd.DataFrame({
        'CLNR': clnrs,
        'ASPECT25': np.random.uniform(0, 360, NUM_PLOTS),
        'SLOPE25': np.random.uniform(0, 90, NUM_PLOTS),
        'PH': np.random.normal(6.5, 1.0, NUM_PLOTS).clip(3, 9),
        'Z25': np.random.normal(1200, 300, NUM_PLOTS).clip(300, 2500),
        'BEWIRTINT1': np.random.choice([1.0, 2.0, 3.0], NUM_PLOTS),
        'NAISHSTKOMB': np.random.choice(range(1, 8), NUM_PLOTS)
    })

    # --- 3. Define Time Steps ---
    history_invnrs = [150, 250, 350, 450, 550]
    future_invnrs = [650, 750, 850, 950, 1050, 1150, 1250]
    all_invnrs = history_invnrs + future_invnrs
    base_year = 1985

    # --- 4. Loop to Create Each RCP File ---
    for rcp in SCENARIOS:
        print(f"Processing {rcp}...")
        
        rows = []
        temp_offset = 0 if rcp == "RCP26" else (1.5 if rcp == "RCP45" else 3.0)

        for inv_idx, invnr in enumerate(all_invnrs):
            current_year = base_year + (inv_idx * 10)
            
            for _, site in site_df.iterrows():
                
                # --- A. Generate Climate (Exists for ALL rows) ---
                trend = (inv_idx * 0.1 * temp_offset)
                climate_data = {
                    'mean_dry_days_count': np.random.normal(230 + trend, 10),
                    'mean_frost_days_count': np.random.normal(100 - trend*2, 20),
                    'mean_gdd_sum': np.random.normal(1800 + trend*50, 400),
                    'mean_pr_sum': np.random.normal(1300, 300),
                    'mean_pr_variance': np.random.normal(25, 5),
                    'mean_tas_mean': np.random.normal(8.0 + trend, 2.5),
                    'mean_tas_variance': np.random.normal(50, 5),
                    'mean_tasmax_mean': np.random.normal(12.0 + trend, 3.0),
                    'mean_tasmax_variance': np.random.normal(70, 10),
                    'mean_tasmin_mean': np.random.normal(4.0 + trend, 2.0),
                    'mean_tasmin_variance': np.random.normal(40, 5),
                }

                # --- B. Generate Tree Data ---
                if invnr <= 550:
                    # History (Train/Test/Start): Valid values
                    basfph = np.random.normal(30, 15)
                    basfph = np.clip(basfph, 0.5, 95)
                    hwsw_prop = np.random.uniform(0, 1)
                    basfph_sq = basfph ** 2
                else:
                    # Future (Prediction): NaNs
                    basfph = np.nan
                    hwsw_prop = np.nan
                    basfph_sq = np.nan

                # --- C. Build Row ---
                row = {
                    'CLNR': site['CLNR'],
                    'INVNR': invnr,
                    'INVYR': current_year,
                    'Time_Diff_years': 10.0, 
                    'ASPECT25': site['ASPECT25'],
                    'SLOPE25': site['SLOPE25'],
                    'PH': site['PH'],
                    'Z25': site['Z25'],
                    'BEWIRTINT1': site['BEWIRTINT1'],
                    'NAISHSTKOMB': site['NAISHSTKOMB'],
                    'BASFPH': basfph,
                    'BASFPH_squared': basfph_sq,
                    'HWSW_prop': hwsw_prop,
                }
                row.update(climate_data)
                rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows)

        # --- 5. Calculate Targets ---
        df = df.sort_values(by=['CLNR', 'INVNR'])
        df['BASFPH_next_INVNR'] = df.groupby('CLNR')['BASFPH'].shift(-1)
        df['HWSW_prop_next_INVNR'] = df.groupby('CLNR')['HWSW_prop'].shift(-1)
        
        # --- 6. Formatting ---
        df['BEWIRTINT1'] = df['BEWIRTINT1'].astype(float)
        df['NAISHSTKOMB'] = df['NAISHSTKOMB'].astype(int)

        # --- 7. Save ---
        filename = f"MOCK_DATA_{rcp}.csv"
        file_path = OUTPUT_DIR / filename
        df.to_csv(file_path, index=False)
        print(f" -> Saved {filename}")

    print("\nAll 3 mock files created.")

if __name__ == "__main__":
    generate_mock_files()