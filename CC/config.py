import os

class Config:
    """
    Central configuration class for the GARCH-Copula project.
    """
    # --- Base Directories ---
    BASE_DIR = "CC"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "model")
    SIMULATION_DIR = os.path.join(BASE_DIR, "simulation")
    ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")

    # --- 1. Data Processing Configuration (for 1_data_processing.py) ---
    START_DATE = "2007-01-01"
    END_DATE = "2025-06-01"
    TICKERS = ["^GSPC", "^GDAXI"]

    # --- 2. Marginal Model Configuration (for 2_marginal_model_estimation.py) ---
    IN_SAMPLE_RATIO = 0.80  # 80% for training, 20% for testing
    # GARCH model specifications
    VOL_FAMILIES = {
        "GARCH":  dict(vol="GARCH",  o=0),
        "GJR":    dict(vol="GARCH",  o=1),
        "EGARCH": dict(vol="EGARCH"),
        "APARCH": dict(vol="APARCH")
    }
    DISTRIBUTIONS = ["t", "skewt", "ged"]
    PQ_GRID = [(1, 1)]
    MEAN_SPEC = {"Constant": dict(mean="Constant"), 
                 "AR":       dict(mean="AR", lags=1)}

    # --- 3. Copula Simulation Configuration (for 3_copula_simulation.py) ---
    # WORKFLOW CONTROL: Set to True for the initial run to find best parameters.
    # Set to False to skip tuning and use the saved 'best_config.pkl' for faster runs.
    RUN_OPTUNA_TUNING = False

    SIMS_PER_DAY = 25000
    OOS_START_DATE = "2020-01-02"
    N_JOBS = -1  # Use all available CPU cores

    # --- 4. Backtesting Configuration (for 4_backtesting_analysis.py) ---
    ALPHA = 0.01  # For 99% VaR / ES
    MODELS_TO_TEST = ("Gaussian", "StudentT", "Clayton", "Gumbel")
    
    # --- Portfolio Constraints ---
    WEIGHT_FLOOR = 0.25
    WEIGHT_CAP = 0.75