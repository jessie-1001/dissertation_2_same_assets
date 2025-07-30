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