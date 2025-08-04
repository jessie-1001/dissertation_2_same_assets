best_config = {
    "refit_freq": 5,
    "beta_cap": 0.9753773006054616,
    "tail_adj": 1.143264736356668,
    "sims": 25000,
    "alpha": 0.01
}
from config import Config
import pickle
with open(f"{Config.SIMULATION_DIR}/best_config.pkl", "wb") as f:
    pickle.dump(best_config, f)