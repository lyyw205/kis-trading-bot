# ml_train_seq_model_cr.py
from c_config import CR_UNIVERSE_STOCKS
from f_ml_train import train_seq_model_for_universe

if __name__ == "__main__":
    train_seq_model_for_universe(
        CR_UNIVERSE_STOCKS,
        region_filter="CR",
        model_setting_key="active_model_path_coin",
        note_prefix="[CR_COIN] ",
        model_dir="models_coin",
    )
