from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    feature_columns: tuple[str, ...] = ("rooms", "sqft")
    target_column: str = "price"
    test_size: float = 0.2
    random_state: int = 42


DEFAULT_CONFIG = TrainingConfig()
