from dataclasses import asdict, dataclass
from typing import Final


WINDOW_SIZE: Final[int] = 10
FORECAST_STEPS: Final[int] = 4
NUM_FEATURES: Final[int] = 20
FEATURE_DIM: Final[int] = NUM_FEATURES

# Explicit dataset-derived attack category groups.
# NOTE: These groups represent heuristic operational categories derived from raw dataset
# attack classes (e.g. UNSW-NB15 attack_cat and CIC-IDS2017 label). They are NOT verified
# MITRE ATT&CK enterprise technique attributions.
ATTACK_CATEGORY_GROUPS: Final[list[str]] = [
    "Normal",          # Index 0: Benign / Normal background traffic
    "Reconnaissance",  # Index 1: PortScan, Network Probes
    "Analysis",        # Index 2: Bot, Security Scans
    "Fuzzers",         # Index 3: Protocol/Format fuzzing
    "Exploits",        # Index 4: Web exploits, Brute Force
    "DoS",             # Index 5: Denial of Service, DDoS
    "Generic",         # Index 6: Generic/unclassified attacks, Patator
    "Shellcode",       # Index 7: Buffer/stack payload execution
    "Backdoor",        # Index 8: Infiltration, C2/backdoors
    "Worms",           # Index 9: Self-propagating malware
]

NUM_ATTACK_CATEGORIES: Final[int] = len(ATTACK_CATEGORY_GROUPS)
IGNORE_CATEGORY_INDEX: Final[int] = -100  # Mask index for unmapped or missing categorical labels

# Legacy alias for backward compatibility with existing tests
MITRE_STAGE_MAP: dict[str, dict[str, str]] = {
    "Normal": {
        "tactic": "None",
        "tactic_id": "N/A",
        "technique_id": "N/A",
    },
    "Reconnaissance": {
        "tactic": "Reconnaissance",
        "tactic_id": "TA0043",
        "technique_id": "T1595",
    },
    "Analysis": {
        "tactic": "Discovery",
        "tactic_id": "TA0007",
        "technique_id": "T1046",
    },
    "Fuzzers": {
        "tactic": "Discovery",
        "tactic_id": "TA0007",
        "technique_id": "T1046",
    },
    "Exploits": {
        "tactic": "Execution",
        "tactic_id": "TA0002",
        "technique_id": "T1203",
    },
    "DoS": {
        "tactic": "Impact",
        "tactic_id": "TA0040",
        "technique_id": "T1499",
    },
    "Generic": {
        "tactic": "Defense Evasion",
        "tactic_id": "TA0005",
        "technique_id": "T1027",
    },
    "Shellcode": {
        "tactic": "Execution",
        "tactic_id": "TA0002",
        "technique_id": "T1059",
    },
    "Backdoor": {
        "tactic": "Persistence",
        "tactic_id": "TA0003",
        "technique_id": "T1505",
    },
    "Worms": {
        "tactic": "Lateral Movement",
        "tactic_id": "TA0008",
        "technique_id": "T1210",
    },
}


@dataclass
class ForecastOutput:
    current_stage: str
    forecast_probs: list[float]
    predicted_next_stage: str
    top_features: list[dict[str, float | str]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def get_mock_forecast() -> ForecastOutput:
    return ForecastOutput(
        current_stage="Reconnaissance",
        forecast_probs=[0.58, 0.24, 0.12, 0.06],
        predicted_next_stage="Exploits",
        top_features=[
            {"feature": "sbytes", "importance": 0.28},
            {"feature": "dbytes", "importance": 0.21},
            {"feature": "dur", "importance": 0.17},
            {"feature": "rate", "importance": 0.14},
            {"feature": "sload", "importance": 0.11},
        ],
    )
