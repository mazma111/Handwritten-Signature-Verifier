from dataclasses import dataclass
from pathlib import Path


@dataclass
class CONFIG:
    dataset = Path('../datasets/CEDAR/')
    original_signatures_class = 'full_org'
    forged_signatures_class = 'full_forg'
    size = (350, 543)
    mean = 0.015957806259393692
    std = 0.08474896848201752
