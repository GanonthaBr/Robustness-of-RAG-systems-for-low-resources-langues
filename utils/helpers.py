"""Utility functions"""

import json
import os
from typing import Any, Dict


def save_json(data: Any, filepath: str):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Dict:
    """Load data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    import logging
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )