"""
Configuration loader with YAML/JSON support, environment overrides, and CLI overrides.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from .schemas import RunConfig


def load_config(path: str) -> RunConfig:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        path: Path to config file
        
    Returns:
        RunConfig validated instance
        
    Raises:
        ValueError: If file format is unsupported or YAML is required but PyYAML not installed
    """
    config_path = Path(path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    suffix = config_path.suffix.lower()
    
    if suffix in [".yaml", ".yml"]:
        if not HAS_YAML:
            raise ValueError(
                "YAML config file requires PyYAML. Install with: pip install PyYAML\n"
                "Alternatively, use JSON config file."
            )
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    elif suffix == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {suffix}. Use .yaml, .yml, or .json")
    
    # Validate and return
    return RunConfig(**config_dict)


def apply_env_overrides(cfg: RunConfig) -> RunConfig:
    """
    Apply environment variable overrides to configuration.
    
    Environment variables must follow pattern: IOBT__{section}__{key}
    Example: IOBT__execution__slippage_bps=2
    
    Args:
        cfg: Base RunConfig
        
    Returns:
        RunConfig with environment overrides applied
    """
    env_prefix = "IOBT__"
    
    # Build override dict from environment
    overrides: Dict[str, Any] = {}
    
    for key, value in os.environ.items():
        if not key.startswith(env_prefix):
            continue
        
        # Parse key: IOBT__execution__slippage_bps -> ["execution", "slippage_bps"]
        # Or: IOBT__selector__liquidity__min_bid -> ["selector", "liquidity", "min_bid"]
        # Normalize to lowercase for Windows compatibility (env vars are often uppercase)
        parts = key[len(env_prefix):].lower().split("__")
        if len(parts) < 2:
            continue
        
        section = parts[0]
        nested_keys = parts[1:]  # Can be multiple levels deep
        
        # Parse value (try JSON first, fallback to string)
        try:
            parsed_value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            # Try parsing as number or boolean
            if value.lower() == "true":
                parsed_value = True
            elif value.lower() == "false":
                parsed_value = False
            elif value.lower() == "null":
                parsed_value = None
            else:
                try:
                    if "." in value:
                        parsed_value = float(value)
                    else:
                        parsed_value = int(value)
                except ValueError:
                    parsed_value = value
        
        if section not in overrides:
            overrides[section] = {}
        
        # Build nested structure
        current = overrides[section]
        for i, key_part in enumerate(nested_keys[:-1]):
            if key_part not in current:
                current[key_part] = {}
            current = current[key_part]
        
        # Set final value
        current[nested_keys[-1]] = parsed_value
    
    if not overrides:
        return cfg
    
    # Merge overrides into config using model_dump and model_validate
    config_dict = cfg.model_dump()
    
    for section, values in overrides.items():
        if section in config_dict:
            # Deep merge for nested dicts
            if isinstance(config_dict[section], dict) and isinstance(values, dict):
                config_dict[section] = _deep_merge(config_dict[section], values)
            else:
                config_dict[section].update(values)
        else:
            config_dict[section] = values
    
    # Re-validate
    return RunConfig(**config_dict)


def apply_cli_overrides(cfg: RunConfig, sets: List[str]) -> RunConfig:
    """
    Apply CLI --set key=value overrides to configuration.
    
    Supports nested keys: execution.slippage_bps=2 or selector.liquidity.min_bid=2.0
    Uses json.loads for typed values, falls back to string.
    
    Args:
        cfg: Base RunConfig
        sets: List of "key=value" strings from CLI --set flags
        
    Returns:
        RunConfig with CLI overrides applied
    """
    if not sets:
        return cfg
    
    overrides: Dict[str, Any] = {}
    
    for set_str in sets:
        if "=" not in set_str:
            raise ValueError(f"Invalid --set format: {set_str}. Expected 'key=value'")
        
        key_str, value_str = set_str.split("=", 1)
        
        # Parse nested key: "execution.slippage_bps" -> ["execution", "slippage_bps"]
        # Or "selector.liquidity.min_bid" -> ["selector", "liquidity", "min_bid"]
        key_parts = key_str.split(".")
        if len(key_parts) < 2:
            raise ValueError(f"Invalid --set key format: {key_str}. Expected 'section.key' or 'section.nested.key'")
        
        section = key_parts[0]
        nested_keys = key_parts[1:]  # Can be multiple levels deep
        
        # Parse value (try JSON first, fallback to string)
        try:
            parsed_value = json.loads(value_str)
        except (json.JSONDecodeError, ValueError):
            # Fallback to string, but try common types
            if value_str.lower() == "true":
                parsed_value = True
            elif value_str.lower() == "false":
                parsed_value = False
            elif value_str.lower() == "null":
                parsed_value = None
            else:
                # Try to parse as number
                try:
                    if "." in value_str:
                        parsed_value = float(value_str)
                    else:
                        parsed_value = int(value_str)
                except ValueError:
                    parsed_value = value_str
        
        # Build nested dict structure
        if section not in overrides:
            overrides[section] = {}
        
        # Navigate/create nested structure
        current = overrides[section]
        for i, key in enumerate(nested_keys[:-1]):
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set final value
        current[nested_keys[-1]] = parsed_value
    
    # Merge overrides into config
    config_dict = cfg.model_dump()
    
    for section, values in overrides.items():
        if section in config_dict:
            # Deep merge for nested dicts
            if isinstance(config_dict[section], dict) and isinstance(values, dict):
                config_dict[section] = _deep_merge(config_dict[section], values)
            else:
                config_dict[section] = values
        else:
            config_dict[section] = values
    
    # Re-validate
    return RunConfig(**config_dict)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result

