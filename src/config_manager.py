#!/usr/bin/env python3
"""
Configuration Management Module
Handles loading and validating configuration files

Created: 2025
"""

import json
import os
from typing import Dict, Any, Optional

class ConfigManager:
    """Configuration manager for face recognition system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config.json"
        self.config = self._get_default_config()
        
        if os.path.exists(self.config_path):
            self.load_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "min_face_size": 90,
            "face_threshold": 0.5,
            "recognition_threshold": 0.6,
            "liveness_threshold": 0.93,
            "max_blur": -25.0,
            "max_angle": 10.0,
            "max_database_items": 2000,
            "show_landmarks": True,
            "show_legend": True,
            "enable_liveness": True,
            "enable_blur_filter": True,
            "auto_add_faces": False,
            "database_path": "face_database_mobilefacenet.json",
            "images_directory": "images",
            "models": {
                "face_detector": {
                    "model_path": "models/opencv_face_detector_uint8.pb",
                    "config_path": "models/opencv_face_detector.pbtxt",
                    "input_width": 320,
                    "input_height": 240,
                    "confidence_threshold": 0.5,
                    "nms_threshold": 0.4
                },
                "feature_extractor": {
                    "model_path": "models/mobilefacenet.onnx",
                    "input_size": [112, 112],
                    "feature_dim": 128
                },
                "liveness_detector": {
                    "model_path": "models/liveness_model.onnx",
                    "threshold": 0.93
                }
            },
            "video": {
                "display_width": 640,
                "display_height": 480,
                "fps_buffer_size": 16
            },
            "colors": {
                "recognized": [255, 255, 255],
                "stranger": [80, 255, 255],
                "too_tiny": [255, 237, 178],
                "fake": [127, 127, 255],
                "landmarks": [0, 255, 255],
                "legend": [180, 180, 0]
            }
        }
    
    def load_config(self) -> bool:
        """
        Load configuration from file
        Returns:
            True if loaded successfully
        """
        try:
            with open(self.config_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Update default config with loaded values
            self._deep_update(self.config, loaded_config)
            
            print(f"Configuration loaded from {self.config_path}")
            return True
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
    
    def save_config(self) -> bool:
        """
        Save current configuration to file
        
        ⚠️ WARNING: This will OVERWRITE config.json!
        Only call this method if you explicitly want to save/reset config.
        Main application (app/cli.py) does NOT use this method.
        Config changes in app/cli.py are runtime-only and not persisted.
        
        Returns:
            True if saved successfully
        """
        try:
            # Create backup before overwriting
            if os.path.exists(self.config_path):
                import shutil
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{self.config_path}.backup.{timestamp}"
                shutil.copy2(self.config_path, backup_path)
                print(f"📁 Backup created: {backup_path}")
            
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            print(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        Args:
            key: Configuration key (supports dot notation, e.g., 'models.face_detector.threshold')
            default: Default value if key not found
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """
        Deep update dictionary
        Args:
            base_dict: Base dictionary to update
            update_dict: Dictionary with updates
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def validate_config(self) -> bool:
        """
        Validate configuration values
        Returns:
            True if configuration is valid
        """
        errors = []
        
        # Validate numeric ranges
        if not 0 <= self.get('recognition_threshold', 0) <= 1:
            errors.append("recognition_threshold must be between 0 and 1")
        
        if not 0 <= self.get('liveness_threshold', 0) <= 1:
            errors.append("liveness_threshold must be between 0 and 1")
        
        if self.get('min_face_size', 0) <= 0:
            errors.append("min_face_size must be positive")
        
        if self.get('max_database_items', 0) <= 0:
            errors.append("max_database_items must be positive")
        
        # Validate file paths
        database_path = self.get('database_path')
        if database_path:
            database_dir = os.path.dirname(database_path)
            if database_dir and not os.path.exists(database_dir):
                errors.append(f"Database directory does not exist: {database_dir}")
        
        images_dir = self.get('images_directory')
        if images_dir and not os.path.exists(images_dir):
            print(f"Warning: Images directory does not exist: {images_dir}")
        
        # Print errors
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    def print_config(self):
        """Print current configuration"""
        print("=== Current Configuration ===")
        self._print_dict(self.config, indent=0)
    
    def _print_dict(self, d: Dict, indent: int):
        """Recursively print dictionary"""
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")

def load_config_file(config_path: str) -> Optional[Dict]:
    """
    Load configuration from file
    Args:
        config_path: Path to configuration file
    Returns:
        Configuration dictionary or None if failed
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return None