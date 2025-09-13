"""
Configuration management for audiobook features
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional


class AudiobookConfig:
    """Configuration manager for audiobook features."""
    
    def __init__(self, config_path: str = "audiobook_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create defaults."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
        
        # Default configuration
        default_config = {
            "voice_library_path": "voice_library",
            "default_voice": None,
            "project_path": "audiobook_projects",
            "audio_format": "wav",
            "sample_rate": 24000,
            "batch_size": 1,
            "auto_save_voices": True,
            "backup_voices": True
        }
        
        self.save_config(default_config)
        return default_config
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Save configuration to file."""
        if config is not None:
            self.config.update(config)
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> bool:
        """Set configuration value and save."""
        self.config[key] = value
        return self.save_config()
    
    @property
    def voice_library_path(self) -> str:
        """Get voice library path."""
        return self.get("voice_library_path", "voice_library")
    
    @voice_library_path.setter
    def voice_library_path(self, path: str) -> None:
        """Set voice library path."""
        self.set("voice_library_path", path)
    
    @property
    def project_path(self) -> str:
        """Get project path."""
        return self.get("project_path", "audiobook_projects")
    
    @project_path.setter
    def project_path(self, path: str) -> None:
        """Set project path."""
        self.set("project_path", path)


# Global configuration instance
config = AudiobookConfig()
