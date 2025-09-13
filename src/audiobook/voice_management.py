"""
Voice management utilities for audiobook generation.

Handles voice profile CRUD operations, voice library management, and voice selection.
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any


def ensure_voice_library_exists(voice_library_path: str) -> None:
    """Ensure the voice library directory exists.
    
    Args:
        voice_library_path: Path to voice library directory
    """
    os.makedirs(voice_library_path, exist_ok=True)


def get_voice_profiles(voice_library_path: str) -> List[Dict[str, Any]]:
    """Get all voice profiles from the voice library.
    
    Args:
        voice_library_path: Path to voice library directory
        
    Returns:
        List of voice profile dictionaries
    """
    ensure_voice_library_exists(voice_library_path)
    profiles = []
    
    try:
        for item in os.listdir(voice_library_path):
            profile_dir = os.path.join(voice_library_path, item)
            if os.path.isdir(profile_dir):
                config_file = os.path.join(profile_dir, "config.json")
                if os.path.exists(config_file):
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            profile = json.load(f)
                            profile['voice_name'] = item
                            profiles.append(profile)
                    except Exception as e:
                        print(f"Warning: Could not load profile {item}: {e}")
    except Exception as e:
        print(f"Warning: Could not read voice library: {e}")
    
    return profiles


def get_voice_choices(voice_library_path: str) -> List[str]:
    """Get list of voice names for dropdown choices.
    
    Args:
        voice_library_path: Path to voice library directory
        
    Returns:
        List of voice names
    """
    profiles = get_voice_profiles(voice_library_path)
    return [p.get('display_name', p['voice_name']) for p in profiles]


def get_voice_info_html(voice_library_path: str) -> str:
    """Generate HTML info for voice library display.
    
    Args:
        voice_library_path: Path to voice library directory
        
    Returns:
        HTML string with voice information
    """
    profiles = get_voice_profiles(voice_library_path)
    if not profiles:
        return "<p>No voices found in library. Create your first voice profile!</p>"
    
    html_parts = []
    for profile in profiles:
        name = profile.get('display_name', profile['voice_name'])
        description = profile.get('description', 'No description provided')
        language = profile.get('language', 'Not specified')
        
        html_parts.append(f"""
        <div style="border: 1px solid #ddd; padding: 10px; margin: 5px 0; border-radius: 5px;">
            <strong>{name}</strong> ({language})<br>
            <small>{description}</small>
        </div>
        """)
    
    return "".join(html_parts)


def save_voice_profile(voice_library_path: str, voice_name: str, profile_data: Dict[str, Any]) -> bool:
    """Save a voice profile to the library.
    
    Args:
        voice_library_path: Path to voice library directory
        voice_name: Name for the voice profile (folder name)
        profile_data: Dictionary containing voice settings and metadata
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_voice_library_exists(voice_library_path)
        profile_dir = os.path.join(voice_library_path, voice_name)
        os.makedirs(profile_dir, exist_ok=True)
        
        config_file = os.path.join(profile_dir, "config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error saving voice profile: {e}")
        return False


def load_voice_profile(voice_library_path: str, voice_name: str) -> Optional[Dict[str, Any]]:
    """Load a specific voice profile.
    
    Args:
        voice_library_path: Path to voice library directory
        voice_name: Name of the voice profile to load
        
    Returns:
        Voice profile dictionary or None if not found
    """
    config_file = os.path.join(voice_library_path, voice_name, "config.json")
    if not os.path.exists(config_file):
        return None
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            profile = json.load(f)
            profile['voice_name'] = voice_name
            return profile
    except Exception as e:
        print(f"Error loading voice profile: {e}")
        return None


def delete_voice_profile(voice_library_path: str, voice_name: str) -> bool:
    """Delete a voice profile from the library.
    
    Args:
        voice_library_path: Path to voice library directory
        voice_name: Name of the voice profile to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        profile_dir = os.path.join(voice_library_path, voice_name)
        if os.path.exists(profile_dir):
            shutil.rmtree(profile_dir)
            return True
        return False
    except Exception as e:
        print(f"Error deleting voice profile: {e}")
        return False


def copy_reference_audio(voice_library_path: str, voice_name: str, source_path: str) -> Optional[str]:
    """Copy reference audio file to voice profile directory.
    
    Args:
        voice_library_path: Path to voice library directory
        voice_name: Name of the voice profile
        source_path: Path to the source audio file
        
    Returns:
        Path to the copied file or None if failed
    """
    try:
        profile_dir = os.path.join(voice_library_path, voice_name)
        os.makedirs(profile_dir, exist_ok=True)
        
        # Determine file extension
        ext = os.path.splitext(source_path)[1].lower()
        if not ext:
            ext = '.wav'
        
        dest_path = os.path.join(profile_dir, f"reference{ext}")
        shutil.copy2(source_path, dest_path)
        return dest_path
    except Exception as e:
        print(f"Error copying reference audio: {e}")
        return None
