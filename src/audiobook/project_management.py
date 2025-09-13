"""
Project management for audiobook creation
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


def ensure_project_directory(project_path: str) -> None:
    """Ensure the project directory exists."""
    os.makedirs(project_path, exist_ok=True)


def create_audiobook_project(project_path: str, project_name: str, 
                           description: str = "", metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Create a new audiobook project.
    
    Args:
        project_path: Base path for projects
        project_name: Name of the project
        description: Project description
        metadata: Additional project metadata
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_project_directory(project_path)
        project_dir = os.path.join(project_path, project_name)
        os.makedirs(project_dir, exist_ok=True)
        
        # Create project structure
        os.makedirs(os.path.join(project_dir, "chapters"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "characters"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "output"), exist_ok=True)
        
        # Create project metadata
        project_info = {
            "name": project_name,
            "description": description,
            "created": datetime.now().isoformat(),
            "characters": {},
            "chapters": [],
            "settings": metadata or {}
        }
        
        info_file = os.path.join(project_dir, "project.json")
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(project_info, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error creating project: {e}")
        return False


def get_projects(project_path: str) -> List[Dict[str, Any]]:
    """Get all audiobook projects.
    
    Args:
        project_path: Base path for projects
        
    Returns:
        List of project information dictionaries
    """
    projects = []
    try:
        ensure_project_directory(project_path)
        for item in os.listdir(project_path):
            project_dir = os.path.join(project_path, item)
            if os.path.isdir(project_dir):
                info_file = os.path.join(project_dir, "project.json")
                if os.path.exists(info_file):
                    try:
                        with open(info_file, 'r', encoding='utf-8') as f:
                            project = json.load(f)
                            project['path'] = project_dir
                            projects.append(project)
                    except Exception as e:
                        print(f"Warning: Could not load project {item}: {e}")
    except Exception as e:
        print(f"Warning: Could not read projects: {e}")
    
    return projects


def get_project_characters(project_path: str, project_name: str) -> Dict[str, Any]:
    """Get characters for a specific project.
    
    Args:
        project_path: Base path for projects
        project_name: Name of the project
        
    Returns:
        Dictionary of character voice assignments
    """
    project_dir = os.path.join(project_path, project_name)
    info_file = os.path.join(project_dir, "project.json")
    
    if not os.path.exists(info_file):
        return {}
    
    try:
        with open(info_file, 'r', encoding='utf-8') as f:
            project = json.load(f)
            return project.get("characters", {})
    except Exception as e:
        print(f"Error loading project characters: {e}")
        return {}


def assign_voice_to_character(project_path: str, project_name: str, 
                            character_name: str, voice_name: str) -> bool:
    """Assign a voice to a character in a project.
    
    Args:
        project_path: Base path for projects
        project_name: Name of the project
        character_name: Name of the character
        voice_name: Name of the voice profile
        
    Returns:
        True if successful, False otherwise
    """
    try:
        project_dir = os.path.join(project_path, project_name)
        info_file = os.path.join(project_dir, "project.json")
        
        if not os.path.exists(info_file):
            return False
        
        with open(info_file, 'r', encoding='utf-8') as f:
            project = json.load(f)
        
        if "characters" not in project:
            project["characters"] = {}
        
        project["characters"][character_name] = {
            "voice": voice_name,
            "assigned": datetime.now().isoformat()
        }
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(project, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error assigning voice to character: {e}")
        return False
