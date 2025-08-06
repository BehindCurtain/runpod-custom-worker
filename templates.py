"""Template configurations for RunPod Custom Worker with shared model storage."""

import json
import os
from pathlib import Path

# Template data will be loaded from JSON file
_templates_data = None
_templates_file = "templates.json"

def _load_templates():
    """Load templates from JSON file."""
    global _templates_data
    
    if _templates_data is None:
        # Try to find templates.json in current directory or script directory
        current_dir = Path.cwd()
        script_dir = Path(__file__).parent
        
        templates_path = None
        for search_dir in [current_dir, script_dir]:
            potential_path = search_dir / _templates_file
            if potential_path.exists():
                templates_path = potential_path
                break
        
        if templates_path is None:
            raise FileNotFoundError(f"Templates file '{_templates_file}' not found in {current_dir} or {script_dir}")
        
        try:
            with open(templates_path, 'r', encoding='utf-8') as f:
                _templates_data = json.load(f)
            print(f"✓ Templates loaded from {templates_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in templates file '{templates_path}': {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load templates from '{templates_path}': {e}")
    
    return _templates_data

def get_templates():
    """Get all templates dictionary."""
    data = _load_templates()
    return data["templates"]

def get_default_template():
    """Get default template name."""
    data = _load_templates()
    return data["default_template"]

# Backward compatibility - expose as module-level variables
def _get_templates_compat():
    """Get templates dictionary (backward compatibility)."""
    return get_templates()

def _get_default_template_compat():
    """Get default template name (backward compatibility)."""
    return get_default_template()

# Module-level variables for backward compatibility
TEMPLATES = _get_templates_compat()
DEFAULT_TEMPLATE = _get_default_template_compat()

def get_template(template_name):
    """Get template configuration by name."""
    templates = get_templates()
    if template_name not in templates:
        raise ValueError(f"Template '{template_name}' not found. Available templates: {list(templates.keys())}")
    return templates[template_name]

def get_all_unique_checkpoints():
    """Get all unique checkpoints across all templates."""
    templates = get_templates()
    unique_checkpoints = {}
    
    for template_name, template in templates.items():
        checkpoint = template["checkpoint"]
        if checkpoint["filename"] not in unique_checkpoints:
            unique_checkpoints[checkpoint["filename"]] = checkpoint
    
    return unique_checkpoints

def get_all_unique_loras():
    """Get all unique LoRAs across all templates."""
    templates = get_templates()
    unique_loras = {}
    
    for template_name, template in templates.items():
        for lora in template["loras"]:
            if lora["filename"] not in unique_loras:
                unique_loras[lora["filename"]] = lora
    
    return unique_loras

def list_templates():
    """List all available templates with descriptions."""
    templates = get_templates()
    return {
        name: {
            "name": template["name"],
            "description": template["description"]
        }
        for name, template in templates.items()
    }

def reload_templates():
    """Reload templates from JSON file (useful for development)."""
    global _templates_data
    _templates_data = None
    return _load_templates()

# Ensure templates are loaded when module is imported
try:
    _load_templates()
except Exception as e:
    print(f"⚠ Warning: Could not load templates on import: {e}")
    print("Templates will be loaded on first access.")
