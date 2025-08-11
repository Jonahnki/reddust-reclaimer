#!/usr/bin/env python3
"""Validate SBML models and data integrity for RedDust Reclaimer."""

import sys
import os
from pathlib import Path
import xml.etree.ElementTree as ET

def validate_xml_files():
    """Validate XML files in the models directory."""
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Models directory not found")
        return False
    
    success = True
    xml_files = list(models_dir.glob("*.xml"))
    
    if not xml_files:
        print("⚠️  No XML model files found in models/ directory")
        return True  # Not an error if no models exist yet
    
    for model_file in xml_files:
        try:
            # Basic XML validation
            ET.parse(model_file)
            print(f"✅ {model_file.name}: Valid XML structure")
            
            # Check if it's an SBML file
            with open(model_file, 'r') as f:
                content = f.read()
                if 'sbml' in content.lower():
                    print(f"✅ {model_file.name}: SBML format detected")
                else:
                    print(f"ℹ️  {model_file.name}: Generic XML file")
                    
        except ET.ParseError as e:
            print(f"❌ {model_file.name}: XML parsing error - {e}")
            success = False
        except Exception as e:
            print(f"❌ {model_file.name}: Validation error - {e}")
            success = False
    
    return success

def validate_data_integrity():
    """Validate data files and directory structure."""
    required_dirs = ["scripts", "models", "tests"]
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"❌ Missing required directories: {missing_dirs}")
        return False
    
    print("✅ All required directories present")
    return True

def main():
    """Main validation function."""
    print("🔬 Starting RedDust Reclaimer model validation...")
    
    xml_valid = validate_xml_files()
    structure_valid = validate_data_integrity()
    
    if xml_valid and structure_valid:
        print("🎉 All validations passed!")
        return 0
    else:
        print("❌ Some validations failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
