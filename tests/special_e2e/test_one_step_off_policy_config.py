#!/usr/bin/env python3
"""
Quick configuration validation for one_step_off_policy tests.
This script validates that all required config files and imports are available.
"""

import importlib.util
import os
import sys


def test_config_files():
    """Test that required config files exist."""
    config_files = [
        "recipe/one_step_off_policy/config/async_ppo_trainer.yaml",
        "recipe/one_step_off_policy/config/async_ppo_megatron_trainer.yaml",
    ]

    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✓ Found config file: {config_file}")
        else:
            print(f"✗ Missing config file: {config_file}")
            return False

    return True

def test_imports():
    """Test that required modules can be imported."""
    modules_to_test = [
        "recipe.one_step_off_policy.async_main_ppo",
        "recipe.one_step_off_policy.async_ray_trainer",
    ]

    for module_name in modules_to_test:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                print(f"✓ Can import: {module_name}")
            else:
                print(f"✗ Cannot find: {module_name}")
                return False
        except Exception as e:
            print(f"✗ Error importing {module_name}: {e}")
            return False

    return True

def test_script_files():
    """Test that required script files exist."""
    script_files = [
        "tests/special_e2e/run_one_step_off_policy.sh",
        "tests/special_e2e/check_one_step_off_policy_results.py",
    ]

    for script_file in script_files:
        if os.path.exists(script_file):
            print(f"✓ Found script file: {script_file}")
        else:
            print(f"✗ Missing script file: {script_file}")
            return False

    return True

def main():
    """Run all configuration tests."""
    print("Running one_step_off_policy configuration tests...")

    success = True

    print("\n1. Testing config files...")
    if not test_config_files():
        success = False

    print("\n2. Testing imports...")
    if not test_imports():
        success = False

    print("\n3. Testing script files...")
    if not test_script_files():
        success = False

    if success:
        print("\n✅ All configuration tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some configuration tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

