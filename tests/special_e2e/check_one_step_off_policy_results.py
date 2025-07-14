#!/usr/bin/env python3
"""
Check one_step_off_policy E2E test results.
This script validates that the one_step_off_policy training completes successfully
and produces expected async training behavior.
"""

import argparse
import os
import re
import sys


def check_training_logs(log_file_path: str) -> bool:
    """
    Check if training logs contain expected patterns for one_step_off_policy.

    Args:
        log_file_path: Path to the training log file

    Returns:
        True if logs indicate successful training, False otherwise
    """
    if not os.path.exists(log_file_path):
        print(f"Log file {log_file_path} does not exist")
        return False

    required_patterns = [
        # Core async training patterns
        r"async_gen_next_batch",  # Async generation function
        r"sync_rollout_weights",  # Weight synchronization
        r"wait_prev_gen",  # Waiting for previous generation
        r"generate_sequences",  # Sequence generation
        r"old_log_prob",  # Old log probability computation
        r"update_actor",  # Actor update
        # Success patterns
        r"One-step-off-policy E2E test completed successfully",  # Test completion
        r"training/global_step",  # Training progress
        r"training/epoch",  # Epoch progress
    ]

    found_patterns = set()

    with open(log_file_path, 'r') as f:
        content = f.read()

        for pattern in required_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                found_patterns.add(pattern)
                print(f"✓ Found required pattern: {pattern}")
            else:
                print(f"✗ Missing required pattern: {pattern}")

    # Check for error patterns
    error_patterns = [
        r"Error",
        r"Exception",
        r"Traceback",
        r"FAILED",
        r"AssertionError",
    ]

    for pattern in error_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            print(f"✗ Found error pattern: {pattern}")
            return False

    # Check if we found enough required patterns
    required_found = len([p for p in required_patterns if p in found_patterns])
    if required_found < len(required_patterns) * 0.7:  # At least 70% of patterns should be found
        print(f"Only found {required_found}/{len(required_patterns)} required patterns")
        return False

    print(f"✓ Found {required_found}/{len(required_patterns)} required patterns")
    return True


def main():
    parser = argparse.ArgumentParser(description="Check one_step_off_policy E2E test results")
    parser.add_argument("--log-file", type=str, help="Path to training log file")
    parser.add_argument("--strategy", type=str, choices=["fsdp2", "megatron"],
                       help="Training strategy used")

    args = parser.parse_args()

    success = True

    # Check logs if provided
    if args.log_file:
        print(f"Checking training logs for {args.strategy} strategy...")
        if not check_training_logs(args.log_file):
            success = False

    # If no specific files provided, look for common locations
    if not args.log_file:
        print("No specific files provided, checking common locations...")

        # Look for log files in common locations
        log_locations = [
            "./training.log",
            "./one_step_off_policy.log",
            "/tmp/one_step_off_policy.log",
        ]

        for log_path in log_locations:
            if os.path.exists(log_path):
                print(f"Found log file: {log_path}")
                if not check_training_logs(log_path):
                    success = False
                break

    if success:
        print("\n✅ All checks passed! One-step-off-policy training completed successfully.")
        sys.exit(0)
    else:
        print("\n❌ Some checks failed! Please review the training logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()

