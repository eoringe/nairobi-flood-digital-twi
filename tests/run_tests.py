"""
tests.run_tests
===============
Simple test runner to execute all unit tests.
"""

import sys
from loguru import logger

from tests.test_pipeline import (
    test_memory_guard_initialization,
    test_processed_arrays_exist,
    test_processed_arrays_shapes,
)
from tests.test_model import (
    test_spatial_autoencoder_shape,
    test_conv_lstm_surrogate_shape,
    test_sub_second_inference_latency,
)
from tests.test_dashboard import (
    test_pydeck_3d_creation,
    test_dashboard_layout_construction,
)

def run_all():
    logger.info("Executing Complete Project Unit Test Suite...")
    
    test_memory_guard_initialization()
    logger.info("✓ test_memory_guard_initialization PASSED")
    
    test_processed_arrays_exist()
    logger.info("✓ test_processed_arrays_exist PASSED")
    
    test_processed_arrays_shapes()
    logger.info("✓ test_processed_arrays_shapes PASSED")
    
    test_spatial_autoencoder_shape()
    logger.info("✓ test_spatial_autoencoder_shape PASSED")
    
    test_conv_lstm_surrogate_shape()
    logger.info("✓ test_conv_lstm_surrogate_shape PASSED")
    
    test_sub_second_inference_latency()
    logger.info("✓ test_sub_second_inference_latency PASSED")
    
    test_pydeck_3d_creation()
    logger.info("✓ test_pydeck_3d_creation PASSED")
    
    test_dashboard_layout_construction()
    logger.info("✓ test_dashboard_layout_construction PASSED")

    logger.info("==================================================")
    logger.info("🎉 ALL 8 UNIT TESTS PASSED SUCCESSFULLY!")
    logger.info("==================================================")

if __name__ == "__main__":
    run_all()
