#!/usr/bin/env python3
"""
Configuration file for the Robotics Data Processing Pipeline
"""

import os
from pathlib import Path

# Application Configuration
class Config:
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'robotics_simulation_key_2024'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # File Upload Configuration
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size
    UPLOAD_FOLDER = 'uploads'
    PROCESSED_FOLDER = 'processed'
    ALLOWED_EXTENSIONS = {'csv', 'tsv', 'txt'}  # Removed zip since we now support directories
    
    # Processing Configuration
    DEFAULT_TRAIN_RATIO = 0.8
    MAX_PLOTS_PER_REQUEST = 5
    PLOT_DPI = 150
    PLOT_FIGSIZE = (20, 15)
    
    # Sensor Plot Configuration
    SENSOR_PLOT_ROWS = 6
    SENSOR_PLOT_COLS = 4
    MAX_SENSORS_TO_PLOT = 24
    
    # Outlier Removal Configuration
    DEFAULT_OUTLIER_THRESHOLD = 500.0
    DEFAULT_OUTLIER_IGNORE_COLS = ['Timestamp', 'time', 'time_ms']
    
    # Time Matching Configuration
    DEFAULT_BASELINE_DURATION_MS = 5000
    
    # Simulation Configuration
    SIMULATION_FPS = 120
    SIMULATION_CAMERA_RES = (1280, 960)
    
    # Server Configuration
    DEFAULT_PORT = int(os.environ.get('FLASK_PORT', 5001))
    DEFAULT_HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    
    # Paths
    BASE_DIR = Path(__file__).parent
    TEMPLATES_DIR = BASE_DIR / 'templates'
    STATIC_DIR = BASE_DIR / 'static'
    
    # Ensure directories exist
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.UPLOAD_FOLDER,
            cls.PROCESSED_FOLDER,
            cls.TEMPLATES_DIR,
            cls.STATIC_DIR
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Validation
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        if cls.MAX_CONTENT_LENGTH <= 0:
            raise ValueError("MAX_CONTENT_LENGTH must be positive")
        
        if not (0 < cls.DEFAULT_TRAIN_RATIO < 1):
            raise ValueError("DEFAULT_TRAIN_RATIO must be between 0 and 1")
        
        if cls.PLOT_DPI <= 0:
            raise ValueError("PLOT_DPI must be positive")
        
        return True

# Development Configuration
class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False

# Production Configuration
class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(24)

# Testing Configuration
class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    WTF_CSRF_ENABLED = False

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(config_name=None):
    """Get configuration class by name"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    return config.get(config_name, config['default'])
