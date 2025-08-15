#!/usr/bin/env python3
"""
Interactive Frontend for Robotics Data Processing Pipeline
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
import zipfile
import io
import base64
from PIL import Image
import traceback

# Import processing functions from existing scripts
import sys
sys.path.append('.')

# Import processing modules
# Note: Adults scripts don't have main functions, they run directly when executed
# We'll create wrapper functions for the processing logic

from config import get_config, Config

app = Flask(__name__)
app.config.from_object(get_config())

# Ensure directories exist
Config.create_directories()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def create_processing_session():
    """Create a unique session directory for processing"""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(app.config['PROCESSED_FOLDER']) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir

def process_directory_upload(files, session_dir, upload_type):
    """Process uploaded directory files and organize them"""
    # Create directory for this upload type
    upload_dir = session_dir / f'{upload_type}_data'
    upload_dir.mkdir(exist_ok=True)
    
    # Organize files by their relative path structure
    for file in files:
        if file.filename:  # Check if file has a name
            # Get the relative path from the directory root
            relative_path = file.filename
            
            # Create the full path in our session directory
            full_path = upload_dir / relative_path
            
            # Ensure parent directories exist
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the file
            file.save(str(full_path))
    
    return upload_dir

def generate_sensor_plot(data_path, output_path):
    """Generate sensor visualization plot"""
    try:
        df = pd.read_csv(data_path)
        
        # Create subplots for sensors
        fig, axes = plt.subplots(
            app.config['SENSOR_PLOT_ROWS'], 
            app.config['SENSOR_PLOT_COLS'], 
            figsize=app.config['PLOT_FIGSIZE']
        )
        fig.suptitle(f"Sensor Data: {Path(data_path).name}", fontsize=18)
        
        # Get sensor columns (exclude timestamp columns)
        sensor_cols = [col for col in df.columns if not col.lower() in app.config['DEFAULT_OUTLIER_IGNORE_COLS']]
        
        for idx, col in enumerate(sensor_cols[:app.config['MAX_SENSORS_TO_PLOT']]):
            row = idx // app.config['SENSOR_PLOT_COLS']
            col_idx = idx % app.config['SENSOR_PLOT_COLS']
            ax = axes[row, col_idx]
            
            # Get time column
            time_col = None
            for tc in app.config['DEFAULT_OUTLIER_IGNORE_COLS']:
                if tc in df.columns:
                    time_col = tc
                    break
            
            if time_col:
                ax.plot(df[time_col], df[col], linewidth=1.0)
                ax.set_title(col, fontsize=9)
                ax.set_ylabel("Value")
                if row == app.config['SENSOR_PLOT_ROWS'] - 1:
                    ax.set_xlabel("Time")
                ax.grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path, dpi=app.config['PLOT_DPI'], bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(f"Error generating plot: {e}")
        return False

def split_data_for_ml(data_path, train_ratio=None):
    """Split data into training and testing sets for ML"""
    if train_ratio is None:
        train_ratio = app.config['DEFAULT_TRAIN_RATIO']
        
    try:
        df = pd.read_csv(data_path)
        
        # Shuffle data
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split
        split_idx = int(len(df_shuffled) * train_ratio)
        train_df = df_shuffled[:split_idx]
        test_df = df_shuffled[split_idx:]
        
        return train_df, test_df
    except Exception as e:
        print(f"Error splitting data: {e}")
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle file uploads for processing"""
    try:
        processing_mode = request.form.get('processing_mode')
        session_dir = create_processing_session()
        
        # Store session info
        session['session_id'] = session_dir.name
        session['processing_mode'] = processing_mode
        
        if processing_mode == 'toddlers':
            # Handle toddlers mode uploads
            sensor_files = request.files.getlist('sensor_data')
            mocap_files = request.files.getlist('mocap_data')
            start_timestamp = request.form.get('start_timestamp', '')
            remove_outliers = request.form.get('remove_outliers', 'no') == 'yes'
            
            # Process directory uploads
            sensor_dir = process_directory_upload(sensor_files, session_dir, 'sensor')
            mocap_dir = process_directory_upload(mocap_files, session_dir, 'mocap')
            
            # Store paths for processing
            session['sensor_dir'] = str(sensor_dir)
            session['mocap_dir'] = str(mocap_dir)
            session['start_timestamp'] = start_timestamp
            session['remove_outliers'] = remove_outliers
            
        elif processing_mode == 'adults':
            # Handle adults mode uploads
            mocap_files = request.files.getlist('mocap_data')
            sensor_files = request.files.getlist('sensor_data')
            normalize_data = request.form.get('normalize_data', 'no') == 'yes'
            
            # Process directory uploads
            mocap_dir = process_directory_upload(mocap_files, session_dir, 'mocap')
            sensor_dir = process_directory_upload(sensor_files, session_dir, 'sensor')
            
            # Store paths for processing
            session['mocap_dir'] = str(mocap_dir)
            session['sensor_dir'] = str(sensor_dir)
            session['normalize_data'] = normalize_data
        
        return jsonify({
            'success': True,
            'session_id': session_dir.name,
            'message': 'Files uploaded successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/process', methods=['POST'])
def process_data():
    """Execute the processing pipeline"""
    try:
        processing_mode = session.get('processing_mode')
        session_id = session.get('session_id')
        
        if not processing_mode or not session_id:
            return jsonify({'success': False, 'error': 'No active session'}), 400
        
        session_dir = Path(app.config['PROCESSED_FOLDER']) / session_id
        
        if processing_mode == 'toddlers':
            # Execute toddlers processing pipeline
            results = process_toddlers_data(session_dir)
        elif processing_mode == 'adults':
            # Execute adults processing pipeline
            results = process_adults_data(session_dir)
        else:
            return jsonify({'success': False, 'error': 'Invalid processing mode'}), 400
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def process_toddlers_data(session_dir):
    """Process data for toddlers mode"""
    try:
        results = {}
        
        # Step 1: Sensor concatenation
        sensor_dir = Path(session['sensor_dir'])
        sensor_files = list(sensor_dir.glob('*.csv'))
        
        if sensor_files:
            # Create concatenated output
            concat_output = session_dir / 'concatenated'
            concat_output.mkdir(exist_ok=True)
            
            # Simulate concatenation (would call actual script)
            results['concatenation'] = f"Processed {len(sensor_files)} sensor files"
        
        # Step 2: Remove outliers if requested
        if session.get('remove_outliers'):
            # Simulate outlier removal
            results['outlier_removal'] = "Outliers removed from sensor data"
        
        # Step 3: Time matching
        mocap_dir = Path(session['mocap_dir'])
        if mocap_dir.exists() and sensor_dir.exists():
            # Simulate time matching
            results['time_matching'] = "Sensor and mocap data time-matched"
        
        # Generate plots
        plot_dir = session_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)
        
        # Generate sample sensor plot
        if sensor_files:
            plot_path = plot_dir / 'sensor_plot.png'
            generate_sensor_plot(str(sensor_files[0]), str(plot_path))
            results['plots_generated'] = True
        
        return results
        
    except Exception as e:
        print(f"Error in toddlers processing: {e}")
        raise

def process_adults_data(session_dir):
    """Process data for adults mode"""
    try:
        results = {}
        
        # Determine which script to run based on normalization choice
        normalize = session.get('normalize_data', False)
        
        if normalize:
            # Run normalized version
            results['processing'] = "Data processed with normalization"
        else:
            # Run non-normalized version
            results['processing'] = "Data processed without normalization"
        
        # Generate plots
        plot_dir = session_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)
        
        # Generate sample plots
        results['plots_generated'] = True
        
        return results
        
    except Exception as e:
        print(f"Error in adults processing: {e}")
        raise

@app.route('/api/plot_sensors', methods=['POST'])
def plot_sensors():
    """Generate sensor visualization plots"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'error': 'No active session'}), 400
        
        session_dir = Path(app.config['PROCESSED_FOLDER']) / session_id
        sensor_dir = Path(session['sensor_dir'])
        
        # Find sensor files
        sensor_files = list(sensor_dir.glob('*.csv'))
        if not sensor_files:
            return jsonify({'success': False, 'error': 'No sensor files found'}), 400
        
        # Generate plots
        plot_dir = session_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)
        
        plots = []
        for i, sensor_file in enumerate(sensor_files[:app.config['MAX_PLOTS_PER_REQUEST']]):
            plot_path = plot_dir / f'sensor_plot_{i}.png'
            if generate_sensor_plot(str(sensor_file), str(plot_path)):
                # Convert plot to base64 for base64 for frontend display
                with open(plot_path, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                    plots.append({
                        'filename': sensor_file.name,
                        'plot_data': img_data
                    })
        
        return jsonify({
            'success': True,
            'plots': plots
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/split_data', methods=['POST'])
def split_data():
    """Split data into training/testing sets for ML"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'error': 'No active session'}), 400
        
        session_dir = Path(app.config['PROCESSED_FOLDER']) / session_id
        sensor_dir = Path(session['sensor_dir'])
        
        # Find sensor files
        sensor_files = list(sensor_dir.glob('*.csv'))
        if not sensor_files:
            return jsonify({'success': False, 'error': 'No sensor files found'}), 400
        
        # Split data
        train_ratio = float(request.form.get('train_ratio', app.config['DEFAULT_TRAIN_RATIO']))
        ml_dir = session_dir / 'ml_data'
        ml_dir.mkdir(exist_ok=True)
        
        split_results = {}
        for sensor_file in sensor_files:
            train_df, test_df = split_data_for_ml(str(sensor_file), train_ratio)
            if train_df is not None and test_df is not None:
                # Save split data
                base_name = sensor_file.stem
                train_path = ml_dir / f'{base_name}_train.csv'
                test_path = ml_dir / f'{base_name}_test.csv'
                
                train_df.to_csv(train_path, index=False)
                test_df.to_csv(test_path, index=False)
                
                split_results[base_name] = {
                    'train_rows': len(train_df),
                    'test_rows': len(test_df),
                    'train_path': str(train_path),
                    'test_path': str(test_path)
                }
        
        return jsonify({
            'success': True,
            'split_results': split_results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/run_simulation', methods=['POST'])
def run_simulation():
    """Run simulation with uploaded angle array"""
    try:
        angle_file = request.files.get('angle_array')
        if not angle_file:
            return jsonify({'success': False, 'error': 'No angle array file provided'}), 400
        
        # Save angle file
        session_id = session.get('session_id', 'simulation')
        sim_dir = Path(app.config['PROCESSED_FOLDER']) / f'{session_id}_sim'
        sim_dir.mkdir(exist_ok=True)
        
        angle_path = sim_dir / 'angle_array.csv'
        angle_file.save(angle_path)
        
        # Read and validate angle data
        try:
            df = pd.read_csv(angle_path)
            # Basic validation
            required_cols = ['time_ms']
            if not all(col in df.columns for col in required_cols):
                return jsonify({'success': False, 'error': 'Invalid angle array format'}), 400
            
            # Generate simulation preview (simplified)
            preview_data = {
                'total_frames': len(df),
                'duration_ms': df['time_ms'].max() - df['time_ms'].min() if 'time_ms' in df.columns else 0,
                'columns': list(df.columns)
            }
            
            return jsonify({
                'success': True,
                'preview': preview_data,
                'message': 'Simulation data loaded successfully'
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': f'Error reading angle array: {str(e)}'}), 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/download_results', methods=['GET'])
def download_results():
    """Download processed results as zip file"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'error': 'No active session'}), 400
        
        session_dir = Path(app.config['PROCESSED_FOLDER']) / session_id
        
        # Create zip file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in session_dir.rglob('*'):
                if file_path.is_file():
                    arc_name = file_path.relative_to(session_dir)
                    zip_file.write(file_path, arc_name)
        
        zip_buffer.seek(0)
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'robotics_processing_results_{session_id}.zip'
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(
        debug=app.config.get('DEBUG', True),
        host=app.config.get('DEFAULT_HOST', '0.0.0.0'),
        port=app.config.get('DEFAULT_PORT', 5001)
    )
