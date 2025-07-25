import os
import uuid
from flask import Flask, render_template, request, send_from_directory, jsonify
from dehazer.haze_remover import HazeRemover
import cv2
import numpy as np

# Initialize the Flask application and the HazeRemover
app = Flask(__name__)
remover = HazeRemover()

# --- Configuration ---
# Define folder names
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

# Set the configuration keys in Flask.
app.config['UPLOADS_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUTS_FOLDER'] = OUTPUT_FOLDER

# Ensure the upload and output directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_image():
    """
    Receives an uploaded image, resizes it if necessary, performs basic 
    dehazing, saves it, and returns the filename.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            filestr = file.read()
            npimg = np.frombuffer(filestr, np.uint8)
            hazy_image_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            # --- NEW: Image Resizing Logic to Save Memory ---
            max_dimension = 1024
            height, width = hazy_image_bgr.shape[:2]

            if height > max_dimension or width > max_dimension:
                # Determine the scaling factor
                if height > width:
                    scale_factor = max_dimension / height
                else:
                    scale_factor = max_dimension / width
                
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                # Resize the image
                hazy_image_bgr = cv2.resize(hazy_image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
            # --- End of New Logic ---

            # Perform dehazing WITHOUT post-enhancement
            dehazed_image_bgr = remover.process(hazy_image_bgr, enhance=False)

            # Save the dehazed image
            unique_id = str(uuid.uuid4())
            dehazed_filename = f"{unique_id}_dehazed.png"
            dehazed_filepath = os.path.join(app.config['OUTPUTS_FOLDER'], dehazed_filename)
            cv2.imwrite(dehazed_filepath, dehazed_image_bgr)

            return jsonify({'filename': dehazed_filename})
        except Exception as e:
            return jsonify({'error': f'Server error: {e}'}), 500

    return jsonify({'error': 'Invalid file'}), 400

@app.route('/enhance', methods=['POST'])
def enhance_image():
    """
    Receives the filename of a dehazed image, applies CLAHE enhancement,
    saves it, and returns the new filename.
    """
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({'error': 'Missing filename'}), 400
    
    filename = data['filename']
    filepath = os.path.join(app.config['OUTPUTS_FOLDER'], filename)

    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
        
    try:
        image_to_enhance = cv2.imread(filepath)
        
        # Apply only the enhancement function
        enhanced_image_bgr = remover._enhance_image(image_to_enhance)

        # Save the new enhanced image
        unique_id = str(uuid.uuid4())
        enhanced_filename = f"{unique_id}_enhanced.png"
        enhanced_filepath = os.path.join(app.config['OUTPUTS_FOLDER'], enhanced_filename)
        cv2.imwrite(enhanced_filepath, enhanced_image_bgr)
        
        #remove the old non-enhanced file
        os.remove(filepath)

        return jsonify({'filename': enhanced_filename})
    except Exception as e:
        return jsonify({'error': f'Server error: {e}'}), 500


@app.route('/display/<folder>/<filename>')
def display_image(folder, filename):
    """Serves images from the 'uploads' or 'outputs' directory."""
    config_key = folder.upper() + '_FOLDER'
    if config_key in app.config:
        return send_from_directory(app.config[config_key], filename)
    return "File not found", 404


if __name__ == '__main__':
    app.run(debug=True)