# Image Dehazer Web Application

A modern web application built with Python and Flask that removes haze from images, featuring a dynamic, soft sky-blue neumorphic interface. The core image processing is based on the Dark Channel Prior algorithm, which effectively restores clear images from hazy conditions.

## 📸 Final Application


![Application Demo](https://github.com/user-attachments/assets/065d1d0d-3048-417c-97b8-00084974b20f)


## 🚀 Features

- **Single-Click Dehazing:** Remove haze from any image with a simple button click.
- **Contrast Enhancement:** Further improve the clarity and contrast of the dehazed image with a dedicated "Enhance" button.
- **Modern Neumorphic UI:** A beautiful and responsive user interface with a soft sky-blue color palette and neumorphic design principles.
- **Dynamic Resizing Layout:** The interface starts with a large input panel that dynamically resizes to be equal with the output panel after processing for an intuitive workflow.
- **Interactive Image Upload:** Click to select or drag-and-drop an image directly into the upload area.
- **Download Functionality:** Easily download the final processed image to your local machine.

## 🛠️ Technology Stack

- **Backend:** Python, Flask, Gunicorn (for production).
- **Image Processing:** OpenCV, NumPy.
- **Frontend:** HTML5, CSS3, JavaScript (using the Fetch API for asynchronous requests).

## ⚙️ Setup and Local Installation

To run this project on your local machine, follow these steps:

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd <repository-folder>
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**
    Install all the required packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**
    Start the Flask development server.
    ```bash
    python app.py
    ```

5.  **Access the Application**
    Open your web browser and navigate to `http://127.0.0.1:5000`.