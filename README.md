Vehicle License Plate Recognition System – Phase 2

Phase 2 Goal: Build the complete license plate reading system using advanced deep learning and OCR techniques.

 1. Features Implemented

Automatic license plate detection

CNN-based character recognition

Tesseract OCR integration

Plate classification (Ghana standard, diplomatic, police, DV…)

GUI for user input and results

Improved accuracy using preprocessing filters

 2. Technologies Used

Python

OpenCV

TensorFlow / Keras

Tesseract OCR

Flask

NumPy

Git & GitHub

 3. Folder Structure
Vehicle-LPR-Phase-2/
│── src/
│── static/
│── dataset/
│── templates/
│── README.md

 4. Setup

Install packages:

pip install -r requirements.txt


Set Tesseract path in code:

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

 5. Running the Application
python src/app.py

 6. Outputs

Detected license plates

Extracted characters

Final GUI application

 7. Future Improvements

Deploy on web

Deploy mobile app version

Improve model training with synthetic plates