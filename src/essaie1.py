import os
import cv2
import numpy as np
import pytesseract
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
MODEL_PATH = "ghana_plate_classifier.h5"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Uncomment if needed on Windows
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load model
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print(f"✅ Loaded CNN model from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Failed to load model: {e}")
else:
    print("⚠️ Model not found. CNN classification disabled.")

# ------------------------------------------------------------
# MANUAL REGION SELECTOR
# ------------------------------------------------------------
drawing = False
ix, iy = -1, -1
rx, ry, rw, rh = 0, 0, 0, 0


def manual_select_region(image):
    global drawing, ix, iy, rx, ry, rw, rh
    clone = image.copy()

    def draw_rect(event, x, y, flags, param):
        nonlocal clone
        global drawing, ix, iy, rx, ry, rw, rh

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            temp = clone.copy()
            cv2.rectangle(temp, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Select Plate", temp)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            rx, ry = ix, iy
            rw, rh = x - ix, y - iy
            cv2.rectangle(clone, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Select Plate", clone)

    cv2.imshow("Select Plate", clone)
    cv2.setMouseCallback("Select Plate", draw_rect)

    print("\n🟩 Draw a rectangle on the image. Press ENTER when done.")
    cv2.waitKey(0)
    cv2.destroyWindow("Select Plate")

    if rw != 0 and rh != 0:
        x1, y1 = min(ix, ix + rw), min(iy, iy + rh)
        x2, y2 = max(ix, ix + rw), max(iy, iy + rh)
        return image[y1:y2, x1:x2]
    else:
        print("❌ No manual region selected.")
        return None

# ------------------------------------------------------------
# 1) Plate Region Detection
# ------------------------------------------------------------


def detect_plate_region(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    edged = cv2.Canny(gray, 30, 200)

    cnts, _ = cv2.findContours(
        edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

    plate, plate_bbox = None, None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h) if h > 0 else 0
            if 2.0 <= aspect <= 8.0 and w > 60 and h > 15:
                plate = orig[y:y + h, x:x + w]
                plate_bbox = (x, y, w, h)
                break

    if plate is None:
        h, w = orig.shape[:2]
        plate = orig[int(0.3 * h):int(0.7 * h), int(0.2 * w):int(0.8 * w)]
        print("⚠️ Using fallback crop for plate region.")
    return plate, plate_bbox

# ------------------------------------------------------------
# 2) OCR Preprocessing amélioré
# ------------------------------------------------------------


def adjust_for_ocr_contrast(plate_img):
    """
    Retourne une image pré-traitée avec texte en noir et fond en blanc
    pour améliorer la reconnaissance OCR.
    """
    img = plate_img.copy()
    avg_color = np.mean(img, axis=(0, 1))  # BGR
    b, g, r = avg_color.astype(int)

    # Convertir en HSV pour mieux séparer teintes
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Décider si le texte est clair ou foncé → inverser si nécessaire
    if (r < 120 and b < 120) or (v.mean() < 128):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)  # texte clair → texte foncé
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE pour contraste local
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Blur léger
    gray = cv2.bilateralFilter(gray, 5, 75, 75)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)

    # Threshold adaptatif
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 35, 15
    )

    return th

# ------------------------------------------------------------
# 2) Improved OCR avec choix du mode
# ------------------------------------------------------------


def read_plate_text(plate_img, mode="A"):
    if plate_img is None:
        return ""

    # Preprocessing (same as before)
    th = adjust_for_ocr_contrast(plate_img)
    th_big = cv2.resize(th, (0, 0), fx=2.5, fy=2.5,
                        interpolation=cv2.INTER_CUBIC)

    height = th_big.shape[0]
    mid = height // 2
    top_half = th_big[0:mid, :]
    bottom_half = th_big[mid:height, :]

    config = "--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"

    mode = mode.upper()

    if mode == "1":
        text = pytesseract.image_to_string(th_big, config=config)
        return "".join(ch for ch in text if ch.isalnum() or ch == "-").upper().strip()

    if mode == "2":
        text_top = pytesseract.image_to_string(top_half, config=config)
        text_bottom = pytesseract.image_to_string(bottom_half, config=config)
        text_top = "".join(ch for ch in text_top if ch.isalnum()
                           or ch == "-").upper().strip()
        text_bottom = "".join(
            ch for ch in text_bottom if ch.isalnum() or ch == "-").upper().strip()
        return (text_top + "-" + text_bottom).strip("-")

    # Auto mode
    bottom_intensity = np.mean(bottom_half)
    if bottom_intensity > 250:
        text = pytesseract.image_to_string(th_big, config=config)
        return "".join(ch for ch in text if ch.isalnum() or ch == "-").upper().strip()

    text_top = pytesseract.image_to_string(top_half, config=config)
    text_bottom = pytesseract.image_to_string(bottom_half, config=config)
    text_top = "".join(ch for ch in text_top if ch.isalnum()
                       or ch == "-").upper().strip()
    text_bottom = "".join(
        ch for ch in text_bottom if ch.isalnum() or ch == "-").upper().strip()
    return (text_top + "-" + text_bottom).strip("-")


# ------------------------------------------------------------
# 3) CNN Prediction
# ------------------------------------------------------------


def predict_plate_category(image_path):
    if model is None:
        return None
    try:
        img = load_img(image_path, target_size=(224, 224))
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        pred = model.predict(arr)
        return pred
    except Exception as e:
        print(f"CNN prediction error: {e}")
        return None

# ------------------------------------------------------------
# 4) Heuristic Plate Type
# ------------------------------------------------------------


def determine_plate_type(text, plate_img):
    plate_type, color = "Unidentified", "Unknown"
    t = (text or "").upper()

    # -------------------------
    # 1️⃣ Check text prefixes first
    # -------------------------
    if "CD" in t:
        plate_type = "Diplomatic"
    elif t.startswith("FS") or t.startswith("F"):
        plate_type = "Fire Service"
    elif t.startswith("GP") or "POLICE" in t:
        plate_type = "Ghana Police"
    elif any(t.startswith(pref) for pref in ("GV", "GRA", "GOV", "GOVT")):
        plate_type = "Government"
    elif t.startswith("M"):
        plate_type = "Motorcycle"
    elif any(t.startswith(pref) for pref in ("D", "DV", "DP")):
        plate_type = "Trade"
    elif any(x in t for x in ("TAXI", "TT", "TRO")):
        plate_type = "Taxi / Trotro"

    # -------------------------
    # 2️⃣ Use color as fallback if type still unidentified
    # -------------------------
    if plate_type == "Unidentified" and plate_img is not None:
        hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        hue_mean = np.mean(h)
        sat_mean = np.mean(s)
        val_mean = np.mean(v)

        # Taxi / Trotro (Gold)
        if 20 <= hue_mean <= 40 and val_mean > 150:
            color = "Gold"
            plate_type = "Taxi / Trotro"

        # Government (Teal)
        elif 35 <= hue_mean <= 85 and val_mean > 80:
            color = "Teal"
            plate_type = "Government"

        # Motorcycle (Indigo)
        elif 110 <= hue_mean <= 140 and val_mean > 50:
            color = "Indigo"
            plate_type = "Motorcycle"

        # White / Personal (tolerant to dust/shadow)
        elif 180 <= val_mean <= 255 and sat_mean < 80:
            color = "White"
            plate_type = "Personal"

        else:
            color = "Unknown"
            plate_type = "Personal"  # fallback

    return plate_type, color


# ------------------------------------------------------------
# 5) Full Pipeline
# ------------------------------------------------------------


def classify_and_read_plate(image_path):
    print(f"\n📸 Processing: {image_path}")

    img = cv2.imread(image_path)

    choice = input(
        "\n🟦 Voulez-vous sélectionner la plaque manuellement ? (O/N) : ").strip().upper()

    if choice == "O":
        plate_region = manual_select_region(img)
        bbox = None
    else:
        plate_region, bbox = detect_plate_region(image_path)

    plate_text = read_plate_text(plate_region)
    cnn_result = predict_plate_category(image_path)

    cnn_class = None
    if cnn_result is not None:
        idx = np.argmax(cnn_result)
        class_names = ["diplomatic", "fire_service", "ghana_police", "government",
                       "motorcycle", "personal", "taxi_trotro", "trade"]
        if idx < len(class_names):
            cnn_class = class_names[idx]

    plate_type, color = determine_plate_type(plate_text, plate_region)

    print(f"🔤 Plate Text: {plate_text or 'Not detected'}")
    print(f"🎨 Color: {color}")
    print(f"🏷️ Type: {plate_type}")
    if cnn_class:
        print(f"🤖 CNN Prediction: {cnn_class.upper()}")

    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Plate Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ------------------------------------------------------------
# GUI File Picker
# ------------------------------------------------------------
# if __name__ == "__main__":
#    root = tk.Tk()
 #   root.withdraw()
#    print("📂 Please select an image file...")
#    image_path = filedialog.askopenfilename(
#        title="Select Vehicle Image",
 #       filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
#    )

#    if image_path:
 #       classify_and_read_plate(image_path)
 #   else:
#        print("❌ No image selected.")
