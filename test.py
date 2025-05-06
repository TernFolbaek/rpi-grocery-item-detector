import cv2
from pyzbar import pyzbar
import requests
import torch
from ultralytics import YOLO
from picamera2 import Picamera2
import time
import numpy as np

botToken = '7667739324:AAF5zhyajw13I2-ESDuWYLh9tTplWLVGzvY'
messageToken = '7731233891'

# Function to send message to Telegram bot
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{botToken}/sendMessage"
    params = {
        'chat_id': messageToken,
        'text': message
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error sending message: {e}")

# Function for fetching product info from Open Food Facts
def fetch_product_info(barcode):
    url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching product data: {e}")
        return None

    data = response.json()
    if data.get("status") == 1:
        return data.get("product", {})
    else:
        print(f"Product not found for barcode: {barcode}")
        return None
        

def main():
    # Initialize picamera2 instead of cv2.VideoCapture
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    
    # Give camera time to initialize
    time.sleep(2)

    model = YOLO('runs/detect/train2/weights/best.pt')  # YOLOv8 model loading

    scanned_barcodes = set()

    detection_timer = 0  # Timer to track the detection duration
    min_confidence = 0.4  # Minimum confidence threshold

    while True:
        # Get frame from picamera2
        frame = picam2.capture_array()
        
        # Run inference with YOLOv8
        results = model(frame)  # YOLOv8 inference

        # Annotate frame with the results
        annotated_frame = results[0].plot()  # Get the annotated frame

        detected_object = False
        # Access the bounding box details in YOLOv8
        for box in results[0].boxes:
            xyxy = box.xyxy.numpy()  # Get box coordinates (x1, y1, x2, y2)
            conf = box.conf.numpy()  # Get confidence score
            cls = box.cls.numpy()  # Get class label

            if conf >= min_confidence:
                detected_object = True
                detection_timer += 1  # Increment timer when an object is detected
                if detection_timer >= 1 * 10:  # 2 seconds (assuming 30 FPS)
                    send_telegram_message("Object detected with at least 40% confidence for 2 seconds.")
                    detection_timer = 0  # Reset the timer after sending the message
                break
            else:
                detection_timer = 0  # Reset timer if confidence drops below threshold

        # Decode any barcodes in the frame
        barcodes = pyzbar.decode(frame)
        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            barcode_data = barcode.data.decode("utf-8")
            barcode_type = barcode.type

            text = f"{barcode_data} ({barcode_type})"
            cv2.putText(annotated_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

            if barcode_data not in scanned_barcodes:
                scanned_barcodes.add(barcode_data)
                print(f"\nFound {barcode_type} barcode: {barcode_data}")

                product = fetch_product_info(barcode_data)
                if product:
                    product_name = product.get("product_name", "Unknown")
                    print(f"Product Name: {product_name}")
                    ingredients = product.get("ingredients_text", "N/A")
                    nutriments = product.get("nutriments", {})
                    print("Ingredients:", ingredients)
                    print("Nutriments:", nutriments)
                else:
                    print("No additional product info available.")

        cv2.imshow("Merged Detection", annotated_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    cv2.destroyAllWindows()
    picam2.stop()

if __name__ == '__main__':
    main()