import cv2
from pyzbar import pyzbar
import requests
import torch
import threading
import time

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
        
# Main function to toggle between modes
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    # Optional: Set resolution (adjust as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp9/weights/best.pt', force_reload=True)
    scanned_barcodes = set()

    detection_timer = 0  # Timer to track the detection duration
    min_confidence = 0.4  # Minimum confidence threshold

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run inference with YOLOv5
        results = model(frame)
        # Annotate frame with the results
        annotated_frame = results.render()[0].copy()

        detected_object = False
        for *xyxy, conf, cls in results.xywh[0]:
            if conf >= min_confidence:
                detected_object = True
                detection_timer += 1  # Increment timer when an object is detected
                if detection_timer >= 1 * 10:  # 2 seconds (assuming 30 FPS)
                    send_telegram_message("Object detected with at least 50% confidence for 2 seconds. {results}")
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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
