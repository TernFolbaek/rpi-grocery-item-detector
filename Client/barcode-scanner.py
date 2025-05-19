import cv2
from pyzbar import pyzbar
import requests
import torch
from ultralytics import YOLO  
import time
botToken = '7667739324:AAF5zhyajw13I2-ESDuWYLh9tTplWLVGzvY'
messageToken = '7731233891'

DJANGO_SERVER_URL = "http://localhost:8000/api"


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

def send_item_to_server(item_name):
    endpoint = f"{DJANGO_SERVER_URL}/detect-item/"
    data = {"name": item_name}

    try:
        response = requests.post(endpoint, json=data, timeout=5)
        response.raise_for_status()
        result = response.json()
        print(f"Item '{item_name} saved to server. Current count{result['count']}")
        return result
    except requests.RequestException as e:
        print(f"Error sending item to server: {e}")
        return None
    

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
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    model = YOLO('best.pt')  # YOLOv8 model loading

    scanned_barcodes = set()
    detected_classes = {}  # Dictionary to track detected classes and their frames count

    detection_timer = 0  # Timer to track the detection duration
    min_confidence = 0.4  # Minimum confidence threshold
    min_detect_frames = 5 # Minimum frames before being confident enough

    sent_items_cache = set()
    resend_cooldown = 5
    last_sent_time = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run inference with YOLOv8
        results = model(frame)  # YOLOv8 inference

        # Annotate frame with the results
        annotated_frame = results[0].plot()  # Get the annotated frame

        detected_object = False
        # Access the bounding box details in YOLOv8
        for box in results[0].boxes:
            xyxy = box.xyxy.numpy()  # Get box coordinates (x1, y1, x2, y2)
            conf = box.conf.numpy()  # Get confidence score
            cls_idx = int(box.cls.numpy()[0])  # Get class index
            cls_name = model.names[cls_idx] if hasattr(model,'names') else f"class_{cls_idx}"
            if conf >= min_confidence:
                if cls_name in detected_classes:
                    detected_classes[cls_name] += 1
                else:
                    detected_classes[cls_name] = 1
                
                current_time = time.time()



                # Send to server
                server_result = send_item_to_server(cls_name)
                if server_result:
                    # Update last sent time
                    last_sent_time[cls_name] = current_time

                    # Send Telegram notif
                    count = server_result.get('count', 1)
                    send_telegram_message(f"Detected {cls_name} (total count: {count})")
                
                detected_classes[cls_name] = 0
                
                cv2.putText(annotated_frame, f"{cls_name} : {conf[0]:.2f}",
                (int(xyxy[0][0]), int(xyxy[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            else:
                if cls_name in detected_classes:
                    detected_classes[cls_name] = 0


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
                    send_item_to_server(product_name)
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
