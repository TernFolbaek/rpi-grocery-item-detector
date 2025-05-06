import cv2
from pyzbar import pyzbar
import requests
import torch
from ultralytics import YOLO
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
    # Open camera with V4L2 backend for Raspberry Pi
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    # Give camera time to initialize
    time.sleep(2)
    
    if not cap.isOpened():
        print("Could not open camera.")
        return

    # Set camera parameters for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for better performance
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))  # Try MJPG format

    print("Loading YOLO model...")
    try:
        # Load YOLOv8 model - specify device='cpu' for Raspberry Pi
        model = YOLO('best.pt', task='detect')
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        cap.release()
        return
    print("YOLO model loaded successfully")

    scanned_barcodes = set()
    detection_timer = 0
    min_confidence = 0.4
    
    # Add retry logic
    retry_count = 0
    max_retries = 5

    # Test camera with a single frame
    ret, test_frame = cap.read()
    if not ret:
        print("Initial camera test failed. Trying to recover...")
    else:
        print(f"Initial camera test successful! Frame shape: {test_frame.shape}")

    print("Starting main detection loop...")
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        
        # Handle failed frame reads
        if not ret:
            print(f"Failed to grab frame, retry {retry_count+1}/{max_retries}")
            retry_count += 1
            if retry_count >= max_retries:
                print("Maximum retries reached, reopening camera")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
                time.sleep(2)
                retry_count = 0
            time.sleep(0.5)
            continue
        
        # Reset retry counter on success
        retry_count = 0
        
        try:
            # Run inference with YOLOv8
            results = model(frame, conf=0.4, iou=0.5, device='cpu')  # Specify CPU device

            # Annotate frame with the results
            annotated_frame = results[0].plot()

            detected_object = False
            # Access the bounding box details in YOLOv8
            for box in results[0].boxes:
                xyxy = box.xyxy.numpy()  # Get box coordinates
                conf = box.conf.numpy()  # Get confidence score
                cls = box.cls.numpy()    # Get class label

                if conf >= min_confidence:
                    detected_object = True
                    detection_timer += 1  # Increment timer when an object is detected
                    if detection_timer >= 10:  # Approximately 2 seconds at 5 FPS
                        send_telegram_message("Object detected with at least 40% confidence for 2 seconds.")
                        detection_timer = 0  # Reset the timer after sending the message
                    break
                else:
                    detection_timer = 0  # Reset timer if confidence drops below threshold

            # Decode any barcodes in the frame
            barcodes = pyzbar.decode(frame)
            for barcode in barcodes:
                (x, y, w, h) = barcode.rect
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

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

            # Show the frame with detections
            cv2.imshow("Merged Detection", annotated_frame)
            
            # Check for key press to exit
            if cv2.waitKey(1) == ord('q'):
                print("Quitting application...")
                break
                
        except Exception as e:
            print(f"Error in processing frame: {e}")
            time.sleep(0.5)  # Short pause on error

    # Clean up resources
    print("Releasing camera and closing windows...")
    cap.release()
    cv2.destroyAllWindows()
    print("Application terminated")

if __name__ == '__main__':
    main()