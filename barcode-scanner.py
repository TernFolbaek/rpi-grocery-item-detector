import cv2
from pyzbar import pyzbar
import requests

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

    # Optional: Set resolution (adjust as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    scanned_barcodes = set()
    print("Scanning for barcodes. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Decode any barcodes in the frame
        barcodes = pyzbar.decode(frame)
        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            barcode_data = barcode.data.decode("utf-8")
            barcode_type = barcode.type

            text = f"{barcode_data} ({barcode_type})"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
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

        cv2.imshow("Barcode Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

