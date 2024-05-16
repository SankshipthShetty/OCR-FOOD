import cv2
from pyzbar.pyzbar import ZBarSymbol, decode
import json

def read_barcode(image_path):

    frame = cv2.imread(image_path)


    barcodes = decode(frame, symbols=[ZBarSymbol.EAN13])


    if barcodes:
        for barcode in barcodes:

            barcode_data = barcode.data.decode('utf-8')
            return barcode_data

    return None


image_path = "C:/Users/sanks/OneDrive/Pictures/Screenshots/data2.jpg"


barcode_data = read_barcode(image_path)

if barcode_data:

    json_data = {"barcode_data": barcode_data}
    with open(".venv/barcode_data.json", "w") as json_file:
        json.dump(json_data, json_file)
    print("Barcode data saved successfully.")
else:
    print("No barcode detected in the image.")
