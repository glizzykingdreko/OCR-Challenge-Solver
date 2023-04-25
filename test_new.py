from ocr_reader import OcrReader

if __name__ == "__main__":
    # Example base64 string input
    base64_string = "..."

    # Extract the digits from the image
    input_image = OcrReader(base64_string, True)
    print(input_image.solve())

