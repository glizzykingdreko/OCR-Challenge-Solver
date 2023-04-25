import cv2, base64, pytesseract
import numpy as np
from typing import List
from typing import Union, List, Tuple, Any

class OcrReader:
    TESSERACT_CONFIG = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789 language_model_penalty_non_freq_dict_word=1'

    @staticmethod
    def _base64_to_image(base64_string: str) -> np.ndarray:
        """
        Converts a base64 string to an image

        Args:
            base64_string (str): The base64 string to convert
        
        Returns:
            np.ndarray: The image
        """
        img_data = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    @staticmethod
    def _rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
        """
        Rotates an image

        Args:
            image (np.ndarray): The image to rotate
            angle (int): The angle to rotate the image by
        
        Returns:
            np.ndarray: The rotated image
        """
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def _is_bold(rect: Any) -> bool:
        """
        Checks if a rect is bold

        Args:
            rect (Any): The rect to check
        
        Returns:
            bool: True if the rect is bold, False otherwise
        """
        box_points = cv2.boxPoints(rect)
        pt1, pt2, pt3, pt4 = box_points
        angle = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]) * 180 / np.pi
        if angle < -45:
            angle = 90 + angle
        aspect_ratio = rect[1][0] / rect[1][1]
        return -30 <= angle <= 30 and aspect_ratio > 1.3

    @staticmethod
    def _compute_image_score(image: np.ndarray) -> float:
        """
        Computes the score of an image

        Args:
            image (np.ndarray): The image to compute the score of
        
        Returns:
            float: The score of the image
        """
        # For this function, you can use the sum of horizontal gradient magnitudes
        # as a proxy for text alignment
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        return np.sum(np.abs(sobelx))

    @staticmethod
    def _dilate_image(image, kernel_size=(3, 3), iterations=1) -> np.ndarray:
        """
        Dilates an image

        Args:
            image (np.ndarray): The image to dilate
            kernel_size (Tuple[int, int]): The kernel size to use for the dilation
            iterations (int): The number of iterations to use for the dilation
        
        Returns:
            np.ndarray: The dilated image
        """
        kernel = np.ones(kernel_size, np.uint8)
        dilated_image = cv2.dilate(image, kernel, iterations=iterations)
        return dilated_image
    
    @staticmethod
    def _correct_skew(image, delta=1, limit=10):
        """
        Corrects the skew of an image

        Args:
            image (np.ndarray): The image to correct the skew of
            delta (int): The delta to use for the angle
            limit (int): The limit to use for the angle
        
        Returns:
            np.ndarray: The corrected image
        """
        best_score = -np.inf
        best_image = image.copy()
        for angle in range(-limit, limit + 1, delta):
            rotated_image = OcrReader._rotate_image(image, angle)
            score = OcrReader._compute_image_score(rotated_image)
            if score > best_score:
                best_score = score
                best_image = rotated_image
        return best_image
    
    @staticmethod
    def _extract_digits_from_image(
        image: np.ndarray,
        kernel_size: Tuple[int, int]=(2, 2),
        iterations: int=2,
    ) -> List[np.ndarray]:
        """
        Extracts the digits from an image
        
        Args:
            image (np.ndarray): The image to extract the digits from
            kernel_size (Tuple[int, int]): The kernel size to use for the dilation
            iterations (int): The number of iterations to use for the dilation
        
        Returns:
            List[np.ndarray]: A list of the extracted digits as images
        """
        # Convert to grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Find the contours in order to identify the digits
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.minAreaRect(c) for c in contours]
        rects = sorted(rects, key=lambda x: x[0][0])
        digit_images = []
        filtered_rects = []
        max_dist = 10
        rects = [cv2.minAreaRect(c) for c in contours]
        rects = [cv2.boxPoints(rect) for rect in rects]
        rects = sorted(rects, key=lambda x: x[0][0])
        for rect in rects:
            center = np.mean(rect, axis=0)
            if not filtered_rects:
                filtered_rects.append(rect)
                continue
            for _, filtered_rect in enumerate(filtered_rects):
                filtered_center = np.mean(filtered_rect, axis=0)
                if np.linalg.norm(center - filtered_center) < max_dist:
                    break
            else:
                filtered_rects.append(rect)
        
        # Now that we have all the rects we need to properly cut
        # the image and extract the digits
        used = []
        for d in filtered_rects:
            rect = box = d
            ida = f"{rect[0][0]}-{rect[0][1]}"
            border_size = 2
            x, y, w, h = cv2.boundingRect(box)
            if w * h < 6:
                # We skip cause the size would be too small
                continue
            x1, y1 = max(x - border_size, 0), max(y - border_size, 0)
            x2, y2 = min(x + w + border_size, image.shape[1]), min(y + h + border_size, image.shape[0])
            digit_image = image[y1:y2, x1:x2]
            digit_image_resized = cv2.resize(digit_image, (int(w * image.shape[0] / h), image.shape[0]))
            if digit_image.shape[:2] == image.shape[:2]:
                # We skip cause the size would be too big
                continue
            if ida in used:
                # We already have this digit
                continue
            if not OcrReader._is_bold(cv2.minAreaRect(d)):
                # We make the italic digits bold by increasing border and
                # rotating it a bit
                digit_image_resized = OcrReader._correct_skew(digit_image_resized, delta=1, limit=10)
                digit_image_resized = OcrReader._dilate_image(digit_image_resized, kernel_size=kernel_size, iterations=iterations)
            
            # Save the extracted digit
            used.append(ida) 
            digit_images.append(digit_image_resized)
        return digit_images

    @staticmethod
    def _combine_digit_images(digit_images: List[np.ndarray]) -> np.ndarray:
        """
        Combine the digit images into a single image.

        Args:
            digit_images (List[np.ndarray]): The list of digit images
        
        Returns:
            np.ndarray: The combined image
        """
        combined_image = np.hstack(digit_images)
        return combined_image
    
    @staticmethod
    def _read_digits_from_image(image: np.ndarray) -> str:
        """
        Read the digits from the image using Tesseract OCR.

        Args:
            image (np.ndarray): The image to read the digits from
        
        Returns:
            str: The digits extracted from the image
        """
        ocr_text = pytesseract.image_to_string(image, config=OcrReader.TESSERACT_CONFIG)
        return "".join([str(char) for char in ocr_text if char.isdigit()])
    
    @staticmethod
    def _debug_image_creation(
        initial_image: np.ndarray,
        output_image: np.ndarray,
        output: List[int]
    ) -> np.ndarray:
        """
        Save the initial challenge image, edited image, and the output digits
        into a final output image to understand the process.

        Args:
            initial_image (np.ndarray): The initial challenge image
            output_image (np.ndarray): The edited image
            output (List[int]): The list of digits extracted from the image
        
        Returns:
            np.ndarray: The final output image
        """
        # Convert images to grayscale if they have 3 channels
        if len(initial_image.shape) == 3 and initial_image.shape[2] == 3:
            initial_image = cv2.cvtColor(initial_image, cv2.COLOR_BGR2GRAY)

        if len(output_image.shape) == 3 and output_image.shape[2] == 3:
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)

        # Convert initial_image to black on white
        initial_image = cv2.bitwise_not(initial_image)

        # Get dimensions of initial and output images
        h1, w1 = initial_image.shape[:2]
        h2, w2 = output_image.shape[:2]

        # Set the height of the black banner
        banner_height = 50

        # Create a new blank image with the combined width of initial and output images and the height of the black banner
        combined_image = np.zeros((max(h1, h2) + banner_height, w1 + w2), dtype=np.uint8)

        # Place the initial image on the left side
        combined_image[:h1, :w1] = initial_image

        # Place the output image on the right side
        combined_image[:h2, w1:] = output_image

        # Write the solution on the black banner
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Solution: {''.join([str(i) for i in output])}"
        font_scale = 1
        font_thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

        # Calculate the position of the text so it's centered on the black banner
        text_x = (combined_image.shape[1] - text_size[0]) // 2
        text_y = combined_image.shape[0] - banner_height // 2 + text_size[1] // 2

        # Write the text on the image
        cv2.putText(combined_image, text, (text_x, text_y), font, font_scale, 255, font_thickness)

        return combined_image
        
    def __init__(
        self,
        base64image: str,
        debug: bool = False
    ) -> None:
        """
        # OcrReader
        This class is used to solve simple OCR challenges using Tesseract OCR.

        It extracts the digits from the image, combines them into a single image,
        and then reads the digits using Tesseract OCR.

        Args:
            `base64image` (str): The base64 encoded image
            `debug` (bool): Whether to save the debug image or not

        ## Usage
        ```python
        from ocr_reader import OcrReader

        # Create an instance of the OcrReader class
        ocr_reader = OcrReader(base64image)

        # Solve the challenge
        solution = ocr_reader.solve()

        # Print the solution
        print(solution)
        >>> "123456"
        ```

        ## Debugging
        If you want to save the debug image, you can pass `debug=True` to the constructor.
        ```python
        from ocr_reader import OcrReader

        # Create an instance of the OcrReader class
        ocr_reader = OcrReader(base64image, debug=True)

        # Solve the challenge
        solution = ocr_reader.solve()

        # Print the solution
        print(solution)
        >>> "123456"

        # The debug image will be saved as `solved_123456.png`
        ```
        """
        self.base64image, self.debug = \
            base64image, debug

    def solve(self) -> Union[str, bool]:
        """
        Solve the challenge.

        Returns:
            Union[str, bool]: The solution or False if the challenge is not solvable
        """
        try:
            input_image = OcrReader._base64_to_image(self.base64image)
        except Exception as e:
            raise Exception("Invalid base64 image") from e
        digit_images = OcrReader._extract_digits_from_image(input_image)
        images_count = len(digit_images)
        combined_image = OcrReader._combine_digit_images(digit_images)
        digits = OcrReader._read_digits_from_image(combined_image)
        if self.debug:
            # Save image for debugging
            debug_image = self._debug_image_creation(
                input_image,
                combined_image,
                digits
            )
            cv2.imwrite(
                f"solved_{digits}.png",
                debug_image
            )
        if len(digits) != images_count:
            # This is a first attempt to solve the challenge
            # after a failed attempt, by inverting the image
            # and applying dilation and erosion to help with
            # recognizing "7" and "1"
            inverted = cv2.bitwise_not(combined_image)
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(inverted, kernel, iterations=2)
            eroded = cv2.erode(dilated, kernel, iterations=2)
            # Apply additional dilation to help with recognizing "7"
            kernel = np.ones((2, 2), np.uint8)
            dilated_seven = cv2.dilate(eroded, kernel, iterations=5)
            digits = OcrReader._read_digits_from_image(dilated_seven)
            if self.debug:
                # Save image for debugging
                debug_image = self._debug_image_creation(
                    input_image,
                    dilated_seven,
                    digits
                )
                cv2.imwrite(
                    f"solved_adv_{digits}.png",
                    debug_image
                )
        return digits if len(digits) == images_count else False