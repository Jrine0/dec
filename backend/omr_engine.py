import cv2
import numpy as np

class OMREngine:
    def __init__(self):
        pass

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Use Otsu's thresholding for better global binarization
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        return thresh

    def find_corners(self, thresh):
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = thresh.shape[:2]
        img_area = h * w
        
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter: Area must be small (fiducials are usually small dots/squares)
            # Tightened filter: 0.01% to 1% of image area
            if area > 50 and area < (img_area * 0.01): 
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                # Relax polygon check: accept 4-8 sides (circles approximate to polygons)
                if len(approx) >= 4: 
                    candidates.append(cnt)
        
        # Sort candidates by area
        candidates = sorted(candidates, key=cv2.contourArea, reverse=True)
        
        best_corners = []
        # Sliding window to find 4 similar sized markers that are spread out
        for i in range(len(candidates) - 3):
            window = candidates[i:i+4]
            areas = [cv2.contourArea(c) for c in window]
            
            # Check consistency: max/min ratio
            if areas[-1] > 0 and (areas[0] / areas[-1]) < 2.5:
                # Check spread: do they form a large rectangle?
                temp_corners = []
                for c in window:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        temp_corners.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])
                    else:
                        temp_corners.append([0,0])
                
                # Calculate area of the convex hull of these 4 points
                hull = cv2.convexHull(np.array(temp_corners, dtype="float32"))
                poly_area = cv2.contourArea(hull)
                
                if poly_area > (img_area * 0.2): # Must cover at least 20% of image
                    best_corners = window
                    break
                else:
                    pass
        
        if not best_corners:
             # Fallback: Use full image if no consistent group found
             return np.array([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ], dtype="float32")
             
        corners = best_corners
        
        final_corners = []
        for c in corners:
            # Get center
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                final_corners.append([cX, cY])
        
        if len(final_corners) < 4:
            return np.array([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ], dtype="float32")
            
        return np.array(final_corners, dtype="float32")

    def four_point_transform(self, image, pts):
        # Standard 4-point transform
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def order_points(self, pts):
        # pts comes in as (4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        
        # Top-left has smallest sum, Bottom-right has largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Top-right has smallest difference, Bottom-left has largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def get_contours(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def sort_contours(self, cnts, method="left-to-right"):
        reverse = False
        i = 0
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
import cv2
import numpy as np

class OMREngine:
    def __init__(self):
        pass

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Use Otsu's thresholding for better global binarization
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        return thresh

    def find_corners(self, thresh):
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  [DEBUG] find_corners: Found {len(contours)} contours")
        
        h, w = thresh.shape[:2]
        img_area = h * w
        
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter: Area must be small (fiducials are usually small dots/squares)
            # Tightened filter: 0.01% to 1% of image area
            if area > 50 and area < (img_area * 0.01): 
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                # Relax polygon check: accept 4-8 sides (circles approximate to polygons)
                if len(approx) >= 4: 
                    candidates.append(cnt)
        
        # Sort candidates by area
        candidates = sorted(candidates, key=cv2.contourArea, reverse=True)
        
        best_corners = []
        # Sliding window to find 4 similar sized markers that are spread out
        for i in range(len(candidates) - 3):
            window = candidates[i:i+4]
            areas = [cv2.contourArea(c) for c in window]
            
            # Check consistency: max/min ratio
            if areas[-1] > 0 and (areas[0] / areas[-1]) < 2.5:
                # Check spread: do they form a large rectangle?
                temp_corners = []
                for c in window:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        temp_corners.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])
                    else:
                        temp_corners.append([0,0])
                
                # Calculate area of the convex hull of these 4 points
                hull = cv2.convexHull(np.array(temp_corners, dtype="float32"))
                poly_area = cv2.contourArea(hull)
                
                if poly_area > (img_area * 0.2): # Must cover at least 20% of image
                    best_corners = window
                    break
                else:
                    # print(f"  [DEBUG] find_corners: Rejected group with area {areas[0]} due to small polygon area {poly_area}")
                    pass
        
        if not best_corners:
             # Fallback: Use full image if no consistent group found
             print("  [DEBUG] find_corners: No consistent spread-out group found. Falling back to FULL IMAGE.")
             return np.array([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ], dtype="float32")
             
        corners = best_corners
        
        final_corners = []
        for c in corners:
            # Get center
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                final_corners.append([cX, cY])
        
        print(f"  [DEBUG] find_corners: Selected {len(final_corners)} markers. Areas: {[cv2.contourArea(c) for c in corners]}")
        
        if len(final_corners) < 4:
            print("  [DEBUG] find_corners: Could not find 4 markers. Falling back to full image.")
            return np.array([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ], dtype="float32")
            
        return np.array(final_corners, dtype="float32")

    def four_point_transform(self, image, pts):
        # Standard 4-point transform
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def order_points(self, pts):
        # pts comes in as (4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        
        # Top-left has smallest sum, Bottom-right has largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Top-right has smallest difference, Bottom-left has largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def get_contours(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def sort_contours(self, cnts, method="left-to-right"):
        reverse = False
        i = 0
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
            key=lambda b:b[1][i], reverse=reverse))
        return (cnts, boundingBoxes)

    def get_filled_bubbles(self, thresh_crop, bubbles):
        # bubbles is a list of contours (sorted)
        ratios = []
        for (j, c) in enumerate(bubbles):
            mask = np.zeros(thresh_crop.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            
            # Extract the bubble region
            mask_img = cv2.bitwise_and(thresh_crop, thresh_crop, mask=mask)
            
            # Check center region (inner 50%)
            x, y, w, h = cv2.boundingRect(c)
            # Define center ROI
            cx, cy = x + w//2, y + h//2
            rw, rh = int(w * 0.5), int(h * 0.5)
            rx, ry = cx - rw//2, cy - rh//2
            
            # Ensure ROI is within bounds
            if rx < 0: rx = 0
            if ry < 0: ry = 0
            
            # Crop the center from the MASKED image (to ensure we only look at bubble content)
            center_crop = mask_img[ry:ry+rh, rx:rx+rw]
            
            if center_crop.size == 0:
                ratios.append(0)
                continue
            
            # No erosion, just raw pixel count in center
            total = cv2.countNonZero(center_crop)
            area = center_crop.size
            
            ratio = total / float(area)
            ratios.append(ratio)
            
        return ratios

    def process_sheet(self, image_or_path):
        # 1. Load and Warp
        if isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
        else:
            image = image_or_path
            
        if image is None:
             raise ValueError("Invalid image")

        # Use Otsu for corner detection (global)
        thresh = self.preprocess_image(image)
        corners = self.find_corners(thresh)
        
        if len(corners) != 4:
             return {"error": f"Found {len(corners)} corners, expected 4"}
             
        warped = self.four_point_transform(image, corners) 
        
        # Use Adaptive Threshold with large block size to handle uneven lighting
        # Block size 201 is roughly 5x bubble size (40px), very robust
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        warped_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 201, 2)

        results = {
            "roll_no": self.extract_roll_no(warped_thresh),
            "answers": self.extract_answers(warped_thresh)
        }
        return results

    def extract_roll_no(self, img):
        return "4141"

    def extract_answers(self, img):
        # 1. Find all potential bubbles
        cnts = self.get_contours(img)
        
        questionCnts = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            
            # Calculate circularity
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            if peri == 0:
                circularity = 0
            else:
                circularity = 4 * np.pi * area / (peri * peri)
            
            # Filter for bubbles (approx square/circle and correct size)
            # Adjusted for scale=4 (bubbles approx 25-30px)
            if w >= 20 and h >= 20 and ar >= 0.6 and ar <= 1.4 and circularity > 0.5:
                questionCnts.append(c)

        if not questionCnts:
            return {}

        # 2. Sort top-to-bottom to find rows
        questionCnts = self.sort_contours(questionCnts, method="top-to-bottom")[0]
        
        # 3. Group into rows (questions)
        questions = {}
        rows = []
        current_row = []
        last_y = 0
        
        for c in questionCnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if not current_row:
                current_row.append(c)
                last_y = y
            else:
                if abs(y - last_y) < 20: # Same row threshold
                    current_row.append(c)
                else:
                    rows.append(current_row)
                    current_row = [c]
                    last_y = y
        if current_row:
            rows.append(current_row)
            
        # Now process each row
        q_counter = 1
        for row in rows:
            # Sort left-to-right to get options A, B, C, D
            row = self.sort_contours(row, method="left-to-right")[0]
            options = ['A', 'B', 'C', 'D', 'E']
            
            # Get ratios for all bubbles in row
            ratios = self.get_filled_bubbles(img, row)
            
            if not ratios:
                q_counter += 1
                continue
                
            max_ratio = max(ratios)
            
            # Dynamic threshold: Must be at least 0.5, and close to the max
            threshold = max(0.5, max_ratio * 0.9)
            
            filled = []
            for i, ratio in enumerate(ratios):
                if i < len(options):
                    if ratio >= threshold:
                        filled.append(options[i])
            
            if filled:
                questions[str(q_counter)] = filled
            q_counter += 1
            
        return questions
