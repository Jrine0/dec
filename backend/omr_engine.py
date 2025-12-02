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
        
        if not best_corners:
             # Fallback: Use full image if no consistent group found
             print("DEBUG: find_corners: Fallback to FULL IMAGE", flush=True)
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
            print("DEBUG: find_corners: Fallback (len < 4)", flush=True)
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
        print("DEBUG: process_sheet called", flush=True)
        # 1. Load and Warp
        if isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
        else:
            image = image_or_path
            
        if image is None:
             raise ValueError("Invalid image")

        # Use Otsu for corner detection (global)
        thresh = self.preprocess_image(image)
        fields = self.extract_fields(warped_thresh)
        
        return fields

    def extract_fields(self, img):
        # img is grayscale
        
        # 1. Edge Detection to find blocks
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        
        cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        
        results = {
            "center_code": "",
            "roll_no": "",
            "set": "",
            "student_name": "",
            "answers": {},
            "marks": 0,
            "total_marks": 0
        }
        
        if len(cnts) < 5:
            print("DEBUG: Not enough contours found to identify all fields", flush=True)
            return results

        # 2. Identify ROIs based on Area and Position
        # Expected Areas (approx):
        # Answers: > 2,000,000 (Largest)
        # Name: > 1,500,000 (2nd Largest)
        # Top Blocks (Center, Roll, Set): ~100,000 - 200,000
        
        answer_contour = cnts[0]
        name_contour = cnts[1]
        
        # Filter for top blocks
        top_blocks = []
        for c in cnts[2:]:
            area = cv2.contourArea(c)
            if 100000 < area < 200000:
                top_blocks.append(c)
        
        # Sort top blocks left-to-right
        top_blocks, _ = self.sort_contours(top_blocks, method="left-to-right")
        
        # We expect at least 3 blocks: Center Code, Roll No, Set
        # But we might find more (e.g. instruction blocks). 
        # Based on user input:
        # Center Code is Left
        # Roll No is Middle
        # Set is Right (but left of Name)
        
        # Let's try to map them by index if we have exactly 3, or use heuristics
        center_code_contour = None
        roll_no_contour = None
        set_contour = None
        
        if len(top_blocks) >= 3:
             # Assuming the 3 required blocks are the ones we found
             # We need to be careful about "instruction" blocks.
             # User said: Pink#4(Center), Yellow#3(Roll), Red#6(Set)
             # In sorted order (x-pos): Center < Roll < Set
             
             # Let's filter by Y position too? They should be roughly at the same height.
             # But simply taking the first 3 sorted by X might work if they are the main blocks.
             
             # Refined strategy: Take the 3 largest from the "top_blocks" list?
             # The user identified #3, #4, #6. 
             # #2 (399k) and #5 (125k) were ignored. #5 is close in size to Set (#6 is 115k).
             # Let's strictly use the user's identified areas if possible, or relative order.
             
             # Center Code (Leftmost)
             center_code_contour = top_blocks[0]
             
             # Roll No (Middle)
             if len(top_blocks) > 1:
                 roll_no_contour = top_blocks[1]
                 
             # Set (Rightmost of the small blocks)
             # Note: Set might be smaller or positioned differently.
             # Let's look for the block that is to the right of Roll No.
             if len(top_blocks) > 2:
                 set_contour = top_blocks[2]
        
        # 3. Process ROIs
        
        # Answers
        # Warp the Answer ROI to ensure it's straight
        peri = cv2.arcLength(answer_contour, True)
        approx = cv2.approxPolyDP(answer_contour, 0.02 * peri, True)
        
        if len(approx) == 4:
             # Use the detected corners to warp
             warped_answers = self.four_point_transform(img, approx.reshape(4, 2))
             print(f"DEBUG: Warped Answer ROI", flush=True)
             
             # Now find bubbles in the warped ROI
             roi_thresh = cv2.adaptiveThreshold(warped_answers, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
             roi_cnts = self.get_contours(roi_thresh)
             answer_bubbles = []
             for c in roi_cnts:
                 if cv2.contourArea(c) > 30:
                     answer_bubbles.append(c) # Local coordinates
             
             if answer_bubbles:
                  results["answers"] = self.process_answers_block(warped_answers, answer_bubbles)
        else:
             # Fallback to bounding rect if corners not found
             x, y, w, h = cv2.boundingRect(answer_contour)
             print(f"DEBUG: Processing Answers ROI (Rect): {x},{y},{w},{h}", flush=True)
             roi = img[y:y+h, x:x+w]
             
             roi_thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
             roi_cnts = self.get_contours(roi_thresh)
             answer_bubbles = []
             for c in roi_cnts:
                 if cv2.contourArea(c) > 30:
                     answer_bubbles.append(c)
             
             if answer_bubbles:
                  results["answers"] = self.process_answers_block(roi, answer_bubbles)

        # Student Name
        x, y, w, h = cv2.boundingRect(name_contour)
        print(f"DEBUG: Processing Student Name ROI: {x},{y},{w},{h}", flush=True)
        roi = img[y:y+h, x:x+w]
        results["student_name"] = self.process_text_block(roi)

        # Center Code
        if center_code_contour is not None:
            x, y, w, h = cv2.boundingRect(center_code_contour)
            print(f"DEBUG: Processing Center Code ROI: {x},{y},{w},{h}", flush=True)
            roi = img[y:y+h, x:x+w]
            results["center_code"] = self.process_numeric_block(roi)
            
        # Roll No
        if roll_no_contour is not None:
            x, y, w, h = cv2.boundingRect(roll_no_contour)
            print(f"DEBUG: Processing Roll No ROI: {x},{y},{w},{h}", flush=True)
            roi = img[y:y+h, x:x+w]
            results["roll_no"] = self.process_numeric_block(roi)

        # Set
        if set_contour is not None:
            x, y, w, h = cv2.boundingRect(set_contour)
            print(f"DEBUG: Processing Set ROI: {x},{y},{w},{h}", flush=True)
            roi = img[y:y+h, x:x+w]
            results["set"] = self.process_set_block(roi)

        return results

        return results

    def process_numeric_block(self, roi):
        # roi is grayscale
        # Threshold for both contours and filling
        thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
        
        # Morphological opening to remove grid lines/noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # 1. Find bubbles
        cnts = self.get_contours(thresh)
        
        bubbles = []
        for c in cnts:
            if cv2.contourArea(c) > 30:
                bubbles.append(c)
        
        if not bubbles: return ""
        
        # Sort left-to-right to find columns
        bubbles, _ = self.sort_contours(bubbles, method="left-to-right")
        
        # Group into columns
        columns = []
        current_col = []
        last_x = -100
        
        for c in bubbles:
            x, y, w, h = cv2.boundingRect(c)
            if not current_col:
                current_col.append(c)
                last_x = x
            else:
                if abs(x - last_x) < 20: # Same column
                    current_col.append(c)
                else:
                    columns.append(current_col)
                    current_col = [c]
                    last_x = x
        if current_col:
            columns.append(current_col)
            
        result = ""
        for col in columns:
            # Sort top-to-bottom
            col, _ = self.sort_contours(col, method="top-to-bottom")
            
            # Check which bubble is filled
            ratios = self.get_filled_bubbles(thresh, col)
            if not ratios: continue
            
            max_idx = np.argmax(ratios)
            if ratios[max_idx] > 0.5: # Threshold
                result += str(max_idx)
            else:
                result += "?"
                
        return result

    def process_text_block(self, roi):
        # roi is grayscale
        # Threshold for both contours and filling
        thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
        
        cnts = self.get_contours(thresh)
        
        bubbles = []
        for c in cnts:
            if cv2.contourArea(c) > 30:
                bubbles.append(c)
        
        if not bubbles: return ""
        
        bubbles, _ = self.sort_contours(bubbles, method="left-to-right")
        
        columns = []
        current_col = []
        last_x = -100
        
        for c in bubbles:
            x, y, w, h = cv2.boundingRect(c)
            if not current_col:
                current_col.append(c)
                last_x = x
            else:
                if abs(x - last_x) < 20: 
                    current_col.append(c)
                else:
                    columns.append(current_col)
                    current_col = [c]
                    last_x = x
        if current_col:
            columns.append(current_col)
            
        result = ""
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for col in columns:
            col, _ = self.sort_contours(col, method="top-to-bottom")
            ratios = self.get_filled_bubbles(thresh, col)
            
            if not ratios: continue
            
            max_idx = np.argmax(ratios)
            if ratios[max_idx] > 0.5:
                if max_idx < len(alphabet):
                    result += alphabet[max_idx]
            else:
                result += " " 
                
        return result.strip()

    def process_set_block(self, roi):
        # roi is grayscale
        # Threshold for both contours and filling
        thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
        
        cnts = self.get_contours(thresh)
        
        bubbles = []
        for c in cnts:
            if cv2.contourArea(c) > 30:
                bubbles.append(c)
        
        if not bubbles: return ""
        
        # Assuming single row or column
        # Sort left-to-right
        bubbles, _ = self.sort_contours(bubbles, method="left-to-right")
        ratios = self.get_filled_bubbles(thresh, bubbles)
        
        options = ['A', 'B', 'C', 'D']
        for i, r in enumerate(ratios):
            if r > 0.5:
                if i < len(options):
                    return options[i]
        return ""

    def process_answers_block(self, img, cluster):
        # Compute threshold for filling check
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
        
        # Calculate average bubble dimensions
        heights = [cv2.boundingRect(c)[3] for c in cluster]
        widths = [cv2.boundingRect(c)[2] for c in cluster]
        if not heights or not widths: return {}
        
        avg_h = sum(heights) / len(heights)
        avg_w = sum(widths) / len(widths)
        
        # Thresholds
        col_gap_thresh = avg_w * 2.0 
        row_gap_thresh = avg_h * 0.8 # Increased from 0.6 to prevent splitting rows 
        
        # 1. Sort left-to-right to find columns
        cluster, _ = self.sort_contours(cluster, method="left-to-right")
        
        columns = []
        current_col = []
        last_x = -1000
        
        for c in cluster:
            x, y, w, h = cv2.boundingRect(c)
            if not current_col:
                current_col.append(c)
                last_x = x
            else:
                if abs(x - last_x) > col_gap_thresh:
                    columns.append(current_col)
                    current_col = [c]
                else:
                    current_col.append(c)
                last_x = x
        if current_col:
            columns.append(current_col)
            
        print(f"DEBUG: Found {len(columns)} columns of questions", flush=True)
        
        # Force 4 columns if we have enough bubbles but grouping failed
        if len(columns) < 4 and len(cluster) > 50:
             print("DEBUG: Force splitting into 4 columns using clustering", flush=True)
             
             # Cluster X-coordinates to find the main block
             x_coords = sorted([cv2.boundingRect(c)[0] for c in cluster])
             clusters = []
             if x_coords:
                 current_cluster = [x_coords[0]]
                 for x in x_coords[1:]:
                     if x - current_cluster[-1] > 50: # Gap > 50px starts new cluster
                         clusters.append(current_cluster)
                         current_cluster = [x]
                     else:
                         current_cluster.append(x)
                 clusters.append(current_cluster)
             
             # Find the main cluster (max count)
             main_cluster_x = max(clusters, key=len)
             min_x = min(main_cluster_x)
             max_x = max(main_cluster_x)
             
             print(f"DEBUG: Main cluster range: {min_x}-{max_x} (Count: {len(main_cluster_x)})", flush=True)
             
             # Filter bubbles to only those in the main cluster
             main_bubbles = []
             for c in cluster:
                 bx = cv2.boundingRect(c)[0]
                 if min_x <= bx <= max_x:
                     main_bubbles.append(c)
             
             # Split main block into 4
             total_width = max_x - min_x
             col_width = total_width / 4
             
             columns = [[], [], [], []]
             for c in main_bubbles:
                 x, _, _, _ = cv2.boundingRect(c)
                 idx = int((x - min_x) / col_width)
                 if idx >= 4: idx = 3
                 columns[idx].append(c)
        
        answers = {}
        q_counter = 1
        
        # 2. Process each column
        for col_bubbles in columns:
            # Sort top-to-bottom to find rows (questions)
            col_bubbles, _ = self.sort_contours(col_bubbles, method="top-to-bottom")
            
            rows = []
            current_row = []
            last_y = -1000
            
            for c in col_bubbles:
                x, y, w, h = cv2.boundingRect(c)
                if not current_row:
                    current_row.append(c)
                    last_y = y
                else:
                    if abs(y - last_y) < row_gap_thresh:
                        current_row.append(c)
                    else:
                        rows.append(current_row)
                        current_row = [c]
                        last_y = y
            if current_row:
                rows.append(current_row)
            
            print(f"DEBUG: Column has {len(rows)} rows", flush=True)
            
            # Process rows in this column
            for row in rows:
                # Sort left-to-right (A, B, C, D)
                row = self.sort_contours(row, method="left-to-right")[0]
                options = ['A', 'B', 'C', 'D']
                
                ratios = self.get_filled_bubbles(thresh, row)
                # DEBUG: Print ratios for the first few rows
                if q_counter <= 5:
                     print(f"DEBUG: Q{q_counter} Ratios: {[round(r, 2) for r in ratios]}", flush=True)
                
                selected = []
                for i, r in enumerate(ratios):
                    if r > 0.25: # Lowered Threshold from 0.5
                        if i < len(options):
                            selected.append(options[i])
                            
                if selected:
                    answers[str(q_counter)] = selected
                q_counter += 1
                
        return answers

