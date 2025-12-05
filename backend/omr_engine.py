import cv2
import numpy as np

class OMREngine:
    def __init__(self):
        pass

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
        # initialize the reverse flag and sort index
        reverse = False
        i = 0
        
        if not cnts:
            return [], []

        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
            
        # handle if we are sorting against the y-coordinate rather than the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
            
        # construct the list of bounding boxes and sort them from top to bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
            key=lambda b:b[1][i], reverse=reverse))
            
        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)

    def process_sheet(self, image_or_path):
        # 1. Load and Warp
        if isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
        else:
            image = image_or_path
            
        if image is None:
             raise ValueError("Invalid image")

        # We will extract fields directly from the original image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fields = self.extract_fields(gray)
        
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
            x, y, w, h = cv2.boundingRect(c)
            # Must be in top half (approx) and correct size
            if 100000 < area < 200000 and y < 1000:
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
             cv2.imwrite("debug_warped_answers.jpg", warped_answers)
             
             # Now find bubbles in the warped ROI
             roi_thresh = cv2.adaptiveThreshold(warped_answers, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
             
             # Apply morphology to remove grid lines
             kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
             roi_thresh = cv2.morphologyEx(roi_thresh, cv2.MORPH_OPEN, kernel)
             
             roi_cnts = self.get_contours(roi_thresh)
             
             # Filter by area (Median-based)
             all_bubbles = [c for c in roi_cnts if cv2.contourArea(c) > 30]
             
             # DEBUG: Log all bubble centroids for slant analysis
             debug_centroids = []
             for c in all_bubbles:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    debug_centroids.append((cx, cy))
             print(f"DEBUG: All Bubble Centroids: {debug_centroids}", flush=True)

             if all_bubbles:
                 areas = [cv2.contourArea(c) for c in all_bubbles]
                 median_area = np.median(areas)
                 print(f"DEBUG: Median Bubble Area: {median_area}", flush=True)
                 
                 answer_bubbles = []
                 for c in all_bubbles:
                     # Filter out edge noise (x < 20)
                     if cv2.boundingRect(c)[0] < 20:
                         continue
                         
                     if 0.2 * median_area < cv2.contourArea(c) < 10.0 * median_area:
                         answer_bubbles.append(c) # Local coordinates
             
                 # DEBUG: Draw detected bubbles
                 debug_bubbles_img = cv2.cvtColor(warped_answers, cv2.COLOR_GRAY2BGR)
                 cv2.drawContours(debug_bubbles_img, answer_bubbles, -1, (0, 255, 0), 1)
                 cv2.imwrite("debug_bubbles_detected.jpg", debug_bubbles_img)

             else:
                 answer_bubbles = []
             
             print(f"DEBUG: Found {len(answer_bubbles)} bubbles after filtering (from {len(all_bubbles)})", flush=True)
             
             if answer_bubbles:
                  results["answers"] = self.process_answers_block(warped_answers, answer_bubbles)
             
             print(f"DEBUG: Found {len(answer_bubbles)} bubbles in Warped Answer ROI", flush=True)
             
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
            results["center_code"] = self.process_numeric_block(roi, max_columns=2)
            
        # Roll No
        if roll_no_contour is not None:
            x, y, w, h = cv2.boundingRect(roll_no_contour)
            print(f"DEBUG: Processing Roll No ROI: {x},{y},{w},{h}", flush=True)
            roi = img[y:y+h, x:x+w]
            results["roll_no"] = self.process_numeric_block(roi, max_columns=4)

        # Set
        if set_contour is not None:
            x, y, w, h = cv2.boundingRect(set_contour)
            print(f"DEBUG: Processing Set ROI: {x},{y},{w},{h}", flush=True)
            roi = img[y:y+h, x:x+w]
            results["set"] = self.process_set_block(roi)

        return results

    def process_numeric_block(self, roi, max_columns=None):
        # ROI-based Grid Sampling for Numeric Blocks (Center Code, Roll No)
        # Assumes 12 rows (2 headers + 10 digits)
        
        thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        h, w = roi.shape
        # Estimate columns based on width or max_columns
        # If max_columns is provided, use it to define grid
        if max_columns:
            num_cols = max_columns
        else:
            # Fallback: estimate based on aspect ratio? 
            # For now, let's assume standard width per column if not provided
            # But usually max_columns IS provided now.
            num_cols = 4 # Default
            
        num_rows = 12 # 2 headers + 10 digits
        
        col_w = w / num_cols
        row_h = h / num_rows
        
        result = ""
        
        for c_idx in range(num_cols):
            x1 = int(c_idx * col_w)
            x2 = int((c_idx + 1) * col_w)
            
            best_r_idx = -1
            max_ratio = 0.0
            
            for r_idx in range(num_rows):
                y1 = int(r_idx * row_h)
                y2 = int((r_idx + 1) * row_h)
                
                pad_x = int(col_w * 0.25)
                pad_y = int(row_h * 0.25)
                
                if y2-pad_y > y1+pad_y and x2-pad_x > x1+pad_x:
                    cell = thresh[y1+pad_y:y2-pad_y, x1+pad_x:x2-pad_x]
                    nz = cv2.countNonZero(cell)
                    area = cell.size
                    if area > 0:
                        ratio = nz / area
                        if ratio > max_ratio:
                            max_ratio = ratio
                            best_r_idx = r_idx
            
            # Map Row to Digit (Row 2 -> 0)
            val_idx = best_r_idx - 2
            
            if max_ratio > 0.2 and 0 <= val_idx <= 9:
                result += str(val_idx)
            else:
                # result += "?" # Don't output ? for final result, maybe just skip or use 0?
                # User prefers 00 if empty? Or just empty string?
                # Let's output 0 if ambiguous? No, better to be empty or ?
                # Previous logic outputted ?
                # But for Center Code "00" was preferred.
                # Let's stick to digits. If fails, maybe "0"?
                # Actually, if max_ratio is low, it's likely empty.
                # For Center Code "01", we need "0" and "1".
                # If we miss, we get nothing.
                pass
                
        # If result length < max_columns, pad?
        # For now, return what we found.
        return result

    def process_set_block(self, roi):
        # ROI-based Grid Sampling for Set
        # Assumes 6 rows (2 headers + 4 options)
        
        thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        h, w = roi.shape
        num_cols = 1
        num_rows = 6
        
        col_w = w / num_cols
        row_h = h / num_rows
        
        options = ['A', 'B', 'C', 'D']
        
        best_r_idx = -1
        max_ratio = 0.0
        
        for r_idx in range(num_rows):
            y1 = int(r_idx * row_h)
            y2 = int((r_idx + 1) * row_h)
            
            pad_x = int(col_w * 0.25)
            pad_y = int(row_h * 0.25)
            
            if y2-pad_y > y1+pad_y:
                cell = thresh[y1+pad_y:y2-pad_y, int(col_w*0.2):int(col_w*0.8)]
                nz = cv2.countNonZero(cell)
                area = cell.size
                if area > 0:
                    ratio = nz / area
                    if ratio > max_ratio:
                        max_ratio = ratio
                        best_r_idx = r_idx
        
        # Map Row to Option (Row 2 -> A)
        opt_idx = best_r_idx - 2
        
        if max_ratio > 0.2 and 0 <= opt_idx < 4:
            return options[opt_idx]
            
        return ""
    def process_text_block(self, roi):
        # ROI-based Grid Sampling for Student Name
        # Assumes 20 columns and 28 rows (2 headers + 26 letters)
        
        # Thresholding
        thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        h, w = roi.shape
        num_cols = 20
        num_rows = 28
        
        col_w = w / num_cols
        row_h = h / num_rows
        
        result = ""
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        for c_idx in range(num_cols):
            x1 = int(c_idx * col_w)
            x2 = int((c_idx + 1) * col_w)
            
            best_r_idx = -1
            max_ratio = 0.0
            
            for r_idx in range(num_rows):
                y1 = int(r_idx * row_h)
                y2 = int((r_idx + 1) * row_h)
                
                # Crop cell with padding
                pad_x = int(col_w * 0.25)
                pad_y = int(row_h * 0.25)
                
                if y2-pad_y > y1+pad_y and x2-pad_x > x1+pad_x:
                    cell = thresh[y1+pad_y:y2-pad_y, x1+pad_x:x2-pad_x]
                    nz = cv2.countNonZero(cell)
                    area = cell.size
                    if area > 0:
                        ratio = nz / area
                        if ratio > max_ratio:
                            max_ratio = ratio
                            best_r_idx = r_idx
            
            # Map Row to Letter (Row 2 -> A)
            letter_idx = best_r_idx - 2
            
            if max_ratio > 0.4 and 0 <= letter_idx < 26:
                result += alphabet[letter_idx]
            else:
                result += " "
                
        # Collapse multiple spaces
        return ' '.join(result.split())
        

    def process_answers_block(self, img, all_bubbles):
        # img is the warped/ROI image (grayscale)
        
        # Improve bubble detection by combining Adaptive and Otsu
        # Adaptive
        thresh_adapt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
        
        # Apply morphology to remove grid lines/noise (Critical for clean contours)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh_adapt = cv2.morphologyEx(thresh_adapt, cv2.MORPH_OPEN, kernel)
        
        cnts_adapt = self.get_contours(thresh_adapt)
        
        # Otsu
        # Blur first
        blur = cv2.GaussianBlur(img, (5,5), 0)
        _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Apply morphology to Otsu too
        thresh_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, kernel)
        
        cnts_otsu = self.get_contours(thresh_otsu)
        
        # Merge contours
        # We need to filter duplicates.
        # Simple way: just append and let the column grouper handle it?
        # Or filter by center distance.
        
        combined_bubbles = list(cnts_adapt)
        
        # Add Otsu bubbles if they don't overlap with existing
        existing_centers = []
        for c in cnts_adapt:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                existing_centers.append((cx, cy))
        
        for c in cnts_otsu:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Check distance
                is_new = True
                for ex_cx, ex_cy in existing_centers:
                    dist = np.sqrt((cx - ex_cx)**2 + (cy - ex_cy)**2)
                    if dist < 10: # Duplicate
                        is_new = False
                        break
                
                if is_new:
                    combined_bubbles.append(c)
                    
        all_bubbles = combined_bubbles

        # Filter bubbles by area (Relaxed)
        # Pre-filter noise (area < 30) to avoid skewing median
        valid_bubbles = [c for c in all_bubbles if cv2.contourArea(c) > 30]
        
        if valid_bubbles:
            areas = [cv2.contourArea(c) for c in valid_bubbles]
            median_area = np.median(areas)
            all_bubbles = valid_bubbles # Use pre-filtered list
        else:
            median_area = 50
            # If no bubbles > 30, maybe they are smaller?
            # Keep original list if valid_bubbles is empty, but median will be small.
            
        filtered_bubbles = []
        for c in all_bubbles:
             if cv2.boundingRect(c)[0] < 20: continue
             if 0.2 * median_area < cv2.contourArea(c) < 10.0 * median_area:
                 filtered_bubbles.append(c)
        
        all_bubbles = filtered_bubbles

        # 1. Cluster by X-coordinate to find columns
        all_bubbles, _ = self.sort_contours(all_bubbles, method="left-to-right")
        
        columns = []
        current_col = []
        last_x = -100
        
        for c in all_bubbles:
            x, y, w, h = cv2.boundingRect(c)
            if not current_col:
                current_col.append(c)
                last_x = x
            else:
                if abs(x - last_x) < 100: # Same column (increased threshold to group A,B,C,D)
                    current_col.append(c)
                else:
                    columns.append(current_col)
                    current_col = [c]
                    last_x = x
        if current_col:
            columns.append(current_col)
            
        print(f"DEBUG: Found {len(columns)} columns of questions", flush=True)
        
        # DEBUG: Visualize columns
        debug_grouping = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
        for i, col in enumerate(columns):
            # Random color
            color = np.random.randint(0, 255, (3,)).tolist()
            for c in col:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(debug_grouping, (x, y), (x + w, y + h), color, 2)
            # Draw bounding box of column
            xs = [cv2.boundingRect(c)[0] for c in col]
            ys = [cv2.boundingRect(c)[1] for c in col]
            ws = [cv2.boundingRect(c)[2] for c in col]
            hs = [cv2.boundingRect(c)[3] for c in col]
            if xs:
                min_x, min_y = min(xs), min(ys)
                max_x = max([x+w for x,w in zip(xs, ws)])
                max_y = max([y+h for y,h in zip(ys, hs)])
                cv2.rectangle(debug_grouping, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
                cv2.putText(debug_grouping, f"Col {i}", (min_x, min_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imwrite("debug_grouping.jpg", debug_grouping)
        
        # Filter columns: Must have at least 10 rows
        valid_columns = []
        for col in columns:
            if len(col) >= 10:
                valid_columns.append(col)
        
        print(f"DEBUG: Found {len(valid_columns)} valid columns (>= 10 bubbles)", flush=True)
        
        # We expect 4 columns. If more, take the 4 largest? Or just sorted by X?
        # Since we sorted bubbles L-R, columns are already sorted L-R.
        # If we have > 4, maybe some are noise.
        # Let's assume the 4 "real" columns are the ones with ~15 rows.
        # If we have exactly 4 valid, great.
        
        if len(valid_columns) > 4:
             # Sort by length (number of bubbles) and take top 4?
             # But we need them in Left-Right order.
             # So find top 4 by length, then sort those 4 by X.
             valid_columns.sort(key=len, reverse=True)
             valid_columns = valid_columns[:4]
             # Re-sort by X (using the first bubble's x)
             valid_columns.sort(key=lambda col: cv2.boundingRect(col[0])[0])
             
        answers = {}
        
        # ---------------------------------------------------------
        # Pass 1: Calculate Row Height for each column
        # ---------------------------------------------------------
        col_params = []
        all_row_heights = []
        
        for col_idx, col in enumerate(valid_columns):
            # Calculate Local Y-Grid for this column
            col_cy = [cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3]//2 for c in col]
            
            row_height = 0
            peaks = []
            
            if col_cy:
                hist_range = (0, img.shape[0])
                hist_bins = int(img.shape[0] / 5)
                hist, bin_edges = np.histogram(col_cy, bins=hist_bins, range=hist_range)
                
                threshold = max(hist) * 0.3
                for i in range(1, len(hist)-1):
                    if hist[i] > threshold and hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                        peaks.append((bin_edges[i] + bin_edges[i+1]) / 2)
                
                if len(peaks) >= 5:
                    peaks.sort()
                    diffs = np.diff(peaks)
                    row_height = np.median(diffs)
                    all_row_heights.append(row_height)
            
            col_params.append({
                'col': col,
                'peaks': peaks,
                'row_height': row_height
            })

        # Consensus Row Height
        if all_row_heights:
            global_row_height = np.median(all_row_heights)
            print(f"DEBUG: Global Consensus Row Height: {global_row_height:.1f} (from {all_row_heights})", flush=True)
        else:
            global_row_height = img.shape[0] / 15.0
            print(f"DEBUG: Global Row Height Fallback: {global_row_height:.1f}", flush=True)

        # ---------------------------------------------------------
        # Pass 2: Process Columns with Global Row Height
        # ---------------------------------------------------------
        for col_idx, params in enumerate(col_params):
            col = params['col']
            peaks = params['peaks']
            # Use global row height
            row_height = global_row_height
            
            # Determine Start Y using peaks and global row height
            start_y = 0
            if peaks:
                # Align first peak to a row index
                # We assume the first peak is Row 0, 1, or 2.
                # Find the phase that minimizes error for all peaks?
                # Simple approach: First peak is Row 0 if < 1.5 * row_height
                if peaks[0] > row_height * 1.5:
                     start_y = peaks[0] - row_height
                else:
                     start_y = peaks[0]
            else:
                start_y = row_height / 2.0
            
            print(f"DEBUG: Col {col_idx} Final Grid: RowH={row_height:.1f}, StartY={start_y:.1f}, Peaks={peaks}", flush=True)

            # Initialize 25 empty rows (to be safe)
            rows = [[] for _ in range(25)]
            unassigned = []
            
            # Assign bubbles to rows using Global Grid
            for c in col:
                x, y, w, h = cv2.boundingRect(c)
                cy = y + h // 2
                cx = x + w // 2
                area = cv2.contourArea(c)
                
                # Calculate row index
                # Row 0 is at StartY
                row_idx = int((cy - start_y + row_height/2) / row_height)
                
                if 0 <= row_idx < 25:
                    rows[row_idx].append(c)
                else:
                    unassigned.append((cx, cy, area, row_idx))
                
            print(f"DEBUG: Column {col_idx} has {len(rows)} rows (Global Grid)", flush=True)
            if unassigned:
                print(f"DEBUG: Col {col_idx} Unassigned Bubbles: {unassigned}", flush=True)
            
            for r_i, r_bubbles in enumerate(rows):
                if r_bubbles:
                    b_info = [(cv2.boundingRect(b)[0] + cv2.boundingRect(b)[2]//2, cv2.boundingRect(b)[1] + cv2.boundingRect(b)[3]//2, cv2.contourArea(b)) for b in r_bubbles]
                    print(f"DEBUG: Col {col_idx} Row {r_i} Bubbles: {b_info}", flush=True)

            # Calculate column bounds using Histogram Peaks (Robust to heavy noise)
            all_cx = [cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]//2 for c in col]
            if not all_cx: continue
            
            # Create histogram
            hist_range = (0, img.shape[1])
            hist_bins = int(img.shape[1] / 5)
            hist, bin_edges = np.histogram(all_cx, bins=hist_bins, range=hist_range)
            
            # Find all peaks
            peaks = []
            peak_heights = []
            threshold = max(hist) * 0.2 # Lower threshold to catch all
            for i in range(1, len(hist)-1):
                if hist[i] > threshold and hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                    peaks.append((bin_edges[i] + bin_edges[i+1]) / 2)
                    peak_heights.append(hist[i])
            
            print(f"DEBUG: Col {col_idx} ALL Peaks: {list(zip(peaks, peak_heights))}", flush=True)
            
            # Adaptive Peak Mapping
            # 1. Identify Q (First strong peak)
            # 2. Walk Q + i * gap to find A, B, C, D
            
            if not peaks:
                 col_min_cx = np.percentile(all_cx, 5)
                 col_max_cx = np.percentile(all_cx, 95)
                 width_span = col_max_cx - col_min_cx
                 slot_width = width_span / 3.0
                 col_peaks = [col_min_cx, col_min_cx + slot_width, col_min_cx + 2*slot_width, col_max_cx]
            else:
                # Assume first peak is Q
                q_peak = peaks[0]
                
                # Estimate gap from first few peaks
                if len(peaks) > 1:
                    gaps = np.diff(peaks)
                    # Filter large gaps (missing columns)
                    valid_gaps = [g for g in gaps if 15 < g < 35]
                    if valid_gaps:
                        avg_gap = np.mean(valid_gaps)
                    else:
                        avg_gap = 22.0 # Default
                else:
                    avg_gap = 22.0
                
                # Predict A, B, C, D
                col_peaks = [q_peak]
                current_target = q_peak + avg_gap # Start looking for A
                
                for i in range(4): # A, B, C, D
                    # Find nearest actual peak to current_target
                    best_p = current_target
                    min_dist = 10.0 # Tolerance
                    found = False
                    
                    for p in peaks:
                        dist = abs(p - current_target)
                        if dist < min_dist:
                            min_dist = dist
                            best_p = p
                            found = True
                    
                    col_peaks.append(best_p)
                    
                    # Update target for next slot
                    # If we found a peak, step from there. If not, step from calculated.
                    if found:
                        current_target = best_p + avg_gap
                    else:
                        current_target = current_target + avg_gap
                
                col_min_cx = col_peaks[1] # A
                col_max_cx = col_peaks[-1] # D
                slot_width = avg_gap
                
                print(f"DEBUG: Col {col_idx} Adaptive: Q={q_peak:.1f}, Gap={avg_gap:.1f} -> A={col_peaks[1]:.1f}, B={col_peaks[2]:.1f}, C={col_peaks[3]:.1f}, D={col_peaks[4]:.1f}", flush=True)

            

            # ---------------------------------------------------------
            # Pass 3: Grid Sampling (Robust to missing bubbles)
            # ---------------------------------------------------------
            
            # Prepare thresholded image for sampling (Once per column)
            if len(img.shape) == 3:
                 gray_roi = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                 gray_roi = img
            
            blur_roi = cv2.GaussianBlur(gray_roi, (5,5), 0)
            _, thresh_roi = cv2.threshold(blur_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Apply morphology to remove grid lines/noise from fill check
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh_roi = cv2.morphologyEx(thresh_roi, cv2.MORPH_OPEN, kernel)

            # Iterate through fixed rows 1 to 15 (Skipping Row 0 as Header/Empty)
            for r_idx in range(1, 16):
                # Calculate row Y center
                # Row 0 is at StartY
                row_cy = start_y + (r_idx * row_height)
                
                q_num = (col_idx * 15) + r_idx # 1-based Q number, matching r_idx
                
                # Initialize ratios for Q, A, B, C, D
                # 0=Q, 1=A, 2=B, 3=C, 4=D
                final_ratios = [0.0] * 5
                
                # Sample each slot
                for slot_idx, peak_x in enumerate(col_peaks):
                    if slot_idx >= 5: break # Safety
                    
                    # Define crop box
                    box_size = 24 # Slightly larger to catch bubble
                    x1 = int(peak_x - box_size/2)
                    x2 = int(peak_x + box_size/2)
                    y1 = int(row_cy - box_size/2)
                    y2 = int(row_cy + box_size/2)
                    
                    # Clamp
                    if x1 < 0: x1 = 0
                    if y1 < 0: y1 = 0
                    if x2 > img.shape[1]: x2 = img.shape[1]
                    if y2 > img.shape[0]: y2 = img.shape[0]
                    
                    if x2 > x1 and y2 > y1:
                        slot_crop = thresh_roi[y1:y2, x1:x2]
                        nz = cv2.countNonZero(slot_crop)
                        area = slot_crop.size
                        if area > 0:
                            ratio = nz / area
                            final_ratios[slot_idx] = ratio
                
                # Determine answer
                options = ['A', 'B', 'C', 'D']
                # Ignore Q (index 0)
                answer_ratios = final_ratios[1:]
                max_idx = np.argmax(answer_ratios)
                max_val = answer_ratios[max_idx]
                
                if max_val > 0.45: # Threshold
                    answers[f"Q{q_num}"] = options[max_idx] # Return string, not list
                else:
                    answers[f"Q{q_num}"] = '' # Return empty string
        
        return answers
