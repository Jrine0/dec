import cv2
import numpy as np

class OMREngine:
    def __init__(self):
        pass

    def apply_gamma(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

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
        return self.extract_fields(gray)

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
            
            if max_ratio > 0.15 and 0 <= val_idx <= 9:
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
        
        if max_ratio > 0.15 and 0 <= opt_idx < 4:
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
            
            if max_ratio > 0.3 and 0 <= letter_idx < 26:
                result += alphabet[letter_idx]
            else:
                result += " "
        
        return ' '.join(result.split())
                
    def process_answers_block(self, img, all_bubbles):
        # img is the warped answer block (grayscale or BGR)
        # all_bubbles is a list of contours found in the block (may be incomplete)

        # Implement robust Grid Sampling
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Apply Gamma and CLAHE for better signal
        gray = self.apply_gamma(gray, gamma=0.5)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Threshold for sampling
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh_roi = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def process_answers_block(self, img, all_bubbles):
        # img is the warped answer block (grayscale or BGR)
        # all_bubbles is a list of contours found in the block (may be incomplete)

        # Implement robust Grid Sampling
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Apply Gamma and CLAHE for better signal
        gray = self.apply_gamma(gray, gamma=0.5)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Threshold for sampling
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh_roi = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # The image has 4 "Panels" of questions.
        # Panel 1: Q1-15. Panel 2: Q16-30. Panel 3: Q31-45. Panel 4: Q46-60.
        
        h, w = img.shape[:2]
        col_width = w // 4
        
        answers = {}
        
        for col_idx in range(4):
            # Define strip
            x_start = col_idx * col_width
            x_end = (col_idx + 1) * col_width
            strip_thresh = thresh_roi[:, x_start:x_end]
            
            # Use fixed positions relative to strip width for stability
            # Strip width is ~502px.
            # Manual Col Calibration: 41%, 52%, 63%, 74% (Base)
            base_percentages = [0.41, 0.52, 0.63, 0.74]
            
            # Apply per-panel shifts as requested
            # Panel 0: Base
            # Panel 1: "slightly towards right" -> Increased to +2.5%
            # Panel 2: "more towards right" -> Increased to +4.5%
            # Panel 3: "much more towards right" -> Increased further to +7.5%
            
            shift = 0.0
            if col_idx == 1:
                shift = 0.025
            elif col_idx == 2:
                shift = 0.045
            elif col_idx == 3:
                shift = 0.075
                
            col_percentages = [p + shift for p in base_percentages]
            cols_x = [int(col_width * p) for p in col_percentages]
            
            # Print for debug
            if col_idx == 0:
                print(f"DEBUG: Using Final Cols X: {cols_x} (Strip W: {col_width})", flush=True)

            # Manual Row Calibration Variables
            gap_size_ratio = 1.15 # Gap size relative to a row height (increased to 115%)
            
            # Total units = 15 rows + 2 gaps
            total_units = 15.0 + (2.0 * gap_size_ratio)
            # Set step size to 0.765 as requested
            step = (h / total_units) * 0.765
            
            rows_y = []
            current_y = step * 3.7 # Start at 3.7x as requested
            
            for i in range(15):
                rows_y.append(int(current_y))
                
                # Increment for next row
                if i == 4 or i == 9: # After 5th and 10th row (index 4 and 9)
                    current_y += step * (1.0 + gap_size_ratio)
                else:
                    current_y += step

            if col_idx == 0:
                print(f"DEBUG: Manual Row Grid: Step={step:.1f}, GapRatio={gap_size_ratio}", flush=True)

            # Iterate 15 rows
            for r in range(15):
                q_num = col_idx * 15 + (r + 1)
                
                # Row Y center
                cy = rows_y[r]
                
                # Sample 4 options
                ratios = []
                for cx in cols_x:
                    # Crop box
                    box = 18
                    x1 = max(0, cx - box//2)
                    x2 = min(col_width, cx + box//2)
                    y1 = max(0, cy - box//2)
                    y2 = min(h, cy + box//2)
                    
                    crop = strip_thresh[y1:y2, x1:x2]
                    if crop.size > 0:
                        ratio = cv2.countNonZero(crop) / crop.size
                        ratios.append(ratio)
                    else:
                        ratios.append(0.0)
                
                # Debug Visualization: Draw grid points on debug image
                # Check if debug image exists for this col, else create
                if f"debug_panel_{col_idx}" not in locals():
                     # Create color version of strip for drawing
                     debug_panel = img[:, x_start:x_end].copy()
                     if len(debug_panel.shape) == 2:
                         debug_panel = cv2.cvtColor(debug_panel, cv2.COLOR_GRAY2BGR)
                else:
                     # Reuse existing (would need dict, omitting reuse for simplicity, overwriting)
                     debug_panel = img[:, x_start:x_end].copy()
                     if len(debug_panel.shape) == 2:
                         debug_panel = cv2.cvtColor(debug_panel, cv2.COLOR_GRAY2BGR)

                # Draw Vertical Lines for this strip
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)] # Blue, Green, Red, Yellow
                for i, cx in enumerate(cols_x):
                    cv2.line(debug_panel, (cx, 0), (cx, h), colors[i % 4], 2)
                
                # Draw Horizontal Rows (Magenta)
                for ry in rows_y:
                    cv2.line(debug_panel, (0, ry), (col_width, ry), (255, 0, 255), 1)

                # Save this strip as debug image
                cv2.imwrite(f"debug_panel_{col_idx}.jpg", debug_panel)


                # Determine answer
                options_labels = ['A', 'B', 'C', 'D']
                if not ratios:
                    answers[f"Q{q_num}"] = ''
                    continue
                        
                max_idx = np.argmax(ratios)
                max_val = ratios[max_idx]
                
                if max_val > 0.02: 
                    answers[f"Q{q_num}"] = options_labels[max_idx]
                else:
                    answers[f"Q{q_num}"] = ''
                    
        return answers
