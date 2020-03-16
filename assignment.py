import cv2 as cv
import numpy as np
import time
import os

# The program will look for an Orings directory in the same location as this file by default
# The program will also read in all .jpg files in the Orings directory
# These default parameters can be changed by altering the variables below

location = './Orings' # Specify the folder that contains all the images, the program will search through it
file_ext = '.jpg' # Specify the file type for the program to read in

def read_images(loc, ext): # Imports and returns a list of all the files with the specified extension (ext) from the specified folder (loc)
    ret = [] # Initialisation of a list which will hold all the file data that is being imported
    for file in os.listdir(loc): # Loop through every file in the specified folder
        if file.endswith(ext): # Check if the file is of the requried extension
            ret.append([os.path.join(loc + '/', file), cv.imread(os.path.join(loc + '/', file), 0)]) # Add a list which holds the full file location as well as the image data
    return ret

def get_hist(img): # Takes in an image and returns a histogram of pixel values found in the image
    hist = np.zeros(256) # Initialise an empty numpy array (full of 0s)
    for i in range(0, img.shape[0]): # Loop through the Y axis of the image
        for j in range(0, img.shape[1]): # Loop through the X axis of the image
            hist[img[i, j]] += 1 # Increment the counter for the pixel value found at the current y,x position
    return hist

def find_peaks(hist): # Takes in a histogram and returns a list of 2 peak values (peak 1: highest counter of pixels: peak 2: highest counter of pixels atleast 50 values away from the first peak)
    peaks = [np.where(hist == max(hist))[0][0]] # Create a list which contains the most occuring pixel value (peak 1)
    p1 = peaks[0] - 50 # Set the minimum distance that peak 2 has to be away from peak 1
    p2 = peaks[0] + 50 # ^
    new_arr = [hist[i] for i in range(len(hist)) if i < p1 or i > p2] # Create a temporary list which holds all the counts for pixels which are more than the minimum distance away from peak 1
    peaks.append(new_arr.index(max(new_arr))) # Find the most occuring pixel value in the temporary list (will be peak 2)
    peaks.sort() # Sort the peaks list so the most occuring peak 1 is at position 0, and peak 2 is at position 1
    return peaks

def get_threshold_val(img): # Takes in an image and calculates and returns a suitable T value for the given image
    hist = get_hist(img[1]) # Call a method to retrieve the histogram of pixel values for the given image
    peaks = find_peaks(hist) # Retrieve the 2 peaks of pixel values in the histogram
    t = peaks[0] + int((peaks[1] - peaks[0]) / 2) # Retrieve the 2 peak values from the histogram for this image, and store the value found in the middle between the peaks
    return t

def threshold(img): # Performs the thresholding operation and returns a new thresholded image
    thr = img[1].copy() # Create a copy of the image, will be used to return the thresholded values
    t = get_threshold_val(img) # Call a method which will calculate and return the T value for the current image
    for i in range(0, thr.shape[0]): # Loop through the y axis
        for j in range(0, thr.shape[1]): # Loop through the x axis
            if thr[i, j] > t: # If the pixel value at the current location higher than the T (threshold) value
                thr[i, j] = 255 # Set it to 255
            else:
                thr[i, j] = 0 # Set it to 0
    return thr

def dilation(img, struct): # Performs the dilation operation on the given image using the morphological structure supplied, and then returns the modified image
    ret = img.copy() # Create a copy of the original image
    for i in range(0, img.shape[0]): # Loop through the y axis
        for j in range(0, img.shape[1]): # Loop through the x axis
            # Check if the current pixel is of a background colour and the index positions required for the morphological structure are within the bounds of the image
            if img[i, j] == 255 and i - 1 >= 0 and j - 1 >= 0 and i + 1 < img.shape[0] and j + 1 < img.shape[1]:
                for y in range(-1, len(struct) - 1): # Loop through the y axis of the morphological structure (centered around the current pixel at 0,0)
                    for x in range(-1, len(struct) - 1): # Loop through the x axis of the morphological structure (centered around the current pixel at 0,0)
                        new_i = i + y # Get the offset i postion that needs to be checked
                        new_j = j + x # Get the offset j postion that needs to be checked
                        # Check if the location being checked is not the current pixel, and that it matches the designation within the morphological structure (1 in struct = 0 in img)
                        if [new_i, new_j] != [i, j] and struct[y + 1][x + 1] == 1 and img[new_i, new_j] == 0:
                            ret[i, j] = 0 # Set the current pixel to foreground
    return ret

def erosion(img, struct): # Performs the erosion operation on the given image using the morphological structure supplied, and then returns the modified image
    ret = img.copy()  # Create a copy of the original image
    for i in range(0, img.shape[0]): # Loop through the y axis
        for j in range(0, img.shape[1]): # Loop through the x axis
            # Check if the current pixel is of a foreground colour and the index positions required for the morphological structure are within the bounds of the image
            if img[i, j] == 0 and i - 1 >= 0 and j - 1 >= 0 and i + 1 < img.shape[0] and j + 1 < img.shape[1]:
                for y in range(-1, 2): # Loop through the y axis of the morphological structure (centered around the current pixel at 0,0)
                    for x in range(-1, 2): # Loop through the x axis of the morphological structure (centered around the current pixel at 0,0)
                        new_i = i + y # Get the offset i postion that needs to be checked
                        new_j = j + x # Get the offset j postion that needs to be checked
                        # Check if the location being checked is not the current pixel, and that it matches the designation within the morphological structure (1 in struct = 255 in img)
                        if [new_i, new_j] != [i, j] and struct[y + 1][x + 1] == 1 and img[new_i, new_j] == 255:
                            ret[i, j] = 255 # Set the current pixel to background
    return ret

def closing(img, struct): # Takes in an image and a morphological structure, and performs the closing operation (dilation followed by erosion), then returns the new image
    ret = dilation(img, struct) # Pass the image and the morphological structure to a method to perform dilation
    ret = erosion(ret, struct) # Pass the now dilated image and the morphological structure to a method to perform erosion
    return ret

def label_components(img): # Takes in an image and performs connected component labelling, returning the list of labels for pixels at every index position on the image
    labels = [[0 for j in range(0, img.shape[1])] for i in range(0, img.shape[0])] # Create a 2d list the size of the image, filled with 0s (0 = unlabelled)
    cur_lab = 1 # Set the current label tag to 1
    for i in range(0, img.shape[0]): # Loop through the y axis of the image
        for j in range(0, img.shape[1]): # Loop through the x axis of the image
            if img[i, j] == 0 and labels[i][j] == 0: # If the current pixel has a foreground colour and is unlabelled (value 0 in the labels list)
                labels[i][j] = cur_lab # Label the current pixel (update its i, j position in the labels list with the current label tag)
                q = [] # Initialise an empty "queue" structure (in my case it is a list taht will be used like a queue)
                q.append([i, j]) # Add the current pixels coordinates to the queue

                while len(q) > 0: # Loop while the lenght of the queue is more than 0 (there are neighbouring coordinates that still need to be processed)
                    item = q.pop(0) # "Pop" the item in the queue at index position 0 (equivalent to the dequeue operation if the data structure was an actual queue)
                    if img[item[0] - 1, item[1]] == 0 and labels[item[0] - 1][item[1]] == 0: # Check if the neighbour above is a foreground pixel and is unlabelled
                        q.append([item[0] - 1, item[1]]) # Add its coordinates to the queue
                        labels[item[0] - 1][item[1]] = cur_lab # Label it with the current component label
                    if img[item[0] + 1, item[1]] == 0 and labels[item[0] + 1][item[1]] == 0: # Check if the neighbour below is a foreground pixel and is unlabelled
                        q.append([item[0] + 1, item[1]]) # Add its coordinates to the queue
                        labels[item[0] + 1][item[1]] = cur_lab # Label it with the current component label
                    if img[item[0], item[1] - 1] == 0 and labels[item[0]][item[1] - 1] == 0: # Check if the neighbour to the left is a foreground pixel and is unlabelled
                        q.append([item[0], item[1] - 1]) # Add its coordinates to the queue
                        labels[item[0]][item[1] - 1] = cur_lab # Label it with the current component label
                    if img[item[0], item[1] + 1] == 0 and labels[item[0]][item[1] + 1] == 0:  # Check if the neighbour to the right is a foreground pixel and is unlabelled
                        q.append([item[0], item[1] + 1]) # Add its coordinates to the queue
                        labels[item[0]][item[1] + 1] = cur_lab # Label it with the current component label
                cur_lab += 1 # We will only reach this point when every connected pixel has been labelled, then increment the label counter
    return labels

def calc_component_area(labels, label):# Takes in a list of labels, and a label for which to calculate the area, and returns the number of pixels (area) with this label
    area = 0 # Counter set to 0
    for y in range(0, len(labels)): # Loop through the y axis of the labels list
        for x in range(0, len(labels[0])): # Loop through the x axis of the labels list
            if labels[y][x] == label: # If the label at this index position matches the one we are trying to count
                area += 1 # Increment the counter
    return area

def paint_labeled_components(img, labels): # Takes in an image and a list of labels, and returns the altered image with labels painted on it
    ret = img.copy() # Create a copy of the image
    for i in range(0, img.shape[0]): # Loop through the y axis
        for j in range(0, img.shape[1]): # Loop through the x axis
                ret[i, j] = labels[i][j] * 255 # Updates the labelled pixel by multiplying it by 255 which sets the foreground colours to white, and background to black (multiplication by 0)
    return ret

def remove_smallest_areas(labels): # Takes in a list of labels, and removes the smallest areas (will only return the labels for the oring)
    u = np.unique(labels) # Extract the list of unique component labels
    if len(u) > 2: # If more than 2 component labels were found (0=unlabelled, 1=oring, 2=something else, etc...)
        u = u[1:] # Remove the unlabelled component area
        areas = []
        for i in range(len(u)): # Loop through all the other areas
            areas.append(calc_component_area(labels, u[i])) # Call a method to calculate the area of the currently selected label/component
        # The next step is to actually remove the smallest area. For this assignment the most we had was 1 extra area that needed to be removed, so this was only performed once
        # The steps below could be put inside a loop until only 1 area is left, which would keep deleting the smallest areas 1 at a time
        smallest_area = u[areas.index(min(areas))] # Retrieve the smallest area label (the label that occurs the least)
        new_labels = []
        for label_set in labels: # Loop through the y axis in the labels list
            new_set = []
            for item in label_set: # Loop through the x axis in the labels list
                if item == smallest_area: # If the label in the current index position mathches the smallest area label
                    new_set.append(0) # Set it to a background colour (get rid of it)
                else:
                    new_set.append(item) # Otherwise don't change it
            new_labels.append(new_set)
        return new_labels # Return the newly constructed labels list
    return labels

def get_centroid(img): # Takes in a list of labels and returns the centroid coordinates by calculating the average i and the averagre j index positions for the main component (the oring)
    pix = 0
    total_j = 0
    total_i = 0
    for i in range(0, len(img)): # Loop through the y axis of the image
        for j in range(0, len(img[0])): # Loop through the x axis of the image
            if img[i][j] == 1:
                pix += 1
                total_j += j
                total_i += i
    return [int(total_i / pix), int(total_j / pix)] # sum of foreground j positions / count of foreground pixels = average j (same for i)

def get_bounding_box(img): # Takes in a list of labels (so only the main component, the oring, gets counted), and calculates the coordinates for the top left corner and bottom right corner of the bounding box
    bound_coords = [0 for i in range(4)] # List used to store the coordinates of interest in the following order: [Min i, Max i, Min j, Max j]
    first_found = False # Used to overcome the problem of min/min coordinate initialisation before the first foreground pixel is found
    for i in range(0, len(img)): # Loop through the y axis of the image
        for j in range(0, len(img[0])): # loop through the x axis of the image
            if img[i][j] == 1: # If the current pixel is labelled 1 (is part of the oring)
                if first_found == True: # Check if this is the first relevant pixel to be processed
                    if i < bound_coords[0]: # If the current i coordinate lower than the lowest i on record
                        bound_coords[0] = i
                    elif i > bound_coords[1]: # If the current i coordinate higher than the highest i on record
                        bound_coords[1] = i
                    if j < bound_coords[2]: # If the current j coordinate lower than the lowest j on record
                        bound_coords[2] = j
                    elif j > bound_coords[3]: # If the current j coordinate higher than the highest j on record
                        bound_coords[3] = j
                else: # If it is the first pixel found, set both the min and the max values for i and j with the current coordinates
                    first_found = True
                    bound_coords[0] = i # Min i
                    bound_coords[1] = i # Max i
                    bound_coords[2] = j # Min j
                    bound_coords[3] = j # Max j
    return bound_coords

def paint_bounding_box(img, bounding_box, result): # Takes in an image, bounding box coordinates, and a result boolean and returns the iamge with a bounding box around the O ring
    pix = (0,0,255) # Set the pixel value to red (default, assuming its faulty)
    if result == True: # If the result is pass (True)
        pix = colour = (0,255,0) # Set the pixel value to green

    # bounding_box = [Min i, Max i, Min j, Max j]
    img[(bounding_box[0] - 1):(bounding_box[1] + 2), bounding_box[2] - 1] = pix # Draws the line on the left side of the O ring
    img[(bounding_box[0] - 1):(bounding_box[1] + 2), bounding_box[3] + 1] = pix # Draws the line on the right side of the O ring
    img[bounding_box[0] - 1, (bounding_box[2] - 1):(bounding_box[3] + 2)] = pix # Draws the line on the top side of the O ring
    img[bounding_box[1] + 1, (bounding_box[2] - 1):(bounding_box[3] + 2)] = pix # Draws the line on the bottom side of the O ring

    return img

def get_radius(img, centroid): # Takes in an image and a centroid coordinate, and returns the average inner and average outer edge radius values in a list (inner = position 0, outer = position 1)
    radius = [0, 0] # Holds the total radius values from the 4 sides being checked
    found = [False, False] # Used for differentiating between the inner and otuer edge radius values (and when to stop looking for them)

    # centroid -> up
    for i in range(centroid[0] - 1, 0, -1): # Loops from the centroid y position to the top of the image
        if not found[0]: # If the smaller radius (centroid up to the inner edge of the oring) has not been found
            radius[0] += 1 # Increment the radius counter by a pixel for the smaller radius
        if not found[1]: # If the bigger radius (centroid up to the outer edge of the oring) has not been found
            radius[1] += 1 # Increment the radius counter by a pixel for the bigger radius
        if img[i, centroid[1]] == 255 and img[i + 1, centroid[1]] == 0: # If the current pixel is of a foregorund colour and the previous pixel (below) was background, then the inner edge has been found
            found[0] = True # Mark the inner edge as found
        if img[i, centroid[1]] == 0 and img[i + 1, centroid[1]] == 255: # If the current pixel is of a background colour and the previous pixel (below) was foreground, then the outer edge has been reached
            found[1] = True # Mark the outer edge as found
        if found[0] == True and found[1] == True: # If both the inner and outer edges have been found, reset the found booleans so they can be used for the radius going the other way
            found = [False, False]
            break # Break out of the loop, as both edges have been found
    
    # The 3 loops below this will follow the exact same steps as outlined above, but going in different directions: centroid to bottom of the image, centroid to left edge, and centroid to right edge

    # centroid -> down
    for i in range(centroid[0], 220): # Loops from the centroid y position to the bottom of the image
        if not found[0]:
            radius[0] += 1
        if not found[1]:
            radius[1] += 1
        if img[i, centroid[1]] == 255 and img[i - 1, centroid[1]] == 0:
            found[0] = True
        if img[i, centroid[1]] == 0 and img[i - 1, centroid[1]] == 255:
            found[1] = True
        if found[0] == True and found[1] == True:
            found = [False, False]
            break

    # centroid -> left
    for j in range(centroid[1], 0, -1): # Loops from the centroid x position to the left edge of the image
        if not found[0]:
            radius[0] += 1
        if not found[1]:
            radius[1] += 1
        if img[centroid[0], j] == 255 and img[centroid[0], j + 1] == 0:
            found[0] = True
        if img[centroid[0], j] == 0 and img[centroid[0], j + 1] == 255:
            found[1] = True
        if found[0] == True and found[1] == True:
            found = [False, False]
            break

    # centroid -> right
    for j in range(centroid[1], 220): # Loops from the centroid x position to the right edge of the image
        if not found[0]:
            radius[0] += 1
        if not found[1]:
            radius[1] += 1
        if img[centroid[0], j] == 255 and img[centroid[0], j - 1] == 0:
            found[0] = True
        if img[centroid[0], j] == 0 and img[centroid[0], j - 1] == 255:
            found[1] = True
        if found[0] == True and found[1] == True:
            found = [False, False]
            break

    radius = [num / 4 for num in radius] # Calculate the average inner radius and average outer radius by dividing both values by 4 (4 radius values were added to the total)
    radius = [num - 1 for num in radius] # Adjust the radius by reducing both the inner and otuer radius by 1, to deal with round of numbers
    return radius

def is_in_circle(x, y, a, b, r): # Takes in: x, y (pixel coordinates to check) and a,b (center point of the circle), and the radius, and returns true if the x,y coordinates are inside the circle
    # (x - a)^2 + (y - b)^2 <= r^2, where ^ is to the power of
    return ((x - a)**2) + ((y - b)**2) <= r**2

def get_result(img, centroid, radius): # Takes in an image, centroid coordinates, and radius values and returns whether the oring passes or is faulty (fails)
    allowed_diff = 2 # The allowed distance between a foreground pixel and the constructed ring for it to still be considered as a pass
    mask_pix = 255 # The pixel value assigned to the "mask" or shape of the constructed ring (used for processing)
    mask = img.copy() # Create a copy of the image, which will be updated and store only the newly constructed shape of the ring
    for i in range(0, img.shape[0]): # Loop through the y axis of the image
        for j in range(0, img.shape[1]): # Loop through the x axis of the image
            # Call a method which calculates if any give pixel location (i, j) is within the expected ring (in the bigger circle with bigger radius, but not in the smaller circle with smaller radius)
            if is_in_circle(j, i, centroid[1], centroid[0], radius[1]) and not is_in_circle(j, i, centroid[1], centroid[0], radius[0]):
                mask[i, j] = mask_pix # If the pixel location is within the constructed ring, mark is with the mask pixel colour
            else:
                mask[i, j] = 0 # Otherwise remove it from the image and set it to a background colour
    faulty_loc = [] # Used to keep track of all the faulty pixel coordinates 
    for i in range(0, img.shape[0]): # Loop through the y axis of the image
        for j in range(0, img.shape[1]): # Loop through the x axis of the image
            # If a background pixel is located where the expected ring should be, or if a foreground pixel is found outside the expected ring
            if img[i, j] == 0 and mask[i, j] == mask_pix or img[i, j] == 255 and mask[i, j] == 0:
                is_faulty = True # Mark the current pixel as faulty
                # The loops below go frmo -2 x and y to +2 x and y from the current pixel location
                for fi in range(-allowed_diff, allowed_diff + 1): # Loop through the y axis of "allowed_diff" number of neighbours on all sides of the current pixel (default value 2)
                    for fj in range(-allowed_diff, allowed_diff + 1): # Loop through the x axis of "allowed_diff" number of neighbours on all sides of the current pixel (default value 2)
                        if [fi, fj] != [0, 0]: # Do not perform any processing if the neighbour that we are checking is actually the current faulty pixel
                            new_i = i + fi # Store the offset i (y) position
                            new_j = j + fj # Store the offset j (x) position
                            if img[new_i, new_j] == 255: # If it is a foreground pixel (is on the main body of the O ring)
                                is_faulty = False # Set the current pixel to not faulty
                if img[i + 3, j + 3] == 255: # Check the south-east corner too (fixes the one O ring that does not get identified correctly due to being skewed slightly)
                    is_faulty = False
                if is_faulty: # If the pixel is still marked as faulty
                    faulty_loc.append([i, j]) # Add its coordinates to the faulty pixel location list (used to classify if the ring is faulty( if length is greater than 0), and to colour in the faulty area
    
    if len(faulty_loc) > 0: # If there are mroe than 0 faulty pixel locations store
        return [False, faulty_loc, mask] # Return false, as the O ring is faulty
    return [True, faulty_loc, mask] # Return true, to show that the ring passes

def paint_faulty_locations(img, expected_ring): # Takes in an image and a mask
    colour = np.array([0, 0, 255]) # Define the colour to paint: red
    for i in range(0, img.shape[0]): # Loop through the y axis of the image
        for j in range(0, img.shape[1]): # Loop through the x axis of the image
            # If a foreground pixel is detected outside the expected ring, or if a background pixel is detected inside the expected ring
            if np.array_equal(expected_ring[i,j], np.array([255, 255, 255])) and np.array_equal(img[i, j], np.array([0, 0, 0])) or np.array_equal(expected_ring[i,j], np.array([0, 0, 0])) and np.array_equal(img[i, j], np.array([255, 255, 255])):
                img[i, j] = colour # Paint the pixel red
    return img

# Takes in the title of the file, the actual image, labels, centroid coordinates, boinding_box coordinates, the processing time, and the totals (which hold the current count of failed and passed origns) and outputs them as an image to the screen
def perform_detection(title, img, labels, centroid, bounding_box, start, totals):
    updated_img = img.copy() # Create a copy of the original image (to "paint" on a copy rather than altering the original)
    updated_img = paint_labeled_components(updated_img, labels) # Paints the labels of the components onto the image (inverts the foreground and background colours in the process)
    radius = get_radius(updated_img, centroid) # Calculates the 2 radius values (inner edge and outer edge of the oring)
    
    result = get_result(updated_img, centroid, radius) # Processes the image and 
    end = time.time() # Get the time when all the processing is done, used to print the total processing time on to the image

    totals[2] += (end - start) # Add the processing time for the current ring to the total processing time count

    updated_img = cv.cvtColor(updated_img, cv.COLOR_GRAY2RGB) # Turns the image into BGR scale, allows use of colour rather than greyscale when image is displayed
    result[2] = cv.cvtColor(result[2], cv.COLOR_GRAY2RGB)
    
    if result[0] == False: # Check if result for this O ring has come back as False (faulty) or True (passing)
        res = 'FAIL' # Set the resulting text
        colour = (0, 0, 255) # Set the colour to red
        totals[1] += 1 # Increment the False total counter
        updated_img = paint_faulty_locations(updated_img, result[2]) # Paint in all the areas that do not match the expected circle
    else:
        res = 'PASS' # Set the resulting text
        colour = (0,255,0) # Set the colour to green
        totals[0] += 1 # Increment the True total counter

    updated_img = paint_bounding_box(updated_img, bounding_box, result[0]) # Paint the bounding box onto the image
    updated_img = cv.putText(updated_img, "Processing Time (s): " + str(end - start), (5, 10), cv.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv.LINE_AA) # Paint the processing time text onto the image

    updated_img = cv.putText(updated_img, res, (centroid[1] - 15, centroid[0] + 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv.LINE_AA) # Print the resulting text (FAIL or PASS) in the center of the O ring
    updated_img = cv.putText(updated_img, "Totals: ", (5, 215), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv.LINE_AA) # Print the title for extra statistics onto the image
    updated_img = cv.putText(updated_img, "Pass {}".format(totals[0]), (40, 215), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv.LINE_AA) # Print the total number of Passing O rings processed onto the image
    updated_img = cv.putText(updated_img, "Fail {}".format(totals[1]), (80, 215), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv.LINE_AA) # Print the total number of Failing O rings processed onto the image
    updated_img = cv.putText(updated_img, title.split('.')[0], (167, 215), cv.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1, cv.LINE_AA) # Print the file name (O ring number) onto the image

    cv.imshow(title, updated_img) # Show the image to the user
    cv.waitKey(0) # Wait for key input from the user before proceeding
    cv.destroyAllWindows() # Destroy all the windows open before moving on
    return totals

images = read_images(location, file_ext) # Call the method to read in all the images, store them in a list

morph_struct = [ # Morphological structure, used for erosion and dilation
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]

totals = [0, 0, 0] # Totals list which keeps track of the total number of Orings that have passed or failed

for img in images: # Loop through the list of images, and process them one by one
    start = time.time() # Get the start time (before any processing is done)
    img_thr = threshold(img) # Call the method to threshold and return the thresolded image
    img_thr = closing(img_thr, morph_struct) # Call the method to perform closing (dilation, followed by erosion)
    img_labels = label_components(img_thr) # Call the method to perform Connected Component Labelling, returns a list of labels
    img_labels = remove_smallest_areas(img_labels) # Use the component labels to remove the components with the smallest areas
    centroid = get_centroid(img_labels) # Calculate the centroid for the image
    bounding_box = get_bounding_box(img_labels) # Calculate the bounding box coordinates for the image (using the labels list to only count the main Oring area)
    totals = perform_detection(img[0].split('/')[-1], img_thr, img_labels, centroid, bounding_box, start, totals) # Call the method to display the image, and return the new totals list

# Print the processing statistics to the console
print("=== Processing Statistics ===")
print("Processing Time:\t{} seconds".format(totals[2]))
print("Total Orings:\t\t{}".format(len(images)))
print("Avg. Processing Time:\t{}".format(totals[2] / len(images)))
print("Passing Orings:\t\t{}".format(totals[0]))
print("Failing Orings:\t\t{}".format(totals[1]))