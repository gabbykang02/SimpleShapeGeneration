import numpy as np
import random
import cv2
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.affinity import rotate
from shapely.geometry import Point as ShapelyPoint
import time  # For tracking time

def gen_triangles(IMG,  side_length, NUM_SHAPES, timeout = 5, enable_overlap = False, random_sizes=False):
    tries = 0
    outputs = gen_triangles2(IMG, side_length, NUM_SHAPES, timeout = timeout, enable_overlap = enable_overlap, random_sizes=random_sizes)
    if outputs == False:
        tries = 0
        while (tries < 10 and outputs == False):
            outputs = gen_triangles2(IMG, side_length, NUM_SHAPES, timeout = timeout, enable_overlap = enable_overlap, random_sizes=random_sizes)
            tries += 1
    return outputs[0], outputs[1], outputs[2], outputs[3]

def gen_stars(IMG, outer_radius, inner_radius, NUM_SHAPES, pre_centers = [], timeout = 5, enable_overlap = False, random_sizes=False):
    tries = 0
    outputs = gen_stars2(IMG, outer_radius, inner_radius, NUM_SHAPES, pre_centers = pre_centers, timeout = timeout, enable_overlap = enable_overlap, random_sizes=random_sizes)
    if outputs == False:
        tries = 0
        while (tries < 10 and outputs == False):
            outputs = gen_stars2(IMG, outer_radius, inner_radius, NUM_SHAPES, pre_centers = pre_centers, timeout = timeout, enable_overlap = enable_overlap, random_sizes=random_sizes)
            tries += 1
    return outputs[0], outputs[1], outputs[2], outputs[3]

def gen_circles(IMG, radius, NUM_SHAPES, pre_centers = [], timeout = 5, enable_overlap = False, random_sizes=False):
    tries = 0
    outputs = gen_circles2(IMG, radius, NUM_SHAPES, pre_centers = pre_centers, timeout = timeout, enable_overlap = enable_overlap, random_sizes=random_sizes)
    if outputs == False:
        tries = 0
        while (tries < 10 and outputs == False):
            outputs = gen_circles2(IMG, radius, NUM_SHAPES, pre_centers = pre_centers, timeout = timeout, enable_overlap = enable_overlap, random_sizes=random_sizes)
            tries += 1
    return outputs[0], outputs[1], outputs[2], outputs[3]



def gen_triangles2(IMG, side_length, NUM_SHAPES, timeout = 5, enable_overlap = False, random_sizes=False):
    # Get the height of an equilateral triangle
    height = np.sqrt(3) / 2 * side_length

    # Get image dimensions
    img_height, img_width = IMG.shape[:2]
    
    # List to hold the centers and all the drawn triangles
    triangle_centers = []
    drawn_triangles = []

    # Ensure the image is in a binary format (0 or 255)
    img_copy = IMG.copy()
    
    # Convert the image to binary if it isn't already
    if len(img_copy.shape) == 3:  # If the image is not single-channel (binary), convert to grayscale
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    
    # Ensure the image is in the correct binary format (uint8 with 0 or 255)
    img_copy = np.where(img_copy > 0, 255, 0).astype(np.uint8)
    start_time = time.time()

    size_list = list()
    while len(triangle_centers) < NUM_SHAPES:

        if time.time() - start_time > timeout:
            return False

        if random_sizes:
            side_length = random.randint(5, 20) * 2
            size_list.append(side_length)

        # Step 1: Randomly select center of the triangle inside the image
        center_x = random.randint(int(side_length / 2), img_width - int(side_length / 2))
        center_y = random.randint(int(height / 2), img_height - int(height / 2))

        # Step 2: Randomly select an orientation (rotation angle)
        angle = random.uniform(0, 360)

        # Step 3: Calculate the vertices of the equilateral triangle based on the center and side_length
        half_side = side_length / 2
        triangle_height = np.sqrt(3) / 2 * side_length
        vertices = np.array([
            [center_x, center_y - 2 / 3 * triangle_height],  # top vertex
            [center_x - half_side, center_y + triangle_height / 3],  # bottom left
            [center_x + half_side, center_y + triangle_height / 3],  # bottom right
        ])

        # Step 4: Rotate the triangle by the random angle
        triangle_polygon = ShapelyPolygon(vertices)
        rotated_triangle = rotate(triangle_polygon, angle, origin=(center_x, center_y))

        overlap = False
        # Step 5: Check if the new triangle overlaps with any existing ones
        if not enable_overlap:
            for existing_triangle in drawn_triangles:
                if rotated_triangle.intersects(existing_triangle):
                    overlap = True
                    break
            
        if not overlap:
            # If no overlap, draw the triangle in white
            rotated_vertices = np.array(rotated_triangle.exterior.coords).astype(int)
            cv2.fillPoly(img_copy, [rotated_vertices], 255)  # Fill with white color
            
            # Add this triangle's center and the triangle itself to the lists
            triangle_centers.append((center_x, center_y))
            drawn_triangles.append(rotated_triangle)

    return img_copy, triangle_centers, size_list, np.sum(img_copy) / (IMG.shape[0] * IMG.shape[1] * 255 * 255)

def gen_stars2(IMG, outer_radius, inner_radius, NUM_SHAPES, pre_centers = [], timeout = 5, enable_overlap = False, random_sizes=False):
    # Get image dimensions
    img_height, img_width = IMG.shape[:2]
    
    # List to hold the centers and all the drawn stars
    star_centers = []
    drawn_stars = []

    # Ensure the image is in a binary format (0 or 255)
    img_copy = IMG.copy()
    
    # Convert the image to binary if it isn't already
    if len(img_copy.shape) == 3:  # If the image is not single-channel (binary), convert to grayscale
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    
    # Ensure the image is in the correct binary format (uint8 with 0 or 255)
    img_copy = np.where(img_copy > 0, 255, 0).astype(np.uint8)
    solo = img_copy # Stores only the stars, so you can return the fill separately
    # Function to calculate the vertices of a star
    def calculate_star_vertices(center_x, center_y, outer_radius, inner_radius):
        vertices = []
        angle_step = np.pi / 5  # 36 degrees for each outer point
        
        for i in range(5):
            # Outer points
            outer_x = center_x + outer_radius * np.cos(i * 2 * np.pi / 5)
            outer_y = center_y - outer_radius * np.sin(i * 2 * np.pi / 5)
            vertices.append((outer_x, outer_y))

            # Inner points
            inner_x = center_x + inner_radius * np.cos((i + 0.5) * 2 * np.pi / 5)
            inner_y = center_y - inner_radius * np.sin((i + 0.5) * 2 * np.pi / 5)
            vertices.append((inner_x, inner_y))

        return np.array(vertices, dtype=int)
    start_time = time.time()
    size_list = list()
    while len(star_centers) < NUM_SHAPES:
        if time.time() - start_time > timeout:
            return False
        
        if random_sizes:
            outer_radius = random.randint(5, 20)
            inner_radius = outer_radius / 2
            size_list.append(outer_radius)

        # Step 1: Randomly select center of the star inside the image
        center_x = random.randint(int(outer_radius), img_width - int(outer_radius))
        center_y = random.randint(int(outer_radius), img_height - int(outer_radius))

        # Step 2: Randomly select a rotation angle for the star
        angle = random.uniform(0, 360)

        # Step 3: Calculate the vertices of the star
        vertices = calculate_star_vertices(center_x, center_y, outer_radius, inner_radius)

        # Step 4: Create a Shapely polygon for the star and rotate it
        star_polygon = ShapelyPolygon(vertices)
        rotated_star = rotate(star_polygon, angle, origin=(center_x, center_y))

        overlap = False
        if not enable_overlap:
            # Step 5: Check if the new star overlaps with any existing ones
            
            for existing_star in drawn_stars:
                if rotated_star.intersects(existing_star):
                    overlap = True
                    break
            if (overlap == False):
                for predefined_center in pre_centers:
                    px, py = predefined_center
                    dist = np.sqrt((center_x - px)**2 + (center_y - py)**2)
                    if dist <= outer_radius * 2:  # If the distance is less than the outer radius, it overlaps
                        overlap = True
                        break

        if not overlap:
            # If no overlap, draw the star in white
            rotated_vertices = np.array(rotated_star.exterior.coords).astype(int)
            cv2.fillPoly(img_copy, [rotated_vertices], 255)  # Fill with white color
            cv2.fillPoly(solo, [rotated_vertices], 255)
            # Add this star's center and the star itself to the lists
            star_centers.append((center_x, center_y))
            drawn_stars.append(rotated_star)

    return img_copy, star_centers, size_list, np.sum(solo) / (IMG.shape[0] * IMG.shape[1] * 255 **2)

def gen_circles2(IMG, radius, NUM_SHAPES, pre_centers = [], timeout = 5, enable_overlap = False, random_sizes=False):
    # Get image dimensions
    img_height, img_width = IMG.shape[:2]
    
    # List to hold the centers and all the drawn circles
    circle_centers = []
    drawn_circles = []

    # Ensure the image is in a binary format (0 or 255)
    img_copy = IMG.copy()
    
    # Convert the image to binary if it isn't already
    if len(img_copy.shape) == 3:  # If the image is not single-channel (binary), convert to grayscale
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    
    # Ensure the image is in the correct binary format (uint8 with 0 or 255)
    img_copy = np.where(img_copy > 0, 255, 0).astype(np.uint8)
    solo = img_copy
    start_time = time.time()
    size_list = list()
    while len(circle_centers) < NUM_SHAPES:
        if time.time() - start_time > timeout:
            return False

        if random_sizes:
            radius = random.randint(5, 20)
            size_list.append(radius)
        # Step 1: Randomly select center of the circle inside the image
        center_x = random.randint(radius, img_width - radius)
        center_y = random.randint(radius, img_height - radius)


        # Step 3: Create a Shapely point for the circle's center
        circle_center = ShapelyPoint(center_x, center_y)
        
        # Step 4: Create a Shapely buffer to represent the circle
        circle = circle_center.buffer(radius)

        # Step 5: Check if the new circle overlaps with any existing ones
        overlap = False
        if not enable_overlap:
            
            for existing_circle in drawn_circles:
                if circle.intersects(existing_circle):
                    overlap = True
                    break
            
            if overlap == False:

                for center in pre_centers:
                    px,py = center
                    dist = np.sqrt((center_x - px)**2 + (center_y - py)**2)

                    if (dist <= radius * 2):
                        overlap = True
                        break
                    

        if not overlap:
            # If no overlap, draw the circle in white
            cv2.circle(img_copy, (center_x, center_y), radius, (255), -1)  # Fill with white color
            cv2.circle(solo, (center_x, center_y), radius, (255), -1)
            # Add this circle's center and the circle itself to the lists
            circle_centers.append((center_x, center_y))
            drawn_circles.append(circle)

    return img_copy, circle_centers, size_list, np.sum(solo) / (IMG.shape[0] * IMG.shape[1])

def generate_sum(TOTAL, num_parts):
    parts = list()
    while len(parts) < num_parts - 1 and TOTAL > 0:
        num_shape = random.randint(0, TOTAL)
        TOTAL = TOTAL - num_shape
        parts.append(num_shape)
    # Handle edge case where it is [TOTAL, 0, 0]
    if len(parts) ==1 and num_parts == 3:
        parts.append(0)
    
    # Add remainder as the 3rd parts
    parts.append(TOTAL)
    return parts
    