import pygame
import math
import time

# Initialize pygame
pygame.init()

# Screen dimensions with margin
width, height = 1400, 800
margin = 50  # Margin to make sure the corners are visible
screen = pygame.display.set_mode((width, height), pygame.NOFRAME)
pygame.display.set_caption("Red Dot Moving in Odd Pattern with Green Grid and Blue Corners")

# Colors
white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 255)
red = (255, 0, 0)

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# Circle parameters (can be ignored since we're not using a circle anymore)
center_x, center_y = (width // 2), (height // 2)  # center of the screen
angle = 0  # initial angle

# Set the duration of the animation (in seconds)
duration = 10000
start_time = time.time()

# Grid parameters
grid_size = 50  # distance between grid lines

# Corners (with margin)
corners = [
    (margin, margin),  # top-left corner
    (width - margin, margin),  # top-right corner
    (margin, height - margin),  # bottom-left corner
    (width - margin, height - margin)  # bottom-right corner
]

radius = 200
degrees_per_second = 60

# Animation loop
running = True
while running:
    screen.fill(white)
    
    # Draw grid
    for x in range(margin, width - margin, grid_size):
        pygame.draw.line(screen, green, (x, margin), (x, height - margin))  # vertical lines
    for y in range(margin, height - margin, grid_size):
        pygame.draw.line(screen, green, (margin, y), (width - margin, y))  # horizontal lines
    
    # Draw blue dots on the corners
    for corner in corners:
        pygame.draw.circle(screen, blue, corner, 5)

    # Use a combination of trig functions to create odd motion
    x = center_x + radius * math.sin(math.radians(angle))
    y = center_y + radius * math.cos(math.radians(angle))

    # Draw the red dot
    pygame.draw.circle(screen, red, (int(x), int(y)), 10)

    # Increment the angle to move the dot
    angle += degrees_per_second / 60
    if angle >= 360:
        angle = 0

    # Update the display
    pygame.display.update()

    # Check for quitting the program
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Limit the frame rate (60 FPS)
    clock.tick(60)

    # Stop the animation after the duration
    if time.time() - start_time > duration:
        running = False

# Quit pygame
pygame.quit()
