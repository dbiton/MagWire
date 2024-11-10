import math
from typing import Tuple
import pymunk
import pymunk.pygame_util
import pygame
import numpy as np
import shapely

# Physics space
space = pymunk.Space()
space.gravity = (0, 0)

# Define constants
magnet_mass = 1.0
wire_rigidity = 1000
wire_length = 60
segment_length = 5
num_segments = wire_length // segment_length

# Static magnet position
pos_static = (500, 300)

# Anchor point
anchor_point = (300, 400)

# Magnetic constants
mu_0 = 4 * np.pi * 10**-7
magnetic_moment_static = (0, 1)
magnetic_moment_moving = (0, 1)

# Create the magwire magnet
moving_body = pymunk.Body(magnet_mass, float("inf"))
moving_body.position = anchor_point[0] + wire_length, anchor_point[1]
moving_shape = pymunk.Circle(moving_body, 10)
moving_shape.elasticity = 0.5
space.add(moving_body, moving_shape)

# Create the robot magnet
static_body = pymunk.Body(magnet_mass, float("inf"))
static_body.position = pos_static
static_shape = pymunk.Circle(static_body, 20)
static_shape.elasticity = 0.5
space.add(static_body, static_shape)

def create_wire():
    # Create the wire as a series of connected segments
    wire_segments = []
    previous_body = None
    for i in range(num_segments):
        segment_body = pymunk.Body(mass=0.1, moment=10)
        segment_body.position = (
            anchor_point[0] + (i + 1) * segment_length,
            anchor_point[1],
        )
        segment_shape = pymunk.Segment(
            segment_body, (0, 0), (segment_length, 0), segment_length / 32
        )
        segment_shape.elasticity = 0.5
        wire_segments.append(segment_body)
        space.add(segment_body, segment_shape)

        if previous_body is not None:
            # Pin joint for the segment connection
            joint = pymunk.PinJoint(
                previous_body, segment_body, (segment_length, 0), (0, 0)
            )
            space.add(joint)

            # Add a rotary spring for bending stiffness
            spring = pymunk.DampedRotarySpring(
                previous_body, segment_body, rest_angle=0, stiffness=wire_rigidity, damping=10
            )
            space.add(spring)

        previous_body = segment_body

    # Connect the first segment to the anchor point
    anchor_joint = pymunk.PinJoint(space.static_body, wire_segments[0], anchor_point, (0, 0))
    space.add(anchor_joint)

    # Connect the last wire segment to the moving magnet
    end_joint = pymunk.PinJoint(previous_body, moving_body, (segment_length, 0), (0, 0))
    space.add(end_joint)

    return wire_segments


def create_collision(angles = [0, 225]):
    n = len(angles)
    square_size = 200
    half_size = square_size // 2
    center_x, center_y = 300, 300
    branch_length = half_size  # Length of each branch
    road_width = 20  # Width of the road and branches
    road_length = 400

    square_polygon = shapely.Polygon([
        (center_x - square_size, center_y + square_size),
        (center_x + square_size // 1.5, center_y + square_size),
        (center_x + square_size // 1.5, center_y - square_size),
        (center_x - square_size, center_y - square_size),    
      ])
    
    collision_polygon = square_polygon

    for angle in angles:
      branch_polygon = shapely.Polygon([
          (center_x - road_width, center_y + road_length),
          (center_x + road_width, center_y + road_length),
          (center_x + road_width, center_y - road_length / 2),
          (center_x - road_width, center_y - road_length / 2),    
        ])
      origin = (center_x, center_y)
      branch_polygon = shapely.affinity.rotate(branch_polygon, angle, origin)
      collision_polygon = collision_polygon.difference(branch_polygon)

    polys = []
    if isinstance(collision_polygon, shapely.Polygon):
        polys = [collision_polygon]
    elif isinstance(collision_polygon, shapely.MultiPolygon):
        polys = collision_polygon.geoms
    for p in polys:
        vertices = list(p.exterior.coords)
        pymunk_shape = pymunk.Poly(space.static_body, vertices)
        space.add(pymunk_shape)

def create_goal(space, x, y, size=20):
    # Define line thickness and color
    color = (255, 0, 0, 255)  # Red color
    thickness = 5

    # Calculate the end points of the two lines forming the "X"
    half_size = size / 2
    line1_start = (x - half_size, y - half_size)
    line1_end = (x + half_size, y + half_size)
    line2_start = (x - half_size, y + half_size)
    line2_end = (x + half_size, y - half_size)

    # Create Pymunk segments for each line, set them as sensors
    line1 = pymunk.Segment(space.static_body, line1_start, line1_end, thickness)
    line1.sensor = True  # No collision
    line2 = pymunk.Segment(space.static_body, line2_start, line2_end, thickness)
    line2.sensor = True  # No collision

    # To draw it in pygame, store the color and line segments
    line1.color = color
    line2.color = color

    # Add segments to space
    space.add(line1)
    space.add(line2)

    return line1, line2  # Return the segments for further customization if needed


def magnetic_force(body0: pymunk.Body, m0: pymunk.Vec2d, body1: pymunk.Body, m1: pymunk.Vec2d):
    """
    Calculate the 2D magnetic force between two magnetic dipoles in pymunk.

    Parameters:
    - body0 (pymunk.Body): The first magnetic body.
    - m0 (pymunk.Vec2d): Magnetic moment vector of the first magnet (A·m²).
    - body1 (pymunk.Body): The second magnetic body.
    - m1 (pymunk.Vec2d): Magnetic moment vector of the second magnet (A·m²).

    Returns:
    - pymunk.Vec2d: Magnetic force vector acting on body0 due to body1.
    """
    # Constants
    mu_0 = 4 * np.pi * 1e-7  # Magnetic constant (N/A²)

    # Vector from body0 to body1
    r_vec = body1.position - body0.position
    r = r_vec.length  # Distance magnitude
    if r == 0:
        raise ValueError("Distance between bodies cannot be zero.")

    # Normalize the distance vector
    r_hat = r_vec.normalized()

    # Convert rotation vectors of body0 and body1 to orientation vectors
    m0_oriented = pymunk.Vec2d(m0.x, m0.y).rotated(body0.angle)
    m1_oriented = pymunk.Vec2d(m1.x, m1.y).rotated(body1.angle)

    # Convert pymunk.Vec2d to numpy arrays for calculations
    m0_np = np.array([m0_oriented.x, m0_oriented.y])
    m1_np = np.array([m1_oriented.x, m1_oriented.y])
    r_hat_np = np.array([r_hat.x, r_hat.y])

    # Calculate dot products
    m0_dot_r = np.dot(m0_np, r_hat_np)
    m1_dot_r = np.dot(m1_np, r_hat_np)
    m0_dot_m1 = np.dot(m0_np, m1_np)

    # Calculate each term in the force equation
    term1 = m1_dot_r * m0_np
    term2 = m0_dot_r * m1_np
    term3 = m0_dot_m1 * r_hat_np
    term4 = -5 * (m0_dot_r) * (m1_dot_r) * r_hat_np

    # Combine terms to calculate force vector
    force_np = (3 * mu_0 / (4 * np.pi * r**5)) * (term1 + term2 + term3 + term4)

    # Convert numpy array back to pymunk.Vec2d
    force_vec = pymunk.Vec2d(force_np[0], force_np[1])
    return force_vec

def simulate(wire_speed: float, magnet_pos: Tuple[float, float]) -> Tuple[float, float]:
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    clock = pygame.time.Clock()
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    create_collision()
    create_goal(space, 450, 150)
    wire_segments = create_wire()

    # Simulation loop
    running = True
    extend_interval = 2.0  # Time in seconds between wire extensions
    time_since_last_extend = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Calculate magnetic force
        r = np.array(moving_body.position) - pos_static

        m0 = pymunk.Vec2d(2000000000, 0.0)  # Magnetic moment of body0 in the x direction
        m1 = pymunk.Vec2d(0.0, 2000000000)  # Magnetic moment of body1 in the y direction

        force = magnetic_force(moving_body, m0, static_body, m1)
        moving_body.apply_force_at_world_point(force, moving_body.position)
        # static_body.angle += 2/60.0 

        # Step the simulation
        space.step(1 / 60.0)
        
        # Extend wire at regular intervals
        time_since_last_extend += 1 / 60.0
        if time_since_last_extend >= extend_interval:
            time_since_last_extend = 0.0

        # Clear screen
        screen.fill((255, 255, 255))

        # Draw objects
        space.debug_draw(draw_options)
        pygame.draw.circle(screen, (0, 0, 255), pos_static, 10)

        # Update screen
        pygame.display.flip()

        # Cap frame rate
        clock.tick(60)

    pygame.quit()
    return moving_body.position
