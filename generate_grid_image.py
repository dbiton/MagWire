from PIL import Image, ImageDraw

def generate_grid_image(
    width_cm,
    height_cm,
    grid_space_size_cm,
    grid_line_width_cm,
    grid_color_rgb,
    corner_square_size_cm,
    corner_square_color_rgb,
    dpi=300
):
    """
    Generate a transparent PNG image of size width_cm x height_cm (in cm at `dpi`),
    with a grid at the specified spacing & line width, plus a filled square in each corner.

    :param width_cm:                Total width of the image in centimeters.
    :param height_cm:               Total height of the image in centimeters.
    :param grid_space_size_cm:      Spacing between grid lines in centimeters.
    :param grid_line_width_cm:      Thickness of each grid line in centimeters.
    :param grid_color_rgb:          Tuple (R, G, B) or (R, G, B, A) for the grid color.
    :param corner_square_size_cm:   Size of each corner square in centimeters.
    :param corner_square_color_rgb: Tuple (R, G, B) or (R, G, B, A) for the corner squares.
    :param dpi:                     The DPI (dots per inch) to assume when generating the image.
                                    Default 300 for print.
    :return:                        A Pillow Image object (RGBA) with the drawn content.
    """

    # Helper to convert centimeters to pixels at the given DPI
    # 1 inch = 2.54 cm, so px = cm * dpi / 2.54
    def cm_to_px(cm):
        return int(round(cm * dpi / 2.54))

    # Convert all relevant dimensions to pixels
    width_px = cm_to_px(width_cm)
    height_px = cm_to_px(height_cm)
    grid_space_px = cm_to_px(grid_space_size_cm)
    grid_line_width_px = max(1, cm_to_px(grid_line_width_cm))  # ensure at least 1 pixel
    corner_square_px = cm_to_px(corner_square_size_cm)

    # Ensure grid_color_rgb and corner_square_color_rgb have alpha channels
    if len(grid_color_rgb) == 3:
        grid_color = (*grid_color_rgb, 255)
    else:
        grid_color = grid_color_rgb

    if len(corner_square_color_rgb) == 3:
        corner_square_color = (*corner_square_color_rgb, 255)
    else:
        corner_square_color = corner_square_color_rgb

    # Create a new RGBA image with transparent background
    image = Image.new("RGBA", (width_px, height_px), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    # ----- Draw the grid -----
    # Vertical lines
    for x in range(0, width_px + 1, grid_space_px):
        draw.line([(x, 0), (x, height_px)], fill=grid_color, width=grid_line_width_px)

    # Horizontal lines
    for y in range(0, height_px + 1, grid_space_px):
        draw.line([(0, y), (width_px, y)], fill=grid_color, width=grid_line_width_px)

    # ----- Draw corner squares -----
    # Top-left
    draw.rectangle(
        [(0, 0), (corner_square_px, corner_square_px)],
        fill=corner_square_color
    )
    # Top-right
    draw.rectangle(
        [(width_px - corner_square_px, 0), (width_px, corner_square_px)],
        fill=corner_square_color
    )
    # Bottom-left
    draw.rectangle(
        [(0, height_px - corner_square_px), (corner_square_px, height_px)],
        fill=corner_square_color
    )
    # Bottom-right
    draw.rectangle(
        [
            (width_px - corner_square_px, height_px - corner_square_px),
            (width_px, height_px)
        ],
        fill=corner_square_color
    )

    return image

# Example usage
if __name__ == "__main__":
    # For A4 = 21 cm x 29.7 cm @ 300 DPI
    # Grid lines 1 cm apart, line thickness = 0.05 cm
    # Grid color = (0,0,0) i.e. black; corner squares = (255,0,0) i.e. red
    img = generate_grid_image(
        width_cm=15.0,
        height_cm=20.0,
        grid_space_size_cm=0.5,
        grid_line_width_cm=0.05,
        grid_color_rgb=(0, 0, 255),
        corner_square_size_cm=2.0,
        corner_square_color_rgb=(0, 255, 0),
        dpi=300
    )

    # Save as a PNG file
    img.save("grid_on_A4.png", "PNG")
    print("Saved 'grid_on_A4.png'. Be sure to print at 100% (no scaling).")
