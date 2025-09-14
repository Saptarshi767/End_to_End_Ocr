#!/usr/bin/env python3
"""
Create a test image with a table for OCR testing
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_test_table_image():
    """Create a simple table image for testing OCR"""
    
    # Create a white image
    width, height = 800, 600
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        header_font = ImageFont.truetype("arial.ttf", 24)
    except:
        try:
            font = ImageFont.load_default()
            header_font = ImageFont.load_default()
        except:
            font = None
            header_font = None
    
    # Table data
    headers = ["Name", "Age", "City", "Salary"]
    rows = [
        ["John Doe", "30", "New York", "$75,000"],
        ["Jane Smith", "25", "Los Angeles", "$65,000"],
        ["Bob Johnson", "35", "Chicago", "$80,000"],
        ["Alice Brown", "28", "Houston", "$70,000"],
        ["Charlie Davis", "32", "Phoenix", "$72,000"]
    ]
    
    # Table dimensions
    start_x, start_y = 50, 100
    col_width = 150
    row_height = 40
    
    # Draw title
    title = "Employee Information Table"
    if header_font:
        draw.text((width//2 - 150, 30), title, fill='black', font=header_font)
    else:
        draw.text((width//2 - 150, 30), title, fill='black')
    
    # Draw table borders
    table_width = len(headers) * col_width
    table_height = (len(rows) + 1) * row_height
    
    # Outer border
    draw.rectangle([start_x, start_y, start_x + table_width, start_y + table_height], 
                   outline='black', width=2)
    
    # Column lines
    for i in range(1, len(headers)):
        x = start_x + i * col_width
        draw.line([x, start_y, x, start_y + table_height], fill='black', width=1)
    
    # Row lines
    for i in range(1, len(rows) + 1):
        y = start_y + i * row_height
        draw.line([start_x, y, start_x + table_width, y], fill='black', width=1)
    
    # Draw headers
    for i, header in enumerate(headers):
        x = start_x + i * col_width + 10
        y = start_y + 10
        if font:
            draw.text((x, y), header, fill='black', font=font)
        else:
            draw.text((x, y), header, fill='black')
    
    # Draw data rows
    for row_idx, row in enumerate(rows):
        for col_idx, cell in enumerate(row):
            x = start_x + col_idx * col_width + 10
            y = start_y + (row_idx + 1) * row_height + 10
            if font:
                draw.text((x, y), cell, fill='black', font=font)
            else:
                draw.text((x, y), cell, fill='black')
    
    # Save the image
    output_path = "test_table.png"
    image.save(output_path)
    print(f"✅ Test table image created: {output_path}")
    
    return output_path

def create_simple_table():
    """Create a very simple table for basic OCR testing"""
    
    width, height = 600, 400
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Simple table data
    lines = [
        "Product | Price | Stock",
        "Apple   | $1.50 | 100",
        "Banana  | $0.80 | 150", 
        "Orange  | $2.00 | 75"
    ]
    
    # Draw text lines
    y_start = 50
    line_height = 40
    
    for i, line in enumerate(lines):
        y = y_start + i * line_height
        draw.text((50, y), line, fill='black')
        
        # Draw horizontal line under header
        if i == 0:
            draw.line([50, y + 30, 550, y + 30], fill='black', width=2)
    
    # Save simple table
    simple_path = "simple_table.png"
    image.save(simple_path)
    print(f"✅ Simple table image created: {simple_path}")
    
    return simple_path

if __name__ == "__main__":
    print("🎨 Creating test table images...")
    create_test_table_image()
    create_simple_table()
    print("✅ Test images ready for OCR testing!")
    print("\n📋 Usage:")
    print("1. Run: python run_ui.py")
    print("2. Upload test_table.png or simple_table.png")
    print("3. Process the image to extract table data")