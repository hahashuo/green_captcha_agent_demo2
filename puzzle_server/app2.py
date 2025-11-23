import os
import json
import argparse
import sys
print(sys.executable)
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load ground truth data for a specific type
def load_ground_truth(captcha_type):
    # Use absolute path from the script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'captcha_data', captcha_type, 'ground_truth.json')
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# Get available CAPTCHA types
def get_captcha_types():
    # Use absolute path from the script's directory
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'captcha_data')
    if not os.path.exists(base_dir):
        return []
    return [d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))]

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/get_puzzle')
def get_puzzle_page():
    return render_template('index2.html')

@app.route('/captcha_data/<captcha_type>/<filename>')
def serve_captcha(captcha_type, filename):
    return send_from_directory(os.path.join('captcha_data', captcha_type), filename)

@app.route('/captcha_data/<captcha_type>/<subdir>/<filename>')
def serve_captcha_subdir(captcha_type, subdir, filename):
    return send_from_directory(os.path.join('captcha_data', captcha_type, subdir), filename)

@app.route('/api/get_puzzle', methods=['GET'])
def get_puzzle():
    # Get type and id from query parameters
    puzzle_type = request.args.get('type')
    selected_puzzle = request.args.get('id')

    if not puzzle_type:
        return jsonify({'error': 'type parameter is required'}), 400

    if not selected_puzzle:
        return jsonify({'error': 'id parameter is required'}), 400

    # Check if puzzle type exists
    captcha_types = get_captcha_types()
    if puzzle_type not in captcha_types:
        return jsonify({'error': f'Invalid puzzle type: {puzzle_type}'}), 400

    # Load ground truth for the selected type
    ground_truth = load_ground_truth(puzzle_type)
    if not ground_truth:
        return jsonify({'error': f'No puzzles found for type: {puzzle_type}'}), 404

    # Check if puzzle exists
    if selected_puzzle not in ground_truth:
        return jsonify({'error': f'Puzzle not found: {selected_puzzle}'}), 404

    # Get the appropriate question prompt based on puzzle type
    if puzzle_type == "Dice_Count":
        prompt = ground_truth[selected_puzzle].get('prompt', "Sum up the numbers on all the dice")
    elif puzzle_type == "Geometry_Click":
        prompt = ground_truth[selected_puzzle].get("question", "Click on the geometric shape")
    elif puzzle_type == "Rotation_Match":
        prompt = ground_truth[selected_puzzle].get("prompt", "Use the arrows to rotate the object to match the reference direction")
    elif puzzle_type == "Slide_Puzzle":
        prompt = ground_truth[selected_puzzle].get("prompt", "Drag the slider component to the correct position")
    elif puzzle_type == "Unusual_Detection":
        prompt = ground_truth[selected_puzzle].get("prompt", "Select the unusual items in the image")
    elif puzzle_type == "Image_Recognition":
        prompt = ground_truth[selected_puzzle].get("prompt", "Select all images matching the description")
    elif puzzle_type == "Bingo":
        prompt = ground_truth[selected_puzzle].get("prompt", "Please click two images to exchange their position to line up the same images to a line")
    elif puzzle_type == "Image_Matching":
        prompt = ground_truth[selected_puzzle].get("prompt", "Using the arrows, match the animal in the left and right image.")
    elif puzzle_type == "Patch_Select":
        prompt = ground_truth[selected_puzzle].get("prompt", "Select all squares with the specified objects")
    elif puzzle_type == "Dart_Count":
        prompt = ground_truth[selected_puzzle].get("prompt", "Use the arrows to pick the image where all the darts add up to the number in the left image.")
    elif puzzle_type == "Object_Match":
        prompt = ground_truth[selected_puzzle].get("prompt", "Use the arrows to change the number of objects until it matches the left image.")
    elif puzzle_type == "Select_Animal":
        prompt = ground_truth[selected_puzzle].get("prompt", "Pick a fox")
    elif puzzle_type == "Coordinates":
        prompt = ground_truth[selected_puzzle].get("prompt", "Using the arrows, move Jerry to the indicated seat")
    elif puzzle_type == "Path_Finder":
        prompt = ground_truth[selected_puzzle].get("prompt", "Use the arrows to move the duck to the spot indicated by the cross")
    elif puzzle_type == "Place_Dot":
        prompt = ground_truth[selected_puzzle].get("prompt", "Click to place a Dot at the end of the car's path")
    elif puzzle_type == "Connect_icon":
        prompt = ground_truth[selected_puzzle].get("prompt", "Using the arrows, connect the same two icons with the dotted line as shown on the left.")
    elif puzzle_type == "Click_Order":
        prompt = ground_truth[selected_puzzle].get("prompt", "Click the icons in order as shown in the reference image.")
    elif puzzle_type == "Hold_Button":
        prompt = ground_truth[selected_puzzle].get("prompt", "Hold the button until it finishes loading.")
    elif puzzle_type == "Misleading_Click":
        prompt = ground_truth[selected_puzzle].get("prompt", "Click the image to continue.")
    elif puzzle_type == "Pick_Area":
        prompt = ground_truth[selected_puzzle].get("prompt", "Click on the largest area outlined by the dotted line")
    else:
        prompt = ground_truth[selected_puzzle].get("prompt", "Solve the CAPTCHA puzzle")

    # Add input_type to tell the frontend what kind of input to show
    input_type = "text"
    if puzzle_type == "Dice_Count":
        input_type = "number"
    elif puzzle_type == "Geometry_Click":
        input_type = "click"
    elif puzzle_type == "Rotation_Match":
        input_type = "rotation"
    elif puzzle_type == "Slide_Puzzle":
        input_type = "slide"
    elif puzzle_type == "Unusual_Detection":
        input_type = "multiselect"
    elif puzzle_type == "Image_Recognition":
        input_type = "image_grid"
    elif puzzle_type == "Bingo":
        input_type = "bingo_swap"
    elif puzzle_type == "Image_Matching":
        input_type = "image_matching"
    elif puzzle_type == "Patch_Select":
        input_type = "patch_select"
    elif puzzle_type == "Dart_Count":
        input_type = "dart_count"
    elif puzzle_type == "Object_Match":
        input_type = "object_match"
    elif puzzle_type == "Select_Animal":
        input_type = "select_animal"
    elif puzzle_type == "Coordinates":
        input_type = "image_matching"
    elif puzzle_type == "Path_Finder":
        input_type = "image_matching"
    elif puzzle_type == "Place_Dot":
        input_type = "place_dot"
    elif puzzle_type == "Connect_icon":
        input_type = "connect_icon"
    elif puzzle_type == "Click_Order":
        input_type = "click_order"
    elif puzzle_type == "Hold_Button":
        input_type = "hold_button"
    elif puzzle_type == "Misleading_Click":
        input_type = "click"
    elif puzzle_type == "Pick_Area":
        input_type = "click"

    # For Rotation_Match, include additional data needed for the interface
    additional_data = {}
    if puzzle_type == "Rotation_Match":
        # Get reference image and object base name
        reference_image = ground_truth[selected_puzzle].get("reference_image")
        object_base_image = ground_truth[selected_puzzle].get("object_base_image")

        if not reference_image or not object_base_image:
            return jsonify({'error': f'Invalid rotation puzzle data: {selected_puzzle}'}), 500

        # Format paths for these images
        ref_path = f'/captcha_data/{puzzle_type}/{reference_image}'

        # Get object base name without extension to construct rotated image paths
        object_base = os.path.splitext(object_base_image)[0]

        # Construct the initial object image path (0 degrees rotation)
        object_path = f'/captcha_data/{puzzle_type}/{object_base}_0.png'

        additional_data = {
            "reference_image": ref_path,
            "object_image": object_path,
            "object_base": object_base,
            "current_angle": 0
        }
    # For Slide_Puzzle, include the component image path and target position data
    elif puzzle_type == "Slide_Puzzle":
        # Get component image name
        component_image = ground_truth[selected_puzzle].get("component_image")

        if not component_image:
            return jsonify({'error': f'Invalid slide puzzle data: {selected_puzzle}'}), 500

        # Format path for the component image
        component_path = f'/captcha_data/{puzzle_type}/{component_image}'

        additional_data = {
            "component_image": component_path,
            "background_image": f'/captcha_data/{puzzle_type}/{selected_puzzle}'
        }
    # For Unusual_Detection, include the grid size
    elif puzzle_type == "Unusual_Detection":
        # Get grid size from ground truth
        grid_size = ground_truth[selected_puzzle].get("grid_size", [2, 3])  # Default to 2x3 grid if not specified

        additional_data = {
            "grid_size": grid_size
        }
    # For Image_Recognition, include the grid images
    elif puzzle_type == "Image_Recognition":
        # Get images array from ground truth
        images = ground_truth[selected_puzzle].get("images", [])
        grid_size = [3, 3]  # Default grid size for image recognition (3x3)

        # Get the subfolder name from the puzzle_id or use a specific subfolder field
        subfolder = ground_truth[selected_puzzle].get("subfolder", selected_puzzle)

        # Include image paths in response - dynamically use the subfolder
        image_paths = [f'/captcha_data/{puzzle_type}/{subfolder}/{img}' for img in images]

        additional_data = {
            "images": image_paths,
            "grid_size": grid_size,
            "question": ground_truth[selected_puzzle].get("question", "Select matching images")
        }
    # For Bingo, include the grid size
    elif puzzle_type == "Bingo":
        # Get grid size from ground truth
        grid_size = ground_truth[selected_puzzle].get("grid_size", [3, 3])  # Default to 3x3 grid if not specified

        additional_data = {
            "grid_size": grid_size,
            "solution_line": ground_truth[selected_puzzle].get("solution_line", {}),
            "answer": ground_truth[selected_puzzle].get("answer", [])
        }
    # For Image_Matching, include the reference image and options
    elif puzzle_type == "Image_Matching":
        # Get the reference image and option images
        reference_image = ground_truth[selected_puzzle].get("reference_image")
        option_images = ground_truth[selected_puzzle].get("option_images", [])
        correct_option_index = ground_truth[selected_puzzle].get("correct_option_index", 0)

        if not reference_image or not option_images:
            return jsonify({'error': f'Invalid image matching data: {selected_puzzle}'}), 500

        # Format paths for these images
        ref_path = f'/captcha_data/{puzzle_type}/{reference_image}'
        option_paths = [f'/captcha_data/{puzzle_type}/{img}' for img in option_images]

        additional_data = {
            "reference_image": ref_path,
            "option_images": option_paths,
            "current_option_index": 0,
            "correct_option_index": correct_option_index
        }
    # For Patch_Select, include the grid size and target object
    elif puzzle_type == "Patch_Select":
        # Get grid size from ground truth, default to 6x6 grid
        grid_size = ground_truth[selected_puzzle].get("grid_size", [5, 5])
        target_object = ground_truth[selected_puzzle].get("target_object", "moon")
        correct_patches = ground_truth[selected_puzzle].get("correct_patches", [])

        additional_data = {
            "grid_size": grid_size,
            "target_object": target_object,
            "correct_patches": correct_patches
        }
    # For Dart_Count, include the reference image and options
    elif puzzle_type == "Dart_Count":
        # Get the reference image and option images
        reference_image = ground_truth[selected_puzzle].get("reference_image")
        option_images = ground_truth[selected_puzzle].get("option_images", [])
        correct_option_index = ground_truth[selected_puzzle].get("correct_option_index", 0)
        reference_number = ground_truth[selected_puzzle].get("reference_number", 0)

        if not reference_image or not option_images:
            return jsonify({'error': f'Invalid dart count data: {selected_puzzle}'}), 500

        # Format paths for these images
        ref_path = f'/captcha_data/{puzzle_type}/{reference_image}'
        option_paths = [f'/captcha_data/{puzzle_type}/{img}' for img in option_images]

        additional_data = {
            "reference_image": ref_path,
            "option_images": option_paths,
            "current_option_index": 0,
            "correct_option_index": correct_option_index,
            "reference_number": reference_number
        }
    # For Object_Match, include the reference image and options
    elif puzzle_type == "Object_Match":
        # Get the reference image and option images
        reference_image = ground_truth[selected_puzzle].get("reference_image")
        option_images = ground_truth[selected_puzzle].get("option_images", [])
        correct_option_index = ground_truth[selected_puzzle].get("correct_option_index", 0)

        if not reference_image or not option_images:
            return jsonify({'error': f'Invalid object match data: {selected_puzzle}'}), 500

        # Format paths for these images
        ref_path = f'/captcha_data/{puzzle_type}/{reference_image}'
        option_paths = [f'/captcha_data/{puzzle_type}/{img}' for img in option_images]

        additional_data = {
            "reference_image": ref_path,
            "option_images": option_paths,
            "current_option_index": 0,
            "correct_option_index": correct_option_index
        }
    # For Select_Animal, include the grid size and target object
    elif puzzle_type == "Select_Animal":
        # Get grid size from ground truth, default to 2x3 grid
        grid_size = ground_truth[selected_puzzle].get("grid_size", [2, 3])
        target_object = ground_truth[selected_puzzle].get("target_object", "fox")
        correct_patches = ground_truth[selected_puzzle].get("correct_patches", [])

        additional_data = {
            "grid_size": grid_size,
            "target_object": target_object,
            "correct_patches": correct_patches
        }
    # For Coordinates, include the reference image and options
    elif puzzle_type == "Coordinates":
        # Get the reference image and option images
        reference_image = ground_truth[selected_puzzle].get("reference_image")
        option_images = ground_truth[selected_puzzle].get("option_images", [])
        correct_option_index = ground_truth[selected_puzzle].get("correct_option_index", 0)

        if not reference_image or not option_images:
            return jsonify({'error': f'Invalid coordinates data: {selected_puzzle}'}), 500

        # Format paths for these images
        ref_path = f'/captcha_data/{puzzle_type}/{reference_image}'
        option_paths = [f'/captcha_data/{puzzle_type}/{img}' for img in option_images]

        additional_data = {
            "reference_image": ref_path,
            "option_images": option_paths,
            "current_option_index": 0,
            "correct_option_index": correct_option_index
        }
    # For Path_Finder, include the reference image and options
    elif puzzle_type == "Path_Finder":
        # Get the reference image and option images
        reference_image = ground_truth[selected_puzzle].get("reference_image")
        options = ground_truth[selected_puzzle].get("options", [])
        correct_option = ground_truth[selected_puzzle].get("correct_option", 0)

        if not reference_image or not options:
            return jsonify({'error': f'Invalid path finder data: {selected_puzzle}'}), 500

        # Format paths for these images
        ref_path = f'/captcha_data/{puzzle_type}/{reference_image}'
        option_paths = [f'/captcha_data/{puzzle_type}/{img}' for img in options]

        additional_data = {
            "reference_image": ref_path,
            "option_images": option_paths,
            "current_option_index": 0,
            "correct_option_index": correct_option
        }
    # For Connect_icon, include the reference image and options
    elif puzzle_type == "Connect_icon":
        # Get the reference image and option images
        reference_image = ground_truth[selected_puzzle].get("reference_image")
        options = ground_truth[selected_puzzle].get("options", [])
        correct_option = ground_truth[selected_puzzle].get("correct_option", 0)

        if not reference_image or not options:
            return jsonify({'error': f'Invalid connect icons data: {selected_puzzle}'}), 500

        # Format paths for these images
        ref_path = f'/captcha_data/{puzzle_type}/{reference_image}'
        option_paths = [f'/captcha_data/{puzzle_type}/{img}' for img in options]

        additional_data = {
            "reference_image": ref_path,
            "option_images": option_paths,
            "current_option_index": 0,
            "correct_option_index": correct_option
        }
    # For Click_Order, include the order image path
    elif puzzle_type == "Click_Order":
        # Get the order image from ground truth
        order_image = ground_truth[selected_puzzle].get("order_image")

        if not order_image:
            return jsonify({'error': f'Invalid click order data: {selected_puzzle}'}), 500

        # Format path for the order image
        order_path = f'/captcha_data/{puzzle_type}/{order_image}'

        additional_data = {
            "order_image": order_path,
            "tolerance": ground_truth[selected_puzzle].get("tolerance", 20)
        }
    # For Hold_Button, include the hold time
    elif puzzle_type == "Hold_Button":
        # Get the required hold time from ground truth
        hold_time = ground_truth[selected_puzzle].get("hold_time", 3)  # Default to 3 seconds if not specified

        additional_data = {
            "hold_time": hold_time
        }
    # For Misleading_Click, include the area to avoid
    elif puzzle_type == "Misleading_Click":
        # Get the area to avoid from ground truth
        avoid_area = ground_truth[selected_puzzle].get("avoid_area", {"x": 0, "y": 0, "width": 0, "height": 0})

        additional_data = {
            "avoid_area": avoid_area
        }
    else:
        prompt = ground_truth[selected_puzzle].get("prompt", "Solve the CAPTCHA puzzle")

    response_data = {
        'puzzle_type': puzzle_type,
        'image_path': f'/captcha_data/{puzzle_type}/{selected_puzzle}' if puzzle_type != "Rotation_Match" else None,
        'puzzle_id': selected_puzzle,
        'prompt': prompt,
        'input_type': input_type,
        'debug_info': f"Type: {puzzle_type}, Input: {input_type}, Puzzle: {selected_puzzle}"
    }

    # Add any additional data for specific puzzle types
    if additional_data:
        response_data.update(additional_data)

    return jsonify(response_data)

@app.route('/api/types', methods=['GET'])
def get_types():
    """Get available CAPTCHA types"""
    return jsonify({
        'types': get_captcha_types()
    })

@app.route('/api/list_puzzles', methods=['GET'])
def list_puzzles():
    """List all available puzzles for a given captcha type"""
    captcha_type = request.args.get('type')

    if not captcha_type:
        return jsonify({'error': 'type parameter is required'}), 400

    # Check if puzzle type exists
    captcha_types = get_captcha_types()
    if captcha_type not in captcha_types:
        return jsonify({'error': f'Invalid puzzle type: {captcha_type}'}), 400

    # Load ground truth for the selected type
    ground_truth = load_ground_truth(captcha_type)
    if not ground_truth:
        return jsonify({'error': f'No puzzles found for type: {captcha_type}'}), 404

    # Return list of puzzle keys
    return jsonify({
        'type': captcha_type,
        'puzzles': list(ground_truth.keys())
    })

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the CAPTCHA puzzle server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind the server to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=7861,
                       help='Port to bind the server to (default: 7861)')
    args = parser.parse_args()

    # For local development
    if os.environ.get('DEVELOPMENT'):
        app.run(debug=True, host=args.host, port=args.port)
    else:
        # For production on Hugging Face Spaces
        app.run(host=args.host, port=args.port)
