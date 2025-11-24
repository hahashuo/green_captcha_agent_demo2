# CAPTCHA Viewer

This repository is forked from [MetaAgentX/OpenCaptchaWorld](https://github.com/MetaAgentX/OpenCaptchaWorld).

## Overview

The CAPTCHA Viewer (`app2.py`) is a lightweight Flask application that provides a viewer to access to individual CAPTCHA puzzles in the OpenCaptchaWorld dataset. Unlike the main application (`app.py`) which provides a full web interface for interactive puzzle solving, the viewer is designed for developers and researchers who want direct access to specific puzzles.

## Features

- **Direct Puzzle Access**: Retrieve specific puzzles by type and ID
- **Type Discovery**: List all available CAPTCHA types
- **Puzzle Discovery**: List all puzzles within a specific type

## Running the Viewer

Start the Flask application:

```bash
python app2.py
```

The application will be available at: `http://localhost:7861/`

## API Usage

### 1. Get Available CAPTCHA Types

Retrieve a list of all available CAPTCHA types:

```
GET http://localhost:7861/api/types
```

**Response:**
```json
{
  "types": [
    "Unusual_Detection",
    "Connect_icon",
    "Select_Animal",
    "Coordinates",
    "Hold_Button",
    "Slide_Puzzle",
    "Geometry_Click",
    "Path_Finder",
    "Image_Matching",
    "Image_Recognition",
    "Object_Match",
    "Place_Dot",
    "Click_Order",
    "Pick_Area",
    "Patch_Select",
    "Dart_Count",
    "Misleading_Click",
    "Dice_Count",
    "Bingo",
    "Rotation_Match"
  ]
}
```

### 2. List Puzzles for a Specific Type

Get all puzzle IDs for a given CAPTCHA type:

```
GET http://localhost:7861/api/list_puzzles?type=Connect_icon
```

**Response:**
```json
{
  "type": "Connect_icon",
  "puzzles": [
    "puzzle1.json",
    "puzzle2.json",
    "puzzle3.json",
    ...
  ]
}
```

### 3. Get a Specific Puzzle

View a specific puzzle by navigating to this URL in your browser, specifying both the puzzle type and ID. Once you've completed the puzzle, click the `Download Result` button to save the result as a JSON file.

```
http://localhost:7861/get_puzzle?type=Dart_Count&id=dart_puzzle_1.json
```


## Related Documentation

For information about the main OpenCaptchaWorld platform and benchmark, see the [main README](./README.md).

