# Business Card Analyzer

This project is a Business Card Analyzer that uses OCR (Optical Character Recognition) and a local LLM (Language Model) to extract and analyze information from business cards.

## Features

- Import and display business card images.
- Perform OCR to extract text from the images.
- Analyze the extracted text using a local LLM to identify key details such as name, title, company, address, phone number, and email.
- Display the analysis results in a structured format.

## Requirements

- Python 3.x
- OpenCV
- EasyOCR
- Pillow
- GPT4All
- Tkinter (for GUI)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/business-card-analyzer.git
    cd business-card-analyzer
    ```

2. Install the required Python packages:
    ```sh
    pip install opencv-python easyocr pillow gpt4all tkinter
    ```

3. Download the GPT4All model and place it in the project directory:
    ```sh
    # Example model file
    wget https://example.com/path/to/Meta-Llama-3-8B-Instruct.Q4_0.gguf -O Meta-Llama-3-8B-Instruct.Q4_0.gguf
    ```

## Usage

### GUI Application

1. Run the GUI application:
    ```sh
    python GUIOcr.py
    ```

2. Use the GUI to import a business card image and analyze it.

### Command Line

1. Place the business card image in the project directory and name it [business_card.jpg].

2. Run the command line script:
    ```sh
    python easyocrtrial.py
    ```

3. The script will preprocess the image, perform OCR, and analyze the extracted text using the local LLM. The results will be printed to the console.

## License

This project is licensed under the GNU General Public License v3.0.