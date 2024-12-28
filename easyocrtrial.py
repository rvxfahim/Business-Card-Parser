import cv2
import easyocr
from PIL import Image
from gpt4all import GPT4All

def main():
    # 1. Initialize OCR
    reader = easyocr.Reader(['en'])
    
    # 2. Read and preprocess image
    image = cv2.imread('business_card.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite('preprocessed_card.jpg', thresh)
    
    # 3. Perform OCR
    text_results = reader.readtext(image)
    
    # Extract just the text components from EasyOCR results
    extracted_text_list = [res[1] for res in text_results]
    combined_ocr_text = "\n".join(extracted_text_list)
    
    # 4. Initialize local LLM (GPT4All)
    # Provide the path to your local model file or use a recognized model name
    model_path = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"  # Example model file
    llm = GPT4All(model_path)
    
    # 5. Prepare a one-shot prompt to parse the OCR text
    #    (Change instructions to fit your specific needs)
    prompt = f"""
You are a helpful assistant. Given the following extracted text in a scattered way with positional metadata from a business card:

---
{combined_ocr_text}
---

Please identify the following details (if available) from the text:
1. Person's Name
2. Title/Designation
3. Organization/Company
4. Address
5. Phone/Mobile
6. Email

Output your findings in a structured JSON format. If something is missing, just mark it as null.
    """
    
    # 6. Run the local LLM inference with the prompt
    #    This is a minimal example. You can adjust the parameters (max_tokens, temp, etc.)
    response = llm.generate(
        prompt,
        max_tokens=300,
        temp=0.2
    )

    # 7. Print or handle the LLM response
    print("=== LLM Output ===")
    print(response)

if __name__ == '__main__':
    main()