import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import cv2
import easyocr
from PIL import Image, ImageTk
import threading
import os
from gpt4all import GPT4All

class BusinessCardAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Business Card Analyzer")
        self.root.geometry("800x600")
        
        # Initialize variables
        self.image_path = None
        self.reader = None
        self.llm = None
        
        # Create GUI elements
        self.create_widgets()
        
        # Initialize OCR and LLM in background
        self.initialize_backend()
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image frame
        image_frame = ttk.LabelFrame(main_frame, text="Business Card Image", padding="5")
        image_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Image preview label
        self.image_label = ttk.Label(image_frame, text="No image selected")
        self.image_label.grid(row=0, column=0, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        self.import_button = ttk.Button(button_frame, text="Import Image", command=self.import_image)
        self.import_button.grid(row=0, column=0, padx=5)
        
        self.analyze_button = ttk.Button(button_frame, text="Analyze Card", command=self.start_analysis, state=tk.DISABLED)
        self.analyze_button.grid(row=0, column=1, padx=5)
        
        # Console output
        console_frame = ttk.LabelFrame(main_frame, text="Analysis Output", padding="5")
        console_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.console = scrolledtext.ScrolledText(console_frame, width=70, height=20)
        self.console.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
    
    def initialize_backend(self):
        def init():
            self.log_message("Initializing OCR engine...")
            self.reader = easyocr.Reader(['en'])
            
            self.log_message("Initializing LLM...")
            model_path = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
            try:
                self.llm = GPT4All(model_path)
                self.log_message("Initialization complete!")
            except Exception as e:
                self.log_message(f"Error initializing LLM: {str(e)}")
        
        # Run initialization in background
        threading.Thread(target=init, daemon=True).start()
    
    def import_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.analyze_button.config(state=tk.NORMAL)
            self.log_message(f"Imported image: {os.path.basename(file_path)}")
    
    def display_image(self, file_path):
        # Load and resize image for display
        image = Image.open(file_path)
        image.thumbnail((300, 300))  # Resize for preview
        photo = ImageTk.PhotoImage(image)
        
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep reference
    
    def start_analysis(self):
        if not self.image_path:
            self.log_message("Please import an image first!")
            return
        
        # Disable buttons during analysis
        self.import_button.config(state=tk.DISABLED)
        self.analyze_button.config(state=tk.DISABLED)
        self.progress.start()
        
        # Run analysis in background
        threading.Thread(target=self.analyze_image, daemon=True).start()
    
    def analyze_image(self):
        try:
            # Read and preprocess image
            self.log_message("Reading and preprocessing image...")
            image = cv2.imread(self.image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Perform OCR
            self.log_message("Performing OCR...")
            text_results = self.reader.readtext(image)
            extracted_text_list = [res[1] for res in text_results]
            combined_ocr_text = "\n".join(extracted_text_list)
            
            # Prepare prompt and run LLM
            self.log_message("Analyzing text with LLM...")
            prompt = f"""
            You are a helpful assistant. Given the following extracted text in a scattered way with positional metadata from a business card:
            ---
            {combined_ocr_text}
            ---
            Please identify the following details (if available):
            1. Person's Name
            2. Title/Designation
            3. Organization/Company
            4. Address
            5. Phone/Mobile
            6. Email
            
            Output your findings in a structured JSON format. If something is missing, mark it as null.
            """
            
            response = self.llm.generate(prompt, max_tokens=300, temp=0.2)
            
            self.log_message("\n=== Analysis Results ===")
            self.log_message(response)
            
        except Exception as e:
            self.log_message(f"Error during analysis: {str(e)}")
        
        finally:
            # Re-enable buttons and stop progress bar
            self.root.after(0, self.finish_analysis)
    
    def finish_analysis(self):
        self.import_button.config(state=tk.NORMAL)
        self.analyze_button.config(state=tk.NORMAL)
        self.progress.stop()
    
    def log_message(self, message):
        self.console.insert(tk.END, message + "\n")
        self.console.see(tk.END)

if __name__ == '__main__':
    root = tk.Tk()
    app = BusinessCardAnalyzer(root)
    root.mainloop()