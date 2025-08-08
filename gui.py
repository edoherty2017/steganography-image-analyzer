
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import os
import encoder
import decoder
import steganalysis 

LOREM_IPSUM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. "

class SteganoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Steganography Analysis Tool")
        # --- MODIFIED: Start with a slightly larger, resizable window ---
        self.root.geometry("1000x800")
        self.root.minsize(800, 600)
        self.root.configure(bg="#2E2E2E")

        self.image_path = None
        self.resized_image_path = None
        self.encoded_image_path = None
        self.max_chars = 0
        
        self.output_dir = "results"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # --- Top Control Frame ---
        control_frame = tk.Frame(root, bg="#2E2E2E")
        control_frame.pack(pady=10, padx=20, fill="x")

        # --- MODIFIED: Removed custom bg/fg styling from buttons to fix macOS rendering ---
        self.select_btn = tk.Button(control_frame, text="Select Image", command=self.select_image)
        self.select_btn.pack(side=tk.LEFT, padx=5)

        self.resize_btn = tk.Button(control_frame, text="Resize for Test", command=self.resize_image)
        self.resize_btn.pack(side=tk.LEFT, padx=5)
        
        self.fill_btn = tk.Button(control_frame, text="Fill Max Space", command=self.fill_max_message)
        self.fill_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = tk.Button(control_frame, text="Reset All", command=self.reset_all)
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        # --- Image Comparison Frame ---
        image_comp_frame = tk.Frame(root, bg="#2E2E2E")
        image_comp_frame.pack(pady=10, padx=20, fill="x")

        self.image_label_left = tk.Label(image_comp_frame, text="Original / Resized Image", bg="#3C3C3C", fg="white", height=15)
        self.image_label_left.pack(side=tk.LEFT, padx=10, fill="both", expand=True)

        self.image_label_right = tk.Label(image_comp_frame, text="Encoded Image", bg="#3C3C3C", fg="white", height=15)
        self.image_label_right.pack(side=tk.RIGHT, padx=10, fill="both", expand=True)

        # --- Message & Action Frame ---
        msg_action_frame = tk.Frame(root, bg="#2E2E2E")
        msg_action_frame.pack(pady=10, padx=20, fill="x")

        msg_frame = tk.Frame(msg_action_frame, bg="#2E2E2E")
        msg_frame.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 20))
        self.message_text = scrolledtext.ScrolledText(msg_frame, wrap=tk.WORD, width=60, height=5, bg="#3C3C3C", fg="white", insertbackground="white")
        self.message_text.pack(fill="x", expand=True)
        self.message_text.bind("<KeyRelease>", self.check_text_length)
        self.char_count_label = tk.Label(msg_frame, text="Remaining: 0", bg="#2E2E2E", fg="white")
        self.char_count_label.pack(anchor="e")

        action_frame = tk.Frame(msg_action_frame, bg="#2E2E2E")
        action_frame.pack(side=tk.RIGHT)
        self.encode_btn = tk.Button(action_frame, text="Run Full Analysis", command=self.perform_full_analysis, height=2)
        self.encode_btn.pack(pady=5)
        self.decode_btn = tk.Button(action_frame, text="Decode Image", command=self.perform_decode, height=2)
        self.decode_btn.pack(pady=5)
        
        # --- MODIFIED: Analysis label now expands to fill remaining space ---
        self.analysis_label = tk.Label(root, text="Analysis Results Will Appear Here", bg="#3C3C3C", fg="white")
        self.analysis_label.pack(pady=10, padx=20, fill="both", expand=True)

        self.status_label = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#2E2E2E", fg="white")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def check_text_length(self, event=None):
        self.root.after(10, self.update_char_count)

    def update_char_count(self):
        content = self.message_text.get("1.0", "end-1c")
        current_len = len(content)
        if current_len > self.max_chars and self.max_chars > 0:
            self.message_text.delete(f"1.0 + {self.max_chars} chars", tk.END)
            current_len = self.max_chars
        remaining = self.max_chars - current_len
        self.char_count_label.config(text=f"Remaining: {remaining}")

    def calculate_capacity(self, image_path):
        if not image_path: return 0
        try:
            with Image.open(image_path) as img:
                return ((img.width * img.height * 3) // 8) - 8
        except: return 0

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
        if not path: return
        self.reset_all()
        self.image_path = path
        self.display_image(self.image_path, self.image_label_left)
        self.status_label.config(text=f"Selected: {os.path.basename(self.image_path)}")
        self.max_chars = self.calculate_capacity(self.image_path)
        self.update_char_count()

    def resize_image(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "Please select an image first.")
            return
        try:
            img = Image.open(self.image_path)
            img.thumbnail((400, 400), Image.Resampling.LANCZOS) # Slightly smaller thumbnail for resize
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            self.resized_image_path = os.path.join(self.output_dir, f"{base_name}_resized.png")
            img.save(self.resized_image_path, "PNG")
            messagebox.showinfo("Success", f"Image resized and saved as {self.resized_image_path}")
            self.display_image(self.resized_image_path, self.image_label_left)
            self.status_label.config(text=f"Resized to {os.path.basename(self.resized_image_path)}. Ready for analysis.")
            self.max_chars = self.calculate_capacity(self.resized_image_path)
            self.update_char_count()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to resize image: {e}")

    def fill_max_message(self):
        active_image_path = self.resized_image_path if self.resized_image_path else self.image_path
        if not active_image_path:
            messagebox.showwarning("Warning", "Please select or resize an image to set capacity.")
            return
        self.max_chars = self.calculate_capacity(active_image_path)
        full_message = (LOREM_IPSUM * (self.max_chars // len(LOREM_IPSUM) + 1))[:self.max_chars]
        self.message_text.delete("1.0", tk.END)
        self.message_text.insert(tk.INSERT, full_message)
        self.update_char_count()

    def perform_full_analysis(self):
        original_for_analysis = self.resized_image_path if self.resized_image_path else self.image_path
        if not original_for_analysis:
            messagebox.showwarning("Warning", "Please select an image to serve as the 'original' for the test.")
            return
        message = self.message_text.get("1.0", "end-1c").strip()
        if not message:
            messagebox.showwarning("Warning", "Please enter a message to encode.")
            return
        base_name = os.path.splitext(os.path.basename(original_for_analysis))[0]
        self.encoded_image_path = os.path.join(self.output_dir, f"{base_name}_encoded.png")
        try:
            encoder.encode(original_for_analysis, message, self.encoded_image_path)
            self.status_label.config(text=f"Encoded message into {os.path.basename(self.encoded_image_path)}")
            self.display_image(self.encoded_image_path, self.image_label_right)
            
            analysis_output_path = os.path.join(self.output_dir, "comparison_analysis.png")
            steganalysis.analyze_comparison(original_for_analysis, self.encoded_image_path, analysis_output_path)
            
            self.display_image(analysis_output_path, self.analysis_label, is_analysis_plot=True)
            messagebox.showinfo("Success", "Full analysis complete!")
            self.status_label.config(text="Analysis complete.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during analysis: {e}")

    def display_image(self, path, label, is_analysis_plot=False):
        try:
            img = Image.open(path)
            # --- MODIFIED: Use a larger thumbnail for the main analysis plot ---
            if is_analysis_plot:
                img.thumbnail((900, 900))
            else:
                img.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(img)
            label.config(image=photo, text="")
            label.image = photo
        except Exception as e:
            label.config(image='', text=f"Error loading:\n{os.path.basename(path)}")
            print(f"Error displaying image {path}: {e}")

    def perform_decode(self):
        path_to_decode = filedialog.askopenfilename(title="Select Encoded Image", filetypes=[("PNG files", "*.png")])
        if not path_to_decode: return
        try:
            hidden_message = decoder.decode(path_to_decode)
            if hidden_message:
                self.message_text.delete("1.0", tk.END)
                self.message_text.insert(tk.INSERT, hidden_message)
                self.max_chars = 0
                self.update_char_count()
                messagebox.showinfo("Success", "Message decoded successfully!")
                self.status_label.config(text="Message decoded.")
            else:
                messagebox.showerror("Error", "No hidden message found or delimiter is missing.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during decoding: {e}")

    def reset_all(self):
        self.image_path = None
        self.resized_image_path = None
        self.encoded_image_path = None
        self.max_chars = 0
        
        self.image_label_left.config(image='', text="Original / Resized Image")
        self.image_label_left.image = None
        self.image_label_right.config(image='', text="Encoded Image")
        self.image_label_right.image = None
        self.analysis_label.config(image='', text="Analysis Results Will Appear Here")
        self.analysis_label.image = None

        self.message_text.delete("1.0", tk.END)
        self.status_label.config(text="Ready")
        self.update_char_count()

if __name__ == "__main__":
    root = tk.Tk()
    app = SteganoApp(root)
    root.mainloop()