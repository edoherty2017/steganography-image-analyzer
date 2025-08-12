import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk

import encoder
import decoder
import steganalysis


LOREM_IPSUM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut "
    "labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris"
    " nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit"
    " esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in"
    " culpa qui officia deserunt mollit anim id est laborum. "
)


class ImageTab(ttk.Frame):
    """Simple image viewer that auto-fits the image to available space.
    No scrollbars, no zoom UI. Just resizes on window resize.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self._pil_original: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None

        self.label = ttk.Label(self, text="No image", anchor="center")
        self.label.pack(fill="both", expand=True)

        # Re-render when the tab area resizes
        self.bind("<Configure>", self._on_resize)

    def set_image_path(self, path: str):
        try:
            img = Image.open(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {e}")
            return
        self.set_pil_image(img)

    def set_pil_image(self, img: Image.Image):
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        self._pil_original = img
        self.after(10, self._render_fit)

    def clear(self):
        self._pil_original = None
        self._photo = None
        self.label.configure(image="", text="No image")
        self.label.image = None

    # ---- internals ----
    def _on_resize(self, _event):
        if self._pil_original is not None:
            self._render_fit()

    def _render_fit(self):
        if self._pil_original is None:
            return
        w = max(1, self.winfo_width())
        h = max(1, self.winfo_height())
        # Pad a bit so the image doesn't smack the edges
        w = max(1, w - 16)
        h = max(1, h - 16)
        img_w, img_h = self._pil_original.size
        if img_w == 0 or img_h == 0:
            return
        scale = min(w / img_w, h / img_h)
        scale = max(scale, 0.05)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))
        resized = self._pil_original.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self.label.configure(image=self._photo, text="")
        self.label.image = self._photo


class SteganoApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Steganography Analysis Tool")
        self.root.geometry("1150x780")
        self.root.minsize(900, 600)

        # ttk theme & style
        style = ttk.Style(self.root)
        for candidate in ("clam", "vista", "default", "aqua"):
            try:
                style.theme_use(candidate)
                break
            except tk.TclError:
                continue
        style.configure("TFrame", padding=6)
        style.configure("TButton", padding=(10, 6))
        style.configure("TLabel", padding=(2, 2))
        style.configure("TLabelframe", padding=8)

        self.image_path: str | None = None
        self.resized_image_path: str | None = None
        self.encoded_image_path: str | None = None
        self.max_chars = 0

        self.output_dir = "results"
        os.makedirs(self.output_dir, exist_ok=True)

        self._build_menu()
        self._build_body()
        self._build_statusbar()
        self._set_status("Ready")

    # ------------- UI BUILDERS -------------
    def _build_menu(self):
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=False)
        filemenu.add_command(label="Open Image…", command=self.select_image, accelerator="Ctrl+O")
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        helpmenu = tk.Menu(menubar, tearoff=False)
        helpmenu.add_command(label="About", command=lambda: messagebox.showinfo(
            "About",
            "Steganography Analysis Tool Minimal ttk UI with tabs."
        ))
        menubar.add_cascade(label="Help", menu=helpmenu)
        self.root.config(menu=menubar)

        self.root.bind_all("<Control-o>", lambda e: self.select_image())

    def _build_body(self):
        # Toolbar
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side="top", fill="x")

        ttk.Button(toolbar, text="Select Image", command=self.select_image).pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="Resize for Test", command=self.resize_image).pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="Fill Max Space", command=self.fill_max_message).pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="Reset All", command=self.reset_all).pack(side="left")

        # Main area: left controls + right tabs
        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # Left side: message + actions (no visible scrollbars)
        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        msg_group = ttk.Labelframe(left, text="Message to Embed / Decoded Output")
        msg_group.grid(row=0, column=0, sticky="nsew")

        # Plain Text widget without an attached Scrollbar (trackpad/gesture scroll still works)
        self.message_text = tk.Text(msg_group, wrap="word", height=10)
        self.message_text.grid(row=0, column=0, sticky="nsew")

        self.char_count_label = ttk.Label(msg_group, text="Remaining: 0")
        self.char_count_label.grid(row=1, column=0, sticky="e", pady=(6, 0))

        msg_group.columnconfigure(0, weight=1)
        msg_group.rowconfigure(0, weight=1)

        actions = ttk.Labelframe(left, text="Actions")
        actions.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(actions, text="Run Full Analysis", command=self.perform_full_analysis).grid(row=0, column=0, padx=(0, 6), pady=6, sticky="w")
        ttk.Button(actions, text="Decode Image", command=self.perform_decode).grid(row=0, column=1, pady=6, sticky="w")

        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        # Right side: tabs only (no scrollbars; auto-fit on resize)
        right = ttk.Frame(paned)
        paned.add(right, weight=2)

        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill="both", expand=True)

        self.tab_original = ImageTab(self.tabs)
        self.tab_encoded = ImageTab(self.tabs)
        self.tab_analysis = ImageTab(self.tabs)

        self.tabs.add(self.tab_original, text="Original / Resized")
        self.tabs.add(self.tab_encoded, text="Encoded")
        self.tabs.add(self.tab_analysis, text="Analysis")

        # Live char count updates on typing
        self.message_text.bind("<KeyRelease>", lambda e: self._schedule_char_count())

    def _build_statusbar(self):
        self.status = ttk.Label(self.root, text="", anchor="w")
        self.status.pack(side="bottom", fill="x")

    def _set_status(self, text: str):
        self.status.configure(text=text)

    # ------------- Helpers -------------
    def _schedule_char_count(self):
        self.root.after(10, self.update_char_count)

    def update_char_count(self):
        content = self.message_text.get("1.0", "end-1c")
        current_len = len(content)
        if self.max_chars > 0 and current_len > self.max_chars:
            self.message_text.delete(f"1.0 + {self.max_chars} chars", tk.END)
            current_len = self.max_chars
        remaining = max(0, self.max_chars - current_len)
        self.char_count_label.config(text=f"Remaining: {remaining}")

    @staticmethod
    def calculate_capacity(image_path: str | None) -> int:
        if not image_path:
            return 0
        try:
            with Image.open(image_path) as img:
                return ((img.width * img.height * 3) // 8) - 8
        except Exception:
            return 0

    # ------------- Actions -------------
    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
        if not path:
            return
        self.reset_all()
        self.image_path = path
        self.tab_original.set_image_path(self.image_path)
        self._set_status(f"Selected: {os.path.basename(self.image_path)}")
        self.max_chars = self.calculate_capacity(self.image_path)
        self.update_char_count()

    def resize_image(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "Please select an image first.")
            return
        try:
            img = Image.open(self.image_path)
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            self.resized_image_path = os.path.join(self.output_dir, f"{base_name}_resized.png")
            img.save(self.resized_image_path, "PNG")
            messagebox.showinfo("Success", f"Image resized and saved as {self.resized_image_path}")
            self.tab_original.set_image_path(self.resized_image_path)
            self._set_status(f"Resized to {os.path.basename(self.resized_image_path)}. Ready for analysis.")
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
        full_message = (LOREM_IPSUM * (self.max_chars // len(LOREM_IPSUM) + 1))[: self.max_chars]
        self.message_text.delete("1.0", tk.END)
        self.message_text.insert("1.0", full_message)
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
            self._set_status(f"Encoded message into {os.path.basename(self.encoded_image_path)}")
            self.tab_encoded.set_image_path(self.encoded_image_path)

            analysis_output_path = os.path.join(self.output_dir, "comparison_analysis.png")
            steganalysis.analyze_comparison(original_for_analysis, self.encoded_image_path, analysis_output_path)

            self.tab_analysis.set_image_path(analysis_output_path)
            messagebox.showinfo("Success", "Full analysis complete!")
            self._set_status("Analysis complete.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during analysis: {e}")

    def perform_decode(self):
        path_to_decode = filedialog.askopenfilename(title="Select Encoded Image",
                                                    filetypes=[("PNG files", "*.png")])
        if not path_to_decode:
            return
        try:
            hidden_message = decoder.decode(path_to_decode)
            if hidden_message:
                self.message_text.delete("1.0", tk.END)
                self.message_text.insert("1.0", hidden_message)
                self.max_chars = 0
                self.update_char_count()
                messagebox.showinfo("Success", "Message decoded successfully!")
                self._set_status("Message decoded.")
            else:
                messagebox.showerror("Error", "No hidden message found or delimiter is missing.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during decoding: {e}")

    def reset_all(self):
        self.image_path = None
        self.resized_image_path = None
        self.encoded_image_path = None
        self.max_chars = 0

        self.tab_original.clear()
        self.tab_encoded.clear()
        self.tab_analysis.clear()

        self.message_text.delete("1.0", tk.END)
        self.char_count_label.config(text="Remaining: 0")
        self._set_status("Ready")


if __name__ == "__main__":
    root = tk.Tk()
    app = SteganoApp(root)
    root.mainloop()
