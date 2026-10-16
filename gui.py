"""Tkinter desktop front end for LSB image steganography and steganalysis.

The window lets a user pick a PNG cover image, optionally shrink it, type (or
auto-fill) a message, embed the message, and view histogram-based steganalysis
of the cover versus the stego image. It can also decode a message from any
PNG chosen in a file dialog.

Layout: a toolbar (Select Image, Resize for Test, Fill Max Space, Reset All),
a left pane with the message box, a live "Remaining" character counter and
the action buttons, and a right pane with three auto-fitting image tabs
(Original / Resized, Encoded, Analysis).

How it fits with the other modules:
    encoder.py         encode(): writes one message bit into the least
                       significant bit (LSB) of each R, G and B byte
                       (3 bits per pixel, row-major order), followed by a
                       delimiter of 64 zero bits. Used by "Run Full Analysis".
    decoder.py         decode(): reads the LSBs back and returns the text
                       before the first run of 64 zero bits. Used by
                       "Decode Image".
    steganalysis.py    analyze_comparison(): standard and even/odd pairwise
                       histogram plots of cover vs. stego. Used by
                       "Run Full Analysis".
    main.py            Command-line encode/decode front end (not used here).
    pairwise_stats.py  Numeric even/odd pair statistics (not used here).
    make_all.py        Batch experiment runner (not used here).
    gui.py             This module.

Run it from the repository root (the output folder is relative to the
current working directory)::

    python gui.py

Side effects: creates ``results/`` in the current working directory and
writes all generated PNGs there, overwriting files with the same name. The
GUI reports results and errors through message boxes and the status bar and
has no print calls of its own. encoder.encode prints a line to stdout on
success and on failure; decoder.decode prints only when it fails.
"""
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk

import encoder
import decoder
import steganalysis


# Filler text that "Fill Max Space" repeats and truncates to the image's capacity.
LOREM_IPSUM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut "
    "labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris"
    " nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit"
    " esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in"
    " culpa qui officia deserunt mollit anim id est laborum. "
)


class ImageTab(ttk.Frame):
    """Notebook tab that shows one image scaled to fit the tab.

    The image is rescaled from the stored original whenever the frame is
    resized, preserving aspect ratio. There are no scrollbars or zoom
    controls. Shows the text "No image" when empty.

    Attributes:
        label: ttk.Label that displays the scaled image (or placeholder text).
    """

    def __init__(self, parent):
        """Create an empty tab.

        Args:
            parent: Containing widget, normally the ttk.Notebook.
        """
        super().__init__(parent)
        # Full-resolution source image; every resize is computed from this so
        # repeated resizing does not degrade quality.
        self._pil_original: Image.Image | None = None
        # Current scaled PhotoImage. Tk does not hold a Python reference, so
        # it must be kept alive here or the image disappears when GC'd.
        self._photo: ImageTk.PhotoImage | None = None

        self.label = ttk.Label(self, text="No image", anchor="center")
        self.label.pack(fill="both", expand=True)

        # Re-render when the tab area resizes
        self.bind("<Configure>", self._on_resize)

    def set_image_path(self, path: str):
        """Load an image file and display it.

        Args:
            path: Path to an image file Pillow can open.

        Side effects:
            If Pillow cannot open the file, shows an error message box and
            leaves the current image unchanged; that error is not re-raised.
            Errors raised later, when the pixel data is actually read in
            set_pil_image or _render_fit (e.g. a truncated file), are not
            caught here.
        """
        try:
            img = Image.open(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {e}")
            return
        self.set_pil_image(img)

    def set_pil_image(self, img: Image.Image):
        """Display an already-loaded PIL image.

        Modes other than RGB/RGBA (e.g. palette or grayscale) are converted
        to RGB first. Rendering is scheduled 10 ms later with after() rather
        than done immediately, and is redone by _on_resize whenever the tab
        later changes size.

        Args:
            img: Image to show. It is stored as the new original.
        """
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        self._pil_original = img
        self.after(10, self._render_fit)

    def clear(self):
        """Remove the image and show the "No image" placeholder."""
        self._pil_original = None
        self._photo = None
        self.label.configure(image="", text="No image")
        self.label.image = None

    # ---- internals ----
    def _on_resize(self, _event):
        """Handle <Configure> events by re-fitting the image, if any.

        Args:
            _event: Tk event object (unused).
        """
        if self._pil_original is not None:
            self._render_fit()

    def _render_fit(self):
        """Scale the original image to fit the frame and show it.

        Uses the largest uniform scale that fits inside the frame minus a
        16 px margin (floored at 5%), resampling with LANCZOS. Does nothing
        if there is no image or the image has a zero dimension.
        """
        if self._pil_original is None:
            return
        # winfo_* report 1 before the widget is first laid out. The max(1, ...)
        # clamps keep w and h at least 1 after the 16 px margin is
        # subtracted, so they never go negative. (The divisors below are
        # img_w and img_h, which the == 0 check guards, and scale is also
        # floored at 0.05.)
        w = max(1, self.winfo_width())
        h = max(1, self.winfo_height())
        # Pad a bit so the image doesn't smack the edges
        w = max(1, w - 16)
        h = max(1, h - 16)
        img_w, img_h = self._pil_original.size
        if img_w == 0 or img_h == 0:
            return
        # Fit-inside scale: the tighter of the two axes wins, so the whole
        # image is visible and the aspect ratio is preserved. Small images
        # are scaled up as well as large ones down.
        scale = min(w / img_w, h / img_h)
        scale = max(scale, 0.05)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))
        resized = self._pil_original.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self.label.configure(image=self._photo, text="")
        # Extra reference on the widget itself (common Tkinter idiom) so the
        # PhotoImage is not garbage-collected while displayed.
        self.label.image = self._photo


class SteganoApp:
    """Main application window: builds the UI and handles all user actions.

    State:
        image_path: Cover image chosen with Select Image, or None.
        resized_image_path: Path of the 400x400-bounded copy made by
            Resize for Test, or None. When set, it is used instead of
            image_path for capacity, filling and analysis.
        encoded_image_path: Output path the last Run Full Analysis tried
            to write the stego image to, or None. It is set before encoding
            runs, so the file may not exist or may be from an earlier run
            (see the NOTE in perform_full_analysis).
        max_chars: Message capacity in characters for the active image.
            Any value <= 0 means no limit is enforced: 0 with no image or
            after a decode, negative for images under 22 pixels.
        output_dir: Folder for generated files ("results", relative to the
            current working directory).
    """

    def __init__(self, root: tk.Tk):
        """Configure the root window, create results/ and build the UI.

        Args:
            root: The Tk root window to populate.

        Side effects:
            Creates the "results" directory in the current working directory
            if it does not exist.
        """
        self.root = root
        self.root.title("Steganography Analysis Tool")
        self.root.geometry("1150x780")
        self.root.minsize(900, 600)

        # ttk theme & style
        style = ttk.Style(self.root)
        # Use the first theme this Tk build accepts; theme_use raises
        # TclError for unknown themes. "clam" ships with every Tk, so in
        # practice it is always chosen.
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
        """Create the File and Help menus and bind Ctrl+O to Select Image."""
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

        # The accelerator text above is only a label; this binding makes the
        # shortcut actually work, application-wide.
        self.root.bind_all("<Control-o>", lambda e: self.select_image())

    def _build_body(self):
        """Create the toolbar, message pane, action buttons and image tabs.

        Sets self.message_text, self.char_count_label, self.tabs and the three
        ImageTab attributes (tab_original, tab_encoded, tab_analysis).
        """
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
        """Create the one-line status label at the bottom of the window."""
        self.status = ttk.Label(self.root, text="", anchor="w")
        self.status.pack(side="bottom", fill="x")

    def _set_status(self, text: str):
        """Show text in the status bar.

        Args:
            text: Message to display.
        """
        self.status.configure(text=text)

    # ------------- Helpers -------------
    def _schedule_char_count(self):
        """Run update_char_count 10 ms after a key release.

        The update is deferred with root.after instead of running inside the
        event handler. Only <KeyRelease> triggers it, so text inserted
        without a key event is not counted or truncated until the next key
        is released.
        """
        self.root.after(10, self.update_char_count)

    def update_char_count(self):
        """Enforce the character limit and refresh the "Remaining" label.

        If max_chars > 0 and the message is longer, the text after the first
        max_chars characters is deleted. With max_chars <= 0 nothing is
        truncated and the label shows "Remaining: 0".
        """
        # "end-1c" excludes the trailing newline that a Tk Text widget
        # always keeps at the end of its content.
        content = self.message_text.get("1.0", "end-1c")
        current_len = len(content)
        if self.max_chars > 0 and current_len > self.max_chars:
            # Tk index "1.0 + N chars" = N characters after the start.
            self.message_text.delete(f"1.0 + {self.max_chars} chars", tk.END)
            current_len = self.max_chars
        remaining = max(0, self.max_chars - current_len)
        self.char_count_label.config(text=f"Remaining: {remaining}")

    @staticmethod
    def calculate_capacity(image_path: str | None) -> int:
        """Return how many message characters fit in an image.

        Args:
            image_path: Path to the cover image, or None.

        Returns:
            Capacity in characters, assuming 8 bits per character. Returns 0
            if image_path is empty or the image cannot be opened. Can be
            negative for images under 22 pixels. update_char_count treats
            <= 0 as no limit, while fill_max_message inserts an empty
            message for it.
        """
        if not image_path:
            return 0
        try:
            with Image.open(image_path) as img:
                # One hidden bit per R, G and B byte = 3 bits per pixel.
                # // 8 converts bits to whole bytes (characters), and - 8
                # reserves 8 bytes for the 64-zero-bit end delimiter. This
                # matches the capacity check in encoder.encode exactly.
                # NOTE: assumes 1 byte per char, but encoder uses
                # format(ord(c), '08b'), which emits more than 8 bits for code
                # points > 255, so such text can overflow this limit, and
                # decoder reads 8-bit chunks and garbles it ('€5' -> '\x82°').
                return ((img.width * img.height * 3) // 8) - 8
        except Exception:
            return 0

    # ------------- Actions -------------
    def select_image(self):
        """Ask for a PNG cover image and make it the active image.

        Resets all state first (which also clears the message text), then
        shows the image in the Original tab and sets max_chars from its
        capacity. Does nothing if the dialog is
        cancelled.
        """
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
        """Save a copy of the selected image that fits within 400x400.

        The image is shrunk in memory to fit within 400x400 (aspect ratio
        kept; never enlarged, so an image already that small is copied at
        its own size) and saved as a new file; the selected file on disk is
        not modified. The copy becomes the active image for capacity,
        filling and analysis. If the new capacity is > 0, the existing
        message is truncated to it.

        Side effects:
            Writes results/<name>_resized.png (overwriting any existing
            file). Shows a warning if no image is selected, an info box on
            success, and an error box if resizing or saving fails.
        """
        if not self.image_path:
            messagebox.showwarning("Warning", "Please select an image first.")
            return
        try:
            img = Image.open(self.image_path)
            # thumbnail() resizes in place to fit the box and only shrinks.
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
        """Replace the message with filler text that uses the full capacity.

        Repeats LOREM_IPSUM and truncates it to exactly max_chars characters
        for the active image (the resized copy if one exists, else the
        original). If that capacity is <= 0 the message box is left empty.
        Shows a warning if no image is selected.
        """
        active_image_path = self.resized_image_path if self.resized_image_path else self.image_path
        if not active_image_path:
            messagebox.showwarning("Warning", "Please select or resize an image to set capacity.")
            return
        self.max_chars = self.calculate_capacity(active_image_path)
        # Repeat the filler enough times to cover max_chars (one extra copy
        # handles the remainder), then cut to exactly max_chars.
        full_message = (LOREM_IPSUM * (self.max_chars // len(LOREM_IPSUM) + 1))[: self.max_chars]
        self.message_text.delete("1.0", tk.END)
        self.message_text.insert("1.0", full_message)
        self.update_char_count()

    def perform_full_analysis(self):
        """Embed the message in the active image, then plot the steganalysis.

        Encodes the text box contents into the resized copy if one exists,
        else the original, and shows the result in the Encoded tab. It then
        calls steganalysis.analyze_comparison on the blue channel, which
        draws the standard histograms of cover and stego plus histograms of
        the even/odd pair differences h[2k] - h[2k+1]. Flipping an LSB only
        moves a value between 2k and 2k+1, so embedding tends to equalize
        each pair's counts and shrink those differences. Bits are written
        from the first pixel onward, so only the pixels that carry message
        bits are affected and a short message changes the plots very
        little. The composite plot is shown in the Analysis tab.

        Side effects:
            Writes results/<name>_encoded.png, where <name> is the active
            image's base name (e.g. foo_resized_encoded.png after Resize for
            Test), results/comparison_analysis.png and four fixed-name PNGs
            (1_original_standard_hist.png,
            2_encoded_standard_hist.png, 3_original_difference_analysis.png,
            4_encoded_difference_analysis.png) to results/, overwriting
            previous runs. Shows warnings if no image or message is set and
            an error box if encoding or analysis raises.
        """
        original_for_analysis = self.resized_image_path if self.resized_image_path else self.image_path
        if not original_for_analysis:
            messagebox.showwarning("Warning", "Please select an image to serve as the 'original' for the test.")
            return
        # strip() removes leading/trailing whitespace, so the embedded text
        # can differ from what is shown in the box.
        message = self.message_text.get("1.0", "end-1c").strip()
        if not message:
            messagebox.showwarning("Warning", "Please enter a message to encode.")
            return
        base_name = os.path.splitext(os.path.basename(original_for_analysis))[0]
        self.encoded_image_path = os.path.join(self.output_dir, f"{base_name}_encoded.png")
        try:
            # NOTE: encoder.encode prints and returns None (no exception, no
            # file written) if the message is too large or the image is
            # missing, so this code cannot tell that encoding failed. The
            # status bar still says "Encoded message into ...". If a stale
            # <name>_encoded.png from an earlier run exists, it is shown,
            # analyzed and reported as "Full analysis complete!". If not, two
            # "No such file" error dialogs appear (one from
            # ImageTab.set_image_path, one from the except below when
            # analyze_comparison fails), and neither names the real cause.
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
        """Ask for a PNG and show any hidden message in the text box.

        The decoded text replaces the text box contents and max_chars is set
        to 0, so no length limit applies until Select Image, Resize for Test
        or Fill Max Space sets it again. The selected cover image is kept, so
        Run Full Analysis can then be given a message over its capacity.
        Decoding runs on the Tk main thread and scans every pixel, so the
        window is unresponsive until it finishes (about 0.8 s for a
        1920x1080 image in testing).

        Side effects:
            Shows a success box; the "No hidden message found" error box if
            decode returns None (file not found or no delimiter) or an empty
            string; or an error box if decoding raises. Does nothing if the
            dialog is cancelled.
        """
        path_to_decode = filedialog.askopenfilename(title="Select Encoded Image",
                                                    filetypes=[("PNG files", "*.png")])
        if not path_to_decode:
            return
        try:
            # NOTE: decoder.decode finds the first run of 64 zero bits. If
            # the message's last character has an even code point (LSB 0),
            # its trailing zeros join the delimiter, the match starts early,
            # and that character is dropped ('hello world' -> 'hello worl').
            hidden_message = decoder.decode(path_to_decode)
            # decode returns None (after printing) when the file is not found
            # or no delimiter is found; an empty string is also treated as
            # "no message".
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
        """Clear every path, the capacity limit, all three tabs and the text.

        Files already written to results/ are left in place.
        """
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


# Launch the GUI when run as a script: python gui.py
if __name__ == "__main__":
    root = tk.Tk()
    app = SteganoApp(root)
    root.mainloop()
