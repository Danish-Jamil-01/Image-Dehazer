# main.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
from dehazer.haze_remover import HazeRemover

class DehazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Dehazer")
        self.root.geometry("850x600")
        self.root.minsize(700, 500)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self.image_path = None
        self.original_img_display = None
        self.dehazed_img_display = None
        
        self.remover = HazeRemover()
        
        # --- NEW: Variable for the checkbox ---
        self.enhance_var = tk.BooleanVar(value=True)

        os.makedirs("outputs", exist_ok=True)
        self._setup_ui()

    def _setup_ui(self):
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.grid(row=0, column=0, sticky="ew")
        top_frame.columnconfigure(1, weight=1)

        ttk.Label(top_frame, text="Image Path:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.path_entry = ttk.Entry(top_frame)
        self.path_entry.grid(row=0, column=1, sticky="ew")
        self.browse_button = ttk.Button(top_frame, text="Browse...", command=self._browse_image)
        self.browse_button.grid(row=0, column=2, sticky="e", padx=(5, 0))

        image_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        image_frame.grid(row=1, column=0, sticky="nsew")
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)
        image_frame.rowconfigure(0, weight=1)

        orig_panel = ttk.LabelFrame(image_frame, text="Original Image", padding="5")
        orig_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        orig_panel.columnconfigure(0, weight=1)
        orig_panel.rowconfigure(0, weight=1)
        self.original_canvas = ttk.Label(orig_panel)
        self.original_canvas.grid(row=0, column=0)
        
        dehazed_panel = ttk.LabelFrame(image_frame, text="Dehazed Image", padding="5")
        dehazed_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        dehazed_panel.columnconfigure(0, weight=1)
        dehazed_panel.rowconfigure(0, weight=1)
        self.dehazed_canvas = ttk.Label(dehazed_panel)
        self.dehazed_canvas.grid(row=0, column=0)

        bottom_frame = ttk.Frame(self.root, padding="10")
        bottom_frame.grid(row=2, column=0, sticky="ew")
        bottom_frame.columnconfigure(1, weight=1) # Center the checkbox

        action_frame = ttk.Frame(bottom_frame)
        action_frame.grid(row=0, column=0, sticky="w")
        
        self.dehaze_button = ttk.Button(action_frame, text="Dehaze Image", command=self._run_dehazing, state=tk.DISABLED)
        self.dehaze_button.pack(side=tk.LEFT)
        
        # --- NEW: Checkbox for enhancement ---
        self.enhance_check = ttk.Checkbutton(bottom_frame, text="Enhance Contrast", variable=self.enhance_var)
        self.enhance_check.grid(row=0, column=1, sticky="w")
        
        quit_frame = ttk.Frame(bottom_frame)
        quit_frame.grid(row=0, column=2, sticky="e")
        
        self.quit_button = ttk.Button(quit_frame, text="Quit", command=self.root.destroy)
        self.quit_button.pack(side=tk.RIGHT)
        
        self.clear_button = ttk.Button(quit_frame, text="Clear", command=self._clear_ui)
        self.clear_button.pack(side=tk.RIGHT, padx=(0, 5))

    def _browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if file_path:
            self._clear_ui()
            self.image_path = file_path
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, self.image_path)
            self._display_image(self.image_path, self.original_canvas, "original")
            self.dehaze_button.config(state=tk.NORMAL)

    def _display_image(self, path_or_array, canvas, image_type):
        if isinstance(path_or_array, str):
            img = Image.open(path_or_array)
        else:
            img = Image.fromarray(cv2.cvtColor(path_or_array, cv2.COLOR_BGR2RGB))
        
        preview_size = (600, 600)
        img.thumbnail(preview_size, Image.Resampling.LANCZOS)
        
        photo_image = ImageTk.PhotoImage(img)

        if image_type == "original":
            self.original_img_display = photo_image
        else:
            self.dehazed_img_display = photo_image
        
        canvas.config(image=photo_image)
        canvas.image = photo_image

    def _run_dehazing(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first.")
            return

        try:
            self.dehaze_button.config(text="Processing...", state=tk.DISABLED)
            self.root.update_idletasks()
            
            hazy_image_bgr = cv2.imread(self.image_path)
            if hazy_image_bgr is None:
                raise IOError("Could not read the image file.")

            # --- Pass checkbox value to the process method ---
            enhance_enabled = self.enhance_var.get()
            dehazed_image_bgr = self.remover.process(hazy_image_bgr, enhance=enhance_enabled)
            
            output_dir = "outputs"
            final_filename = os.path.join(output_dir, "dehazed_output.png")
            cv2.imwrite(final_filename, dehazed_image_bgr)
            
            if self.remover.intermediates:
                cv2.imwrite(os.path.join(output_dir, "intermediate_dark_channel.png"), self.remover.intermediates['dark_channel'])
                cv2.imwrite(os.path.join(output_dir, "intermediate_transmission.png"), self.remover.intermediates['transmission'])

            self._display_image(dehazed_image_bgr, self.dehazed_canvas, "dehazed")
            messagebox.showinfo("Success", f"Dehazing complete! All images saved in '{output_dir}'.")

        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred: {e}")
        finally:
            self.dehaze_button.config(text="Dehaze Image", state=tk.NORMAL)

    def _clear_ui(self):
        self.image_path = None
        self.path_entry.delete(0, tk.END)
        self.original_canvas.config(image='')
        self.original_img_display = None
        self.dehazed_canvas.config(image='')
        self.dehazed_img_display = None
        self.dehaze_button.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = DehazeApp(root)
    root.mainloop()