import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import numpy as np

class ImageProcessorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Processor")

        self.image_label = tk.Label(master)
        self.image_label.pack()

        self.load_button = tk.Button(master, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.rotate_button = tk.Button(master, text="Rotate", command=self.rotate_image)
        self.rotate_button.pack()

        self.crop_button = tk.Button(master, text="Crop", command=self.crop_image)
        self.crop_button.pack()

        self.flip_button = tk.Button(master, text="Flip", command=self.flip_image)
        self.flip_button.pack()

        self.resize_button = tk.Button(master, text="Resize", command=self.resize_image)
        self.resize_button.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.display_image()

    def display_image(self):
        if self.image is not None:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(self.image)
            self.tk_image = ImageTk.PhotoImage(image=img_pil)
            self.image_label.configure(image=self.tk_image)
    
    def rotate_image(self):
        if hasattr(self, 'image'):
            rows, cols, _ = self.image.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
            self.image = cv2.warpAffine(self.image, M, (cols, rows))
            self.display_image()

    def crop_image(self):
        if hasattr(self, 'image'):
            self.image = self.image[100:400, 100:400]
            self.display_image()

    def flip_image(self):
        if hasattr(self, 'image'):
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.display_image()

    def resize_image(self):
        if hasattr(self, 'image'):
            user_input = simpledialog.askstring("Resize Image", "Enter width and height (e.g., '300 200'):")
            if user_input:
                try:
                    width, height = map(int, user_input.split())
                    self.image = cv2.resize(self.image, (width, height))
                    self.display_image()
                except ValueError:
                    messagebox.showerror("Error", "Invalid input. Please enter valid width and height.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
