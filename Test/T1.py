import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps

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
            self.image = Image.open(file_path)
            self.display_image()

    def display_image(self):
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.image_label.configure(image=self.tk_image)

    def rotate_image(self):
        if hasattr(self, 'image'):
            self.image = self.image.rotate(45)
            self.display_image()

    def crop_image(self):
        if hasattr(self, 'image'):
            self.image = self.image.crop((100, 100, 400, 400))
            self.display_image()

    def flip_image(self):
        if hasattr(self, 'image'):
            self.image = ImageOps.flip(self.image)
            self.display_image()

    def resize_image(self):
        if hasattr(self, 'image'):
            self.image = self.image.resize((self.image.width // 2, self.image.height // 2))
            self.display_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()