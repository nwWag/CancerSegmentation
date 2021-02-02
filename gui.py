import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import os
from models.att_unet import AttUNet
from utils import load
import numpy as np
import torch
from torchvision.transforms import Resize
class GUI:
    def __init__(self):
        # Load model
        pretrained_path = 'store/' + 'std_u'
        self.model = AttUNet()
        print("Created Model")
        load(model=self.model, name=pretrained_path)
        # Define image res
        self.resolution_change = Resize((225,225))

        # super().__init__()
        self.root = tk.Tk()
        # Initial Window Size and Position
        self.root.geometry("1000x1000")  # Width x Height
        # Window Title
        self.root.wm_title("Cancer Segmentation")
        # Set min size for Window
        self.root.minsize(300, 300)

        # Create a frame to organize the widgets
        self.frame = tk.Frame(self.root)
        self.frame.pack(side='top', fill='both', expand=True)

        # Define the toolbar
        self.toolbar = tk.Menu(self.root, bd=1)
        self.root.config(menu=self.toolbar)

        # Define the preview
        self.preview = tk.Frame(self.frame, bg='#3c3f41')
        self.preview.pack(fill='both', expand=True)

        # Define the canvas to draw in
        self.canvas = tk.Canvas(self.preview, bg="#2b2b2b")
        self.canvas.place(relx=.1, rely=.1, relwidth=.8, relheight=.8)

        # Define some class variables
        self.file_im = None
        self.seg_mask = None
        self.loaded_model = "Baseline"
        self.mouse_mode = "select"
        self.pen_size = 15
        self.draw_counter = 0
        self.old_x = None
        self.old_y = None
        self.window_width = 1000
        self.window_height = 1000
        self.prev = None

        self.draw_toolbar()

        # dynamically resize the window
        self.root.bind("<Configure>", self.resize)

        # main loop to keep the window running
        self.root.mainloop()

    def resize(self, event):
        if self.window_width != event.width or self.window_height != event.height:
            self.window_width = event.width
            self.window_height = event.height

            if self.file_im is not None:
                self.show_image()

    def draw_toolbar(self):
        dd_select_image = tk.Menu(self.toolbar, tearoff=0)
        self.toolbar.add_cascade(label="Load Image", menu=dd_select_image)
        dd_select_image.add_command(label="Browse", command=lambda: self.load_image_handle())

        dd_select_model = tk.Menu(self.toolbar, tearoff=0)
        self.toolbar.add_cascade(label="Select Model", menu=dd_select_model)
        dd_select_model.add_command(label="Baseline", command=lambda: self.select_model_handle("Baseline"))
        dd_select_model.add_command(label="Attention", command=lambda: self.select_model_handle("Attention"))

        # dd_select_params = tk.Menu(self.toolbar, tearoff=0)
        # self.toolbar.add_cascade(label="Select Params", menu=dd_select_params)
        # dd_select_params.add_command(label="Default(Baseline)", command=lambda: self.select_param_handle("Baseline"))
        # dd_select_params.add_command(label="Default(Attention)", command=lambda: self.select_param_handle("Attention"))
        # dd_select_params.add_command(label="Import", command=lambda: self.select_param_handle("Import"))

        dd_draw = tk.Menu(self.toolbar, tearoff=0)
        self.toolbar.add_cascade(label="Refine", menu=dd_draw)
        dd_draw.add_command(label="Redraw", command=lambda: self.mouse_mode_handle("draw"))
        dd_draw.add_command(label="Erase", command=lambda: self.mouse_mode_handle("erase"))
        dd_draw.add_command(label="Pen Size", command=lambda: self.pen_size_handle())

        # self.toolbar.entryconfigure("Draw", state='active')

    def pen_size_handle(self):
        pen = tk.Tk()
        scale = tk.Scale(pen, from_=1, to=50, orient="vertical", length=200, width=10, sliderlength=15,
                         command=self.pen_resize)
        scale.set(self.pen_size)
        scale.pack(anchor="center")

    def pen_resize(self, size):
        self.pen_size = int(size)

    def load_image_handle(self):
        '''
        File Dialog for browsing a picture
        Also places the picture to the given preview window
        '''
        file = filedialog.askopenfilename(title="Select A File",
                                          filetypes=(("jpeg files", "*.jpg"), ("png files", ".png")))
        # self.file_im
        self.file_im = os.path.relpath(file)
        torch_image = torch.unsqueeze(torch.from_numpy(np.array(Image.open(self.file_im))).permute(2, 0, 1), dim=0)
        resized_image = self.resolution_change(torch_image).float()
        mask, att = self.model(resized_image)
        mask = mask.squeeze().permute(1,2,0)
        mask = ((torch.argmax(mask, dim=2))) * 128
        self.seg_mask = Image.fromarray(mask.cpu().numpy().astype(np.uint8))
        self.att_mask = Image.fromarray(torch.squeeze(att * 255).cpu().numpy().astype(np.uint8))
        self.show_image()

    def select_model_handle(self, selected):
        # TODO: on model selection reload image
        print(selected)
        # self.showImage()

    def select_param_handle(self, selected):
        # TODO: on model selection reload image
        print(selected)
        # showImage(self.preview)

    def mouse_mode_handle(self, selected):
        if self.mouse_mode == selected:
            self.mouse_mode = 'select'
            # self.toolbar.entryconfigure("Draw", relief="raised")
            # self.toolbar.entryconfigure("Erase", relief="raised")
            # self.btn_draw["bg"] = "white"
            # self.btn_erase["bg"] = "white"
        elif selected == "draw":
            self.mouse_mode = selected
            # self.toolbar.entryconfigure("Draw", relief="sunken")
            # self.btn_draw["state"] = "active"
        elif selected == "erase":
            self.mouse_mode = selected
            # self.toolbar.entryconfigure("Erase", relief="sunken")
            # self.btn_draw["state"] = "active"

    def draw_erase(self, event):
        if self.mouse_mode != 'select':
            self.draw_counter += 1
            paint_color = 80
            draw = ImageDraw.Draw(self.seg_mask)

            if self.mouse_mode == "erase":
                paint_color = 0

            draw.ellipse((int(event.x - self.pen_size / 2), int(event.y - self.pen_size / 2),
                          int(event.x + self.pen_size / 2), int(event.y + self.pen_size / 2)), fill=paint_color,
                         outline=paint_color)

            if self.old_x is not None and self.old_y is not None:
                draw.line((event.x, event.y, self.old_x, self.old_y), fill=paint_color, width=self.pen_size)

            self.old_x = event.x
            self.old_y = event.y

            # only update on every 10th event as to not overload the stack
            if self.draw_counter >= 15:
                self.show_image()

    def show_image(self, *_):

        image = Image.open(self.file_im)
        self.canvas.update()
        im_dim = (self.canvas.winfo_width(), self.canvas.winfo_height())
        image = image.resize(im_dim)
        #self.seg_mask = self.seg_mask.resize(im_dim, resample=0).convert('RGBA')
        self.seg_mask = self.att_mask.resize(im_dim, resample=0).convert('RGBA')

        im_clean = image.convert('RGBA')
        im_mask = Image.new('RGBA',im_clean.size,(0,0,0,100))
        image = Image.composite(im_clean, self.seg_mask, im_mask).convert('RGB')
        tkimage = ImageTk.PhotoImage(image)
        self.prev = tk.Label(self.canvas, image=tkimage)
        self.prev.image = tkimage
        self.prev.place(relwidth=1, relheight=1)

        self.bindings()

    def bindings(self):
        self.prev.bind("<B1-Motion>", self.draw_erase)
        self.prev.bind("<Button-1>", self.draw_erase, add='+')
        self.prev.bind("<Button-1>", self.reset_mouse_pos, add='+')
        self.prev.bind("<ButtonRelease-1>", self.show_image, add='+')
        self.prev.bind("<ButtonRelease-1>", self.reset_mouse_pos, add='+')
        self.draw_counter = 0

    def reset_mouse_pos(self, *_):
        self.old_x = None
        self.old_y = None


if __name__ == '__main__':
    app = GUI()
