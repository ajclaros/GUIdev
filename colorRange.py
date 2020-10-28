from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog
LARGE_FONT= ("Verdana", 12)


class GuiApp(tk.Tk):

    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)
        self.panels = []
        self.images = []
        self.filenames = []
        container = tk.Frame(self)
        container.grid(row=0, column=0)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, Analysis):

            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()




class StartPage(tk.Frame):
    total_images = 0
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="First Page", font=LARGE_FONT)
        label.grid(row=0, column=0, pady=10, padx=10)
        buttons = []
        buttons.append(ttk.Button(self, text="Select Image",
                                      command=lambda: self.select_image(controller, self.total_images)))
        buttons.append(ttk.Button(self, text="Analysis page", command=lambda: controller.show_frame(Analysis)))
        buttons[0].grid(row=2, column=0)
        buttons[1].grid(row=2,column=1)



    def select_image(self, content, gridx):
        path = filedialog.askopenfilename(initialdir = "~/org/research/testimages/")
        name=path.split('/')
        name = name[-1].split('.')
        name = name[0]
        content.filenames.append(name)
        print(name)

        if len(path) > 0:
            image = cv2.imread(path)
            imagecv = image
            content.images.append(imagecv)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = image.resize((200, 200), Image.ANTIALIAS)
            image = ImageTk.PhotoImage(image)
            content.panels.append(tk.Label(image=image))
            image_idx = len(content.panels)-1
            content.panels[image_idx].image = image
            content.panels[image_idx].grid(row = 1,
                                        column = gridx,
                                        padx = 20,
                                        pady = 10)
            self.total_images += 1


class Analysis(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="Analysis"
        label.grid(row=0, column=0, pady=10, padx=10)
        buttons = []
        buttons.append(ttk.Button(self, text= "First page",
                                  command = lambda: controller.show_frame(StartPage)))
        buttons.append(ttk.Button(self, text="Color Histogram",
                                  command = lambda: self.color_histogram(controller)))
        buttons[1].grid(row=2, column=0)


    def color_histogram(self, content):
        image_arr = []
        h = []
        s = []
        v = []
        norm = mpl.colors.Normalize(vmin=0, vmax=255)
        cm = plt.cm.hsv
        fig = plt.figure()
        ax = fig.add_axes([0.05, 0.2, 0.9,0.7])
        ax1 = fig.add_axes([0.05, 0.1, 0.9, 0.1])
        cb1 = mpl.colorbar.ColorbarBase(ax1, cmap = cm, norm=norm, orientation = 'horizontal')
        edgecolors = ['k', 'b', 'g', 'r', 'c', 'm', 'y']
        for i, image in enumerate(content.images):
           image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
           h.append(image.shape[0])
           s.append(image.shape[1])
           v.append(image.shape[2])
           image_arr.append(image.reshape((h[-1]*s[-1], v[-1])))
           n, bins, patches = ax.hist(image_arr[-1].T[0], bins=255,density=True, alpha = 0.5, label=content.filenames[i])
           #for j, p in enumerate(patches):
           #    plt.setp(p,'facecolor',xh', edgecolor=edgecolors[i])

        ax.set_xlim(0,255.0)
        ax.set_ylim(0,.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

app = GuiApp()
app.mainloop()
