from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import tkinter as tk
from tkinter import ttk
import numpy as np
import os
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog
import pandas as pd
from scipy import stats
LARGE_FONT= ("Verdana", 12)


class GuiApp(tk.Tk):

    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)
        self.panels = []
        self.images = []
        self.filenames = []
        self.foldernames = []
        self.hues = []
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
    total_folders = 0
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="First Page", font=LARGE_FONT)
        label.grid(row=0, column=0, pady=10, padx=10)
        buttons = []
        buttons.append(ttk.Button(self, text="Select Directory",
                                      command=lambda: self.select_directory(controller, known='yes')))
#        buttons.append(ttk.Button(self, text="Select Image",
#                                      command=lambda: self.select_image(controller, self.total_images)))
        buttons.append(ttk.Button(self, text="Analysis page", command=lambda: controller.show_frame(Analysis)))
        buttons[0].grid(row=2, column=0)
        buttons[1].grid(row=2,column=1)

    def select_directory(self, content, known='yes'):
        if known == 'yes':
            folders = ['/home/claros/Dropbox/patternize/blueyellow'
                       ,'/home/claros/Dropbox/patternize/6.15.20']
            for folder in folders:
                os.chdir(folder)
                content.images.append([])
                content.foldernames.append(folder)
                for files in os.listdir():
                    self.select_image(content, self.total_images, files)
                self.total_images = 0
                self.total_folders +=1
        else:
            folder = filedialog.askdirectory()
            content.filenames.append(folder)
            content.images.append([])
            os.chdir(folder)
            for files in os.listdir():
                self.select_image(content, self.total_images, files)
            self.total_images = 0
            self.total_folders += 1

    def select_image(self, content, gridx, files):
        path = files
        name=path.split('/')
        name = name[-1].split('.')
        name = name[0]
        content.filenames.append(name)
        if len(path) > 0:
            image = cv2.imread(path)
            imagecv = image
            content.images[-1].append(imagecv)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = image.resize((200, 200), Image.ANTIALIAS)
            image = ImageTk.PhotoImage(image)
            content.panels.append(tk.Label(image=image))
            image_idx = len(content.panels)-1
            content.panels[image_idx].image = image
            content.panels[image_idx].grid(row = 1 + self.total_folders,
                                        column = gridx,
                                        padx = 20,
                                        pady = 10)
            self.total_images += 1


class Analysis(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="Analysis")
        label.grid(row=0, column=0, pady=10, padx=10)
        buttons = []
        buttons.append(ttk.Button(self, text= "First page",
                                  command = lambda: controller.show_frame(StartPage)))
        buttons.append(ttk.Button(self, text="Color Histogram",
                                  command = lambda: self.color_histogram(controller)))
        buttons.append(ttk.Button(self, text="show folders",
                                  command = lambda: self.hue_arr(controller)))
        buttons[0].grid(row=0, column=0)
        buttons[1].grid(row=0, column=1)
        buttons[2].grid(row=0, columns=4)
        h = []
        s = []
        v = []
    def hue_arr(self, content):
        image_arr = []
        h = []
        s = []
        v = []
        image_idx=0
        for folder_idx, folder in enumerate(content.images):
           for j, image in enumerate(folder):
               image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
               h.append(image.shape[0])
               s.append(image.shape[1])
               v.append(image.shape[2])
               reshaped_im = image.reshape((h[-1]*s[-1],v[-1]))
               content.hues.append(np.histogram(
                  reshaped_im.T[0], bins=100)[0])
               #content.hues.append(binned_statistic(reshaped_im.T[0], np.arange(0,360,1)))
               image_idx += 1
        content.hues= pd.DataFrame(content.hues, index=content.filenames).T
        #print(content.hues)
        correlation = content.hues.corr(method=chisquare)
        fig = plt.figure(figsize=(20,20))
        ax=fig.add_subplot()

        ax.set_title('Kruskal Wallice')
        ax.set_xticks(ticks=np.arange(len(content.filenames)))
        ax.set_yticks(ticks=np.arange(len(content.filenames)))
        ax.set_xticklabels(labels=content.filenames, rotation=90, fontdict={'fontsize':4})
        ax.set_yticklabels(labels= content.filenames, fontdict={'fontsize':4})
        hm = ax.imshow(correlation, cmap='winter',interpolation='nearest')
        plt.colorbar(hm)
        plt.show()



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

        folder = content.images[0]
        for i, image in enumerate(folder):
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
        plt.show()
def chisquare(a, b):
    statistic, pval = stats.kruskal(a,b)

    return pval
app = GuiApp()
app.mainloop()
