from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
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
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
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
        self.pages = [StartPage]
        for F in self.pages:

            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(self.pages[0])

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
        buttons.append(ttk.Button(self, text="Color Histogram",
                                  command = lambda: self.color_histogram(controller)))
        buttons.append(ttk.Button(self, text="show folders",
                                  command = lambda: self.hue_arr(controller)))
        for i, button in enumerate(buttons):
            button.grid(row=0, column=i)

    def create_df(self, content):
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
                image_idx += 1
        content.hues= pd.DataFrame(content.hues, index=content.filenames).T #This is what creates the final dataframe of hues to work with

    def create_heatmap(self, content):
        correlation = content.hues.corr(method=kruskal)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_title('Kruskal Wallice')
        ax.set_xticks(ticks=np.arange(len(content.filenames)))
        ax.set_yticks(ticks=np.arange(len(content.filenames)))
        ax.set_xticklabels(labels=content.filenames, rotation=90, fontdict={'fontsize':4})
        ax.set_yticklabels(labels = content.filenames, fontdict = {'fontsize':4})
        hm = ax.imshow(correlation, cmap='winter',interpolation = 'nearest')
        plt.colorbar(hm)
        content.panels.append(FigureCanvasTkAgg(fig, self))
        content.panels[-1].draw()
        content.panels[-1].get_tk_widget().grid(row=1, column=0, pady = 20)




    def create_histogram(self, content):
        folder1_idx = len(content.images[0])
        folders = [content.hues.T.iloc[:folder1_idx], content.hues.T.iloc[folder1_idx:]]
        fig = plt.figure()
        ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])
        ax1 = fig.add_axes([0.05, 0.1, 0.9, 0.1])
        norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
        cm = plt.cm.hsv
        cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cm, norm=norm, orientation='horizontal')
        for i, folder in enumerate(folders):
            print(folder)
            data = folder.mean().values
            y, bin_edges = np.histogram(data, bins = 100, density = True)
            bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
            menStd = np.std(y)
            label = content.foldernames[i]
            label = label.split('/')
            label = label[-1]
            ax.plot(bin_centers, y, label = label)
            ax.errorbar(bin_centers, y, yerr = menStd, fmt = 'o')
        ax.legend()
        ax.set_ylim(0, 0.0002)
        content.panels.append(FigureCanvasTkAgg(fig, self))
        content.panels[-1].draw()
        content.panels[-1].get_tk_widget().grid(row=2, column=0, pady = 20)


#        n, bins, patches = ax.hist(
#            image_arr[-1].T[0], bins=255, density=True, alpha=0.5, label=content.filenames[i])
#
#        ax.set_xlim(0,255.0)
#        ax.set_ylim(0,.3)
#        ax.legend()
#        plt.show()
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
        else:
            folder = filedialog.askdirectory()
            content.foldernames.append(folder)
            content.images.append([])
            os.chdir(folder)
            for files in os.listdir():
                self.select_image(content, self.total_images, files)
            self.total_images = 0
        if len(content.foldernames)==2:
            self.create_df(content)
            self.create_heatmap(content)
            self.create_histogram(content)


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
            self.total_images += 1


def kruskal(a, b):
    statistic, pval = stats.kruskal(a,b)
    return pval

def chisquare(a, b):
    statistic, pval = stats.chisquare(a,b)
    return pval

app = GuiApp()
app.mainloop()
