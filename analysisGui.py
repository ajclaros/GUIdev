from collections import deque
import seaborn as sns
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
        self.nbins = 255
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
    def __init__(self, parent, controller):

        self.num_graphs = 0
        self.total_images = 0
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="First Page", font=LARGE_FONT)
        label.grid(row=0, column=0, pady=10, padx=10)
        buttons = []
        checkboxes = []
        self.true_vals = []
        self.true_vals.append(tk.IntVar())
        self.true_vals.append(tk.IntVar())
        self.true_vals.append(tk.IntVar())
        self.true_vals.append(tk.IntVar())
        self.true_vals.append(tk.IntVar())
        self.true_vals.append(tk.IntVar())

        buttons.append(tk.Button(self, text="quit",
                                  command = lambda: controller.destroy() ))
        buttons.append(tk.Button(self, text="Select Directory",
                                      command=lambda: self.select_directory(controller, known='yes')))
#        buttons.append(ttk.Button(self, text="Select Image",
#                                      command=lambda: self.select_image(controller, self.total_images)))
        #buttons.append(tk.Button(self, text="Color Histogram",
        #                          command = lambda: self.color_histogram(controller)))
        buttons.append(tk.Button(self, text='Color Range picker',
                                 command = lambda:self.color_picker(controller)))
        checkboxes.append(ttk.Checkbutton(self, text= 'H',  variable =self.true_vals[0]))
        checkboxes.append(ttk.Checkbutton(self, text= 'Histogram', variable = self.true_vals[1]))
        checkboxes.append(ttk.Checkbutton(self, text= 'Radarplot', variable = self.true_vals[2]))
        checkboxes.append(ttk.Checkbutton(self, text= 'Chi-squared', variable = self.true_vals[3]))
        checkboxes.append(ttk.Checkbutton(self, text= 'Kruskal Wallis', variable = self.true_vals[4]))
        checkboxes.append(ttk.Checkbutton(self, text= 'Kolmogorov-Smirnov', variable = self.true_vals[5]))
        for i, box in enumerate(checkboxes):
            box.grid(row=1, column = i)
        for i, button in enumerate(buttons):
            button.grid(row=0, column=i)

    def color_picker(self, content):
        new_window = tk.Toplevel(content)
        panels = []
        fig = plt.figure()
        ax1 = fig.add_axes()

        cm = plt.cm.hsv
        cmaplist = [cm(i) for i in range(cm.N)]
        cmaplist = deque(cmaplist)
        cmaplist.rotate(30)
        cmaplist.reverse()
        cm = matplotlib.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, cm.N)
        bounds = np.arange(content.nbins)
        norm = matplotlib.colors.BoundaryNorm(bounds, cm.N)
        cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cm, spacing='proportional',
                                               ticks=np.arange(0,content.nbins,5),
                                               boundaries= bounds, format='%1i', norm=norm,
                                               orientation='horizontal')

        panels.append(FigureCanvasTkAgg(fig, self))
        panels[-1].draw()
        panels[-1].get_tk_widget().grid(row=2 + (self.num_graphs//3),
                                                column = self.num_graphs % 3, pady=20, padx=15)
    def run_analysis(self, content):
        if self.true_vals[0]:
            self.create_df(content)
        if self.true_vals[1]:
            self.create_histogram(content)
        if self.true_vals[2]:
            self.create_radarplot(content)
        if self.true_vals[3]:
            self.create_heatmap(content, func = chisquare)
        if self.true_vals[4]:
            self.create_heatmap(content, func =  kruskal)
        if self.true_vals[5]:
            self.create_heatmap(content, func =  kstest)

    def create_df(self, content, lab=False, channel=None):
        for folder_idx, folder in enumerate(content.images):
            for j, image in enumerate(folder):
                #bw = color.rgb2gray(image)
                #image = color.rgb2hsv(image)
                bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if lab == True:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
                    data = image[np.where(bw<255)].T[channel] #lum=0, A=1, B = 2
                else:
                    image = np.float32(image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    data = image[np.where(bw<255)].T[0] #hue=0, sat=1, val = 2
                content.hues.append(np.histogram(
                   data, bins=content.nbins, density=True)[0])
                print('folder: {}, photo:({}/{})'.format(folder_idx, j+1, len(folder)))

        content.hues= pd.DataFrame(content.hues, index=content.filenames).T #This is what creates the final dataframe of hues to work with
        print(content.hues)

    def create_heatmap(self, content, func = None):
        correlation = content.hues.corr(method=func)
        fig = plt.figure()
        ax = fig.add_subplot()
        #ax.set_title('KS-test')
        ax.set_xticks(ticks=np.arange(len(content.filenames)))
        ax.set_yticks(ticks=np.arange(len(content.filenames)))
        ax.set_xticklabels(labels=content.filenames, rotation=90, fontdict={'fontsize':4})
        ax.set_yticklabels(labels = content.filenames, fontdict = {'fontsize':4})
        hm = ax.imshow(correlation, cmap='Set1',interpolation = 'nearest')
        plt.colorbar(hm)
        content.panels.append(FigureCanvasTkAgg(fig, self))
        content.panels[-1].draw()
        content.panels[-1].get_tk_widget().grid(row=2 + (self.num_graphs//3),
                                                column = self.num_graphs % 3, pady=20, padx=15)
        self.num_graphs+=1


    def create_radarplot(self, content):
        folder1_idx = len(content.images[0])
        folders = [content.hues.T.iloc[:folder1_idx], content.hues.T.iloc[folder1_idx:]]
        fig, ax = plt.subplots(1, 2, subplot_kw=dict(polar=True))
        cm = plt.cm.hsv
        for i, folder in enumerate(folders):
            data = folder.mean().values
            data = data/data.sum()
            theta = folder.mean().index
            r = folder.mean().values
            colors = theta
            ax[i].scatter(np.interp(theta,(theta.min(), theta.max()),(0, 2*3.14)), r, c= colors, cmap='hsv')
        content.panels.append(FigureCanvasTkAgg(fig, self))
        content.panels[-1].draw()
        content.panels[-1].get_tk_widget().grid(row=2 + (self.num_graphs//2),
                                                column = self.num_graphs % 2, pady=20, padx=15)
        self.num_graphs+=1
    def create_histogram(self, content, lab=False, channel=1):
        folder1_idx = len(content.images[0])
        folders = [content.hues.T.iloc[:folder1_idx], content.hues.T.iloc[folder1_idx:]]
        fig = plt.figure()
        #fig = plt.figure(figsize=(20,10))

        ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])
        ax1 = fig.add_axes([0.05, 0.1, 0.9, 0.1])
        if lab==True:
            cm= plt.cm.cool if channel==1 else plt.cm.cividis
            indices = np.arange(content.nbins)
        else:
            cm = plt.cm.hsv
            foldersum = folders[0].mean()+folders[1].mean()
            foldersum = foldersum/foldersum.sum()
            foldersum = deque(foldersum)
            foldersum.rotate(30)
            foldersum.reverse()
            foldersum = np.array(foldersum)

            indices = np.where(foldersum>0.005)
            indices = indices[0]
            print(indices)
        cmaplist = [cm(i) for i in range(cm.N)]
        cmaplist = deque(cmaplist)
        if lab==False:
            cmaplist.rotate(30)
            cmaplist.reverse()
        cm = matplotlib.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, cm.N)
        bounds = np.arange(content.nbins)
        norm = matplotlib.colors.BoundaryNorm(bounds, cm.N)

        cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cm, spacing='proportional', ticks=np.arange(0,content.nbins,5), boundaries= bounds, format='%1i', norm=norm, orientation='horizontal')
        folder_colors = ['b','y']
        for i, folder in enumerate(folders):
            data = folder.mean().values
            data = deque(data)
            if lab==False:
                data.rotate(30)
                data.reverse()
            data = np.array(data)
            data = data/data.sum()
            #y, bin_edges = np.histogram(data, bins = 100)
            #bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
            err = stats.sem(folder)
            label = content.foldernames[i]
            label = label.split('/')
            label = label[-1]
            ax.plot(np.arange(data.size), data, label = label, color=folder_colors[i], linewidth=2)
            ax.errorbar(np.arange(data.size), data, yerr = err, fmt = 'o')
        ax.legend()
        ax.set_xlim(indices[0],indices[-1])
        ax1.set_xlim(indices[0], indices[-1])
        #plt.tight_layout()
#       ax.set_ylim(0, 0.0002)
        content.panels.append(FigureCanvasTkAgg(fig, self))
        content.panels[-1].draw()
        content.panels[-1].get_tk_widget().grid(row=2 + (self.num_graphs//3),
                                                column = self.num_graphs % 3, pady=20, padx=15)
        self.num_graphs+=1



    def select_image(self, content, gridx, files):
        path = files
        name= path.split('/')
        name = name[-1].split('.')
        name = name[0]
        content.filenames.append(name)
        if len(path) > 0:
            image = cv2.imread(path)
            imagefile = image
            content.images[-1].append(imagefile)
            self.total_images += 1

    def select_directory(self, content, known='yes'):
        if known == 'yes':
            folders = ['/home/claros/Dropbox/patternize/b'
                       ,'/home/claros/Dropbox/patternize/y']
            for i, folder in enumerate(folders):
                print('--------------------------------')
                print('Loading Folder: {} done'.format(i))

                os.chdir(folder)
                content.images.append([])
                content.foldernames.append(folder)
                for files in os.listdir():
                    self.select_image(content, self.total_images, files)
                print('{} images uploaded'.format(len(os.listdir())))
                self.total_images = 0
        else:
            folder = filedialog.askdirectory()
            content.foldernames.append(folder)
            content.images.append([])
            os.chdir(folder)
            for i, files in enumerate(os.listdir()):
                self.select_image(content, self.total_images, files)
                print('loading: photo ({}/{})'.format(i+1, len(os.listdir())))
            self.total_images = 0
        if len(content.foldernames)==2:
            self.run_analysis(content)

def kruskal(a, b):
        statistic, pval = stats.kruskal(a,b)
        return pval

def chisquare(a, b):
    chisq , pval = stats.chisquare(a, f_exp= b)
    return pval
def kstest(a, b):
    D, p = stats.kstest(a,b)
    return p

app = GuiApp()
app.mainloop()
