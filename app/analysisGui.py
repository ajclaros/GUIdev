from collections import deque
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import RectangleSelector
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
point_select = None
times = 0
bg=None
home_folder = os.getcwd()
class GuiApp(tk.Tk):


    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)
        self.nbins = 255
        self.panels = []
        self.images = []
        self.filenames = []
        self.foldernames = []
        self.percents = []
        self.values = []
        self.colorspace = None
        self.channel = None
        self.bin_range = False
        container = tk.Frame(self)
        container.grid(row=0, column=0)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        self.pages = [StartPage]
        for F in self.pages:

            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="new")

        self.show_frame(self.pages[0])

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()



class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        self.num_graphs = 0
        self.total_images = 0
        tk.Frame.__init__(self, parent)
        buttons = []
        checkboxes = []
        self.true_vals = []
        self.true_vals.append(tk.IntVar())
        self.true_vals.append(tk.IntVar())
        self.true_vals.append(tk.IntVar())
        self.true_vals.append(tk.IntVar())
        self.true_vals.append(tk.IntVar())
        self.true_vals.append(tk.IntVar())
        colorspace = []
        self.choice = tk.IntVar()
        nbins = tk.StringVar()
        nbins.set(controller.nbins)
        colorspace.append(tk.Radiobutton(self, text = "H", variable = self.choice, value = 0))
        colorspace.append(tk.Radiobutton(self, text = "S", variable = self.choice, value = 1))
        colorspace.append(tk.Radiobutton(self, text = "V", variable = self.choice, value = 2))
        colorspace.append(tk.Radiobutton(self, text = "L*", variable = self.choice, value = 3))
        colorspace.append(tk.Radiobutton(self, text = "A*", variable = self.choice, value = 4))
        colorspace.append(tk.Radiobutton(self, text = "B*", variable = self.choice, value = 5))
        colorspace.append(tk.Radiobutton(self, text = "B", variable = self.choice, value = 6))
        colorspace.append(tk.Radiobutton(self, text = "G", variable = self.choice, value = 7))
        colorspace.append(tk.Radiobutton(self, text = "R", variable = self.choice, value = 8))
        buttons.append(tk.Button(self, text="Quit",
                                  command = lambda: controller.destroy() ))
        #develop exporting function
        buttons.append(tk.Button(self, text='Export', command = lambda: [os.chdir('{}'.format(home_folder)),os.chdir('exports'), controller.percents.to_csv('percents.csv'), controller.values.to_csv('counts'),[x.savefig('{}.png'.format(x.figure.texts[0].get_text[0])) for x in controller.panels]]))

        buttons.append(tk.Button(self, text="Select Directory",
                                      command=lambda: self.select_directory(controller, known='no')))
        buttons.append(tk.Button(self, text='Color Range picker',
                                 command = lambda:self.color_picker(controller)))
        buttons.append(tk.Label(self, text='Num Bins: '))
        buttons.append(tk.Entry(self, textvariable=nbins))
        buttons.append(tk.Button(self, text='Set', command = lambda:[setattr(controller, 'nbins', nbins.get()), print('Set nbins to {}'.format(controller.nbins))]))
        

        checkboxes.append(ttk.Checkbutton(self, text= 'Histogram', variable = self.true_vals[1]))
        checkboxes.append(ttk.Checkbutton(self, text= 'Radarplot', variable = self.true_vals[2]))
        checkboxes.append(ttk.Checkbutton(self, text= 'Chi-squared', variable = self.true_vals[3]))
        checkboxes.append(ttk.Checkbutton(self, text= 'Kruskal Wallis', variable = self.true_vals[4]))
        checkboxes.append(ttk.Checkbutton(self, text= 'Kolmogorov-Smirnov', variable = self.true_vals[5]))
        for i, choice in enumerate(colorspace):
            choice.grid(row =2 +i//3, column= i%3)

        for i, box in enumerate(checkboxes):
            box.grid(row=1, column = i)
        for i, button in enumerate(buttons):
            button.grid(row=0, column=i)
    def getSelected(self):
        choice = self.choice.get()
        if choice//3 == 0:
            return ['HSV', choice%3]
        elif choice//3==1:
            return ["CIE L*A*B*", choice%3]
        else:
            return ['RGB', choice%3]



    def color_picker(self, content, repeat=0, points=None):
        global times, point_select
        times = repeat
        new_window = tk.Toplevel(content)
        buttons = []
        panels = []
        entries = []
        from_int = tk.IntVar()
        to_int = tk.IntVar()
        if points is not None:
            from_int.set(points[0])
            to_int.set(points[1])
        fig = Figure(figsize=(5,2.3), dpi=300)
        fig.subplots_adjust(top=0.7, bottom=0.4)
        ax = fig.add_subplot(111)
        cm = plt.cm.hsv
        cmaplist = [cm(i) for i in range(cm.N)]
        cmaplist = deque(cmaplist)
        gradient_ticks=np.arange(0, content.nbins, 5)
        gradient_ticks=deque(gradient_ticks)
        for i in range(times):
            cmaplist.rotate(30)
            gradient_ticks.rotate(6)

        cm = matplotlib.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, cm.N)
        bounds = np.arange(content.nbins)
        norm = matplotlib.colors.BoundaryNorm(bounds, cm.N)
        cb1 = matplotlib.colorbar.ColorbarBase(ax, cmap=cm, spacing='proportional',
                                               ticks=np.arange(0,content.nbins,5),
                                               boundaries= bounds, format='%1i', norm=norm,
                                               orientation='horizontal')
        cb1.ax.set_xticklabels(labels=[str(i) for i in gradient_ticks], fontsize=5, rotation=-70)
        panels.append(FigureCanvasTkAgg(fig, new_window))
        toggle_selector.RS = RectangleSelector(ax, line_select_callback, drawtype='box', useblit=True,
                                               button=[1, 3],
                                               minspanx=5, minspany=5,
                                               spancoords='pixels',
                                               interactive=True)
        fig.canvas.mpl_connect('key_press_event', toggle_selector)
        panels[-1].draw()
        panels[-1].get_tk_widget().grid(row=2,column=0, columnspan=10)
        buttons.append(tk.Button(new_window, text="quit",
                                  command = lambda: new_window.destroy() ))
        buttons.append(tk.Button(new_window, text='Rotate',
                                 command = lambda:[self.color_picker(content, repeat=times+1), new_window.destroy()]))
        buttons.append(tk.Button(new_window, text='Invert',
                                 command = lambda: [self.color_picker(content, repeat = repeat, points = [point_select[1], point_select[0]]), new_window.destroy(), set_points(point_select[1], point_select[0], fig, ax, toggle_selector.RS, times)]))
        buttons.append(tk.Button(new_window, text='Set Points', command = lambda:[self.color_picker(content, repeat=repeat, points = point_select), new_window.destroy()]))
        buttons.append(tk.Button(new_window, text='OK', command = lambda:[setattr(content, 'bin_range', True), new_window.destroy(), print('Bin_range={}'.format(content.bin_range))]))

        entries.append(tk.Label(new_window, text='From: '))
        entries.append(tk.Entry(new_window, textvariable = from_int))
        entries.append(tk.Label(new_window, text = 'To: '))
        entries.append(tk.Entry(new_window, textvariable = to_int))
        entries.append(tk.Button(new_window, text='use points',
                                 command = lambda: set_points(from_int.get(), to_int.get(), fig, ax, toggle_selector.RS, times)))
        for i, elt in enumerate(entries):
            elt.grid(row=1, column=i)
        for i, button in enumerate(buttons):
            button.grid(row=0, column=i)

    def run_analysis(self, content):
        space, channel  = self.getSelected()
        dispatcher = {"HSV":['H','S','V'], "CIE L*A*B*":['L*','A*','B*'],'RGB':['B','G','R']}
        content.colorspace = space
        content.channel = dispatcher[space][channel]
        print('{}: {}'.format(content.colorspace, content.channel))
        self.create_df(content, space=space, channel=channel )
        if self.true_vals[1].get()==True:
            self.create_histogram(content, space=space, channel=channel)
        if self.true_vals[2].get()==True:
            self.create_radarplot(content)
        if self.true_vals[3].get()==True:
            self.create_heatmap(content, func = 'Chi-squared')
        if self.true_vals[4].get()==True:
            self.create_heatmap(content, func =  'Kruskal-Wallis')
        if self.true_vals[5].get()==True:
            self.create_heatmap(content, func =  'Kolmorogov-Smirnov')

    def create_df(self, content, space=None, channel=None):
        global point_select
        percents = []
        values = []
        for folder_idx, folder in enumerate(content.images):
            for j, image in enumerate(folder):
                #bw = color.rgb2gray(image)
                #image = color.rgb2hsv(image)
                bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if space == 'LAB':
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
                    data = image[np.where(bw<255)].T[channel] #lum=0, A=1, B = 2
                elif space == 'RGB':
                    data = image[np.where(bw<255)].T[channel] # B=0, G=1, R=2
                else:
                    image = np.float32(image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    data = image[np.where(bw<255)].T[channel] #hue=0, sat=1, val = 2

                if content.bin_range==True:
                    if point_select[0]>point_select[1]:
                        data= data[(data>point_select[0])|(data<point_select[1])]
                    else:
                        content.nbins = point_select[1]-point_select[0]
                        data = data[(data>point_select[0] )& (data<point_select[1])]

                percents.append(np.histogram(
                   data, bins=content.nbins, range=(0, content.nbins), density=True)[0])
                values.append(np.histogram(
                   data, bins=content.nbins, density=False)[0])
                print('folder: {}, photo:({}/{})'.format(folder_idx, j+1, len(folder)))

        content.percents= pd.DataFrame(percents, index=content.filenames).T #This is what creates the final dataframe of hues to work with
        content.values = pd.DataFrame(values, index=content.filenames).T
        #print(content.values)

    def create_heatmap(self, content, func = None):
        dispatcher = {'Kruskal-Wallis':kruskal, 'Chi-squared':chisquare, 'Kolmorogov-Smirnov':kstest}
        if func == 'Chi-squared':
            correlation = content.values.corr(method=dispatcher[func])
        else:
            correlation = content.percents.corr(method=dispatcher[func])
        fig = plt.figure()
        ax = fig.add_subplot()
        #ax.set_title('KS-test')
        ax.set_xticks(ticks=np.arange(len(content.filenames)))
        ax.set_yticks(ticks=np.arange(len(content.filenames)))
        ax.set_xticklabels(labels=content.filenames, rotation=90, fontdict={'fontsize':4})
        ax.set_yticklabels(labels = content.filenames, fontdict = {'fontsize':4})
        ax.set_title("{}: {}, channel:{}".format(func, content.colorspace, content.channel))
        if func == 'Chi-squared':
            hm = ax.imshow(correlation, cmap='cool')
        else:
            hm = ax.imshow(correlation, cmap='Set1',interpolation = 'nearest')
        plt.colorbar(hm)
        content.panels.append(FigureCanvasTkAgg(fig, self))
        content.panels[-1].draw()
        content.panels[-1].get_tk_widget().grid(row=5 + (self.num_graphs//3),
                                                column = self.num_graphs % 3, pady=20, padx=15)
        self.num_graphs+=1


    def create_radarplot(self, content):
        folder1_idx = len(content.images[0])
        folders = [content.percents.T.iloc[:folder1_idx], content.percents.T.iloc[folder1_idx:]]
        fig, ax = plt.subplots(1, 2, subplot_kw=dict(polar=True))
        cm = plt.cm.hsv
        for i, folder in enumerate(folders):
            data = folder.mean().values
            data = data/data.sum()
            theta = folder.mean().index
            r = folder.mean().values
            colors = theta
            ax[i].scatter(np.interp(theta,(theta.min(), theta.max()),(0, 2*3.14)), r, c= colors, cmap='hsv')
        fig.suptitle("Radarplots of {}:{}, folder 1 and 2".format(content.colorspace, content.channel))
        content.panels.append(FigureCanvasTkAgg(fig, self))
        content.panels[-1].draw()
        content.panels[-1].get_tk_widget().grid(row=5 + (self.num_graphs//2),
                                                column = self.num_graphs % 2, pady=20, padx=15)
        self.num_graphs+=1
    def create_histogram(self, content, space=None, channel=None):
        folder1_idx = len(content.images[0])
        folders = [content.percents.T.iloc[:folder1_idx], content.percents.T.iloc[folder1_idx:]]
        fig = plt.figure()
        #fig = plt.figure(figsize=(20,10))

        ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])
        ax1 = fig.add_axes([0.05, 0.1, 0.9, 0.1])
        if space=='CIE L*A*B*':
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
        cmaplist = [cm(i) for i in range(cm.N)]
        cmaplist = deque(cmaplist)
        if space=='HSV' and channel==0:
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
            if space== 'HSV' and channel==1:
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
        ax.set_title("{}, channel:{}".format(content.colorspace, content.channel))
        #ax.set_xlim(indices[0],indices[-1])
        #ax1.set_xlim(indices[0], indices[-1])
        #plt.tight_layout()
#       ax.set_ylim(0, 0.0002)
        content.panels.append(FigureCanvasTkAgg(fig, self))
        content.panels[-1].draw()
        content.panels[-1].get_tk_widget().grid(row=5 + (self.num_graphs//3),
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
    a= a*1e5
    b= b*1e5
    chisq , pval = stats.chisquare(f_obs=a.flatten(), f_exp= b.flatten())
    return pval

def kstest(a, b):
    D, p = stats.kstest(a,b)
    return p

def line_select_callback(eclick, erelease):
    """
    Callback for line selection.

    *eclick* and *erelease* are the press and release events.
    """
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    global point_select, times
    toggle_selector.RS.extents = x1, x2, 0, 250
    point_select=(np.ceil(x1-times*30)%255, np.ceil(x2-times*30)%255)
    print('from:{}, to:{}'.format(point_select[0],point_select[1]))

def toggle_selector(event):
    if event.key == 't':
        if toggle_selector.RS.active:
            print(' RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
        else:
            print(' RectangleSelector activated.')
            toggle_selector.RS.set_active(True)

def set_points(from_x, to_x, fig, ax, RS, times):
    global point_select

    temp =(np.ceil(from_x+times*30)%255, np.ceil(to_x+times*30)%255, 0, 255)
    if temp[0]>temp[1]:
        RS.extents = 0, temp[1], 0,255
        RS2 = RectangleSelector(ax, line_select_callback, drawtype='box', useblit=True,
                                               button=[1, 3],
                                               minspanx=5, minspany=5,
                                               spancoords='pixels',
                                               interactive=True)
        RS2.extents = from_x, 255, 0, 255
    else:
        RS.extents = temp[0], temp[1], 0, 255

    point_select = (from_x, to_x)
    print('from:{}, to:{}'.format(point_select[0],point_select[1]))

def setter(a, b):
    a = b


app = GuiApp()
app.mainloop()
