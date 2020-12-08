# Gui for Quantification of color
> In early stages, visualizations are from test data and not representative of a final visualization. The main useful file is colorRange.py where all other files are test functions

Processes folders of photos and creates a dataframe for analysis. Current implementation works with hue, but switching to any other value is not difficult
Two visualizations are: Heatmap of a Kruskall Wallice test comparing individual photos and a histogram of binned hues by folder.
![](screenshot.png)
![](first_photo.png)
![](cielab_000.png)


Added color range selection
Select by:
- Direct values
- Dragging window
- Inverting selection
- Rotating color bar to center specified region
![](colorselection.gif)


