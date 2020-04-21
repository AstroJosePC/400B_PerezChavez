
# # In Class Lab 10 : Template File
# 
# Tutorial to make some interesting plots with widgets and the simulaton data ! 
# 
# 
# Graphical widgets -- helpful functions to make a "graphical user interface", or GUI.
# 
# These widgets need to be able to take input from the mouse and keyboard while the program is running. The most common way this is achieved is to have the code run in an infinite loop which is interrupted whenever input is provided. Some action is taken according to the input, and then the loop starts running again. This sort of algorithm is known as an *event loop* -- the code loops until a user event occurs.
# 
# `matplotlib` provides a number of simple widgets which automatically create an event loop for us. One can create a widget instance, and then tell the widget what function to run when something happens to the widget. Such a function is called a *callback* -- the event loop calls back to the function we give it in order to take some action before starting up again.




import matplotlib.widgets as mw  # get access to the widgets


# external modules
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
import numpy as np

# my modules 
from ReadFile import Read
from CenterOfMass2 import CenterOfMass
from MassProfile import MassProfile

# I took the code from Lab 7 for making contours and made it into a separate script
# NOTE: it is more organized to keep functions in separate scripts 
# and then call them when you want to e.g. make plots or do some analysis. 
from contour import density_contour


# # Part A. Load in Data and make some simple plots
# 
# To do this lab you will need to sftp into nimoy to get the highres data files for this lab:
# MW_000.txt and MW_400.txt
# If you don't have enough space on your computer you can use the low res files. 
# 

# In[ ]:


# Load in disk particles centered on the MW
# this is from the HighRes data files on nimoy so it might take a bit of time to load
# you can use the low res files if this is taking too long or if you don't have enough space on your computer
COM = CenterOfMass("MW_000.txt",2)


# In[ ]:


# Compute COM of the MW at the new position using disk particles
COMP = COM.COM_P(0.1, 2)
COMV = COM.COM_V(COMP[0],COMP[1],COMP[2])
# Determine positions of disk particles relative to COM 
MW_Disk_x = COM.x - COMP[0].value 
MW_Disk_y = COM.y - COMP[1].value 

# Also store the disk velocity in the y direction
MW_Disk_vy = COM.vy - COMV[1].value


# In[ ]:


# Plot the disk of the MW with contours. 


# MW Disk Density 
fig, ax= plt.subplots(figsize=(10, 10))

## ADD HERE
# plot the particle density for MW using plt.hist2d 
# can modify bin number (e.g. bin =100 for low res files)

# note: MW_Disk.x and MW_Disk.y won't be exactly at 0,0 because i was lazy and didn't take out the center of mass pos

#### ADD HERE 
# call density_contour to add contours
# density_contour(x pos, y pos, contour res, contour res, axis, colors=[colors,colors])




# Add axis labels
plt.xlabel('x (kpc)', fontsize=22)
plt.ylabel('y (kpc)', fontsize=22)

#set axis limits
plt.ylim(-30,30)
plt.xlim(-30,30)

#adjust tick label font size
label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size

plt.show()

# # Part B  Zooming in on a plot with widgets
# 
# We can catch characters typed on the keyboard -- *keypress events* -- by connecting a "key_press_event" to a callback function which takes an event as an argument.
# The event object contains a variety of data. The most useful being:
# 
#     event.key       # the key which was pressed
#     event.xdata     # the mouse x-position when the key was pressed
#     event.ydata     # the mouse y-position when the key was pressed
#     
# Another useful widget allows the user to select a rectangular region in some axes object, and then calls a callback function with the bounding coordinates (the extent) of the region selected. This is the RectangleSelector widget.
# 
# Note that click and release are not really that! Click contains the more-negative values and release the more positive values of both x and y coordinates.

# In[ ]:



def callbackRectangle( click, release ): # the events are click and release
    print( f"button {click.button} pressed" )
    print( f"button {release.button} released" )
    extent = [ click.xdata, release.xdata, click.ydata, release.ydata ]
    print( f"box extent is {extent}") 
    
    # ADD - in order to zoom in reset the axes to the clicked box size
   

    # Save the file

    # add the ability to reset the image using an "on key press" function 
def onKeyPressed(event):
    
    if event.key in ['R', 'r']:
        # ADD - to zoom back out reset the axes
        


# In[ ]:


# plot the particle density for the MW Disk and then zoom in on a region of the disk 

fig, ax = plt.subplots(figsize =(10 ,10))                             

# 2d histogram 
plt.hist2d(MW_Disk_x,MW_Disk_y, bins=300, norm=LogNorm(), cmap='magma')
plt.colorbar(label='Number  of  stars  per  bin')

# over plot contours
density_contour(MW_Disk_x, MW_Disk_y, 80, 80, ax=ax,                 colors=['white'])
   
    
## NEW: Rectangle Selector.     
rs = mw.RectangleSelector( ax,                        # the axes to attach to
                           callbackRectangle,         # the callback function
                           drawtype='box',            # draw a box when selecting a region
                           button=[1, 3],             # allow us to use left or right mouse button
                                                      #button 1 is left mouse button
                           minspanx=5, minspany=5,    # don't accept a box of fewer than 5 pixels
                           spancoords='pixels' )      # units for above


#set axis limits
ax.set_ylim(-30,30)
ax.set_xlim(-30,30)


# Add axis labels
plt.xlabel('x (kpc)', fontsize=22)
plt.ylabel('y (kpc)', fontsize=22)

# ADDED THIS
# to detect the 'R' key press to reset the image
plt.connect("key_press_event", onKeyPressed)


plt.show()

# # Part C    Connecting Morphology to Kinematics
# 
# 
# Make a two panel plot.
# Left Panel:  Density 
# Right Panel: Phase Diagram 
# 
# Relect a section of the density plot and see where the particles are on the phase diagram

# In[ ]:


# Let's store the circular velocity of the MW like we did in Lab 7

# ADD MassProfile Object.


# In[ ]:


# Add an array for radii up to 40 kpc


# Store Vcirc 


# In[ ]:


# Step 1) Copy over the call back function and the onkeypressed function

# Step 3) Let figure out how to select a region in the density and plot it also in the right panel
# We also don't want to zoom in on the left panel, instead let's just mark the region we examined. 



# In[ ]:


# Step 2) 
# Add not just the density but also the phase diagram as a separate panel.
# Copy over the plotting code (2D histogram) and modify the figure so that there are now two panels.

# Add a phase diagram: X vs VY



# # Part D:  Flip it around 
# 
# Now Pick based on kinematics and find out where they are in the disk.
# This would be a good way to find e.g. high velocity particles. or particles that are really not obeying the normal kinematics of the disk at the current time.

# In[ ]:


# Copy over the Call back function and the onkeypressed function from Part C
# flip the axes ax[0] < --- > ax[1]


# In[ ]:


# Copy over the Density and phase diagram code
# flip the axes ax[0]<--> ax[1]



# # Part E : Connecting particles across snapshots
# 

# In[ ]:

### UNCOMMENT OUT THE BELOW LINES

#COM_2 = CenterOfMass("MW_400.txt",2) # Load in a different snapshot
#COMP_2 = COM_2.COM_P(0.1, 2) # Compute COM of the MW at the new position using disk particles
#MW_Disk_2_x = COM_2.x - COMP_2[0].value  # Determine positions of disk particles relative to COM 
#MW_Disk_2_y = COM_2.y - COMP_2[1].value 


# In[ ]:


# Copy over the Call back function and the onkeypressed function from Part C


# In[ ]:


# Copy over the plotting script from Part C
# Instead of the phase plot, have the second panel be the MW at a different snapshot


# In[ ]:




