from numpy import pi, sin, cos
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
import math

def circle_x_sine(radius, c1, c2):
    return radius * (1 + c1 * cos(4 * rads) + c2 * cos(8 * rads))

rads = np.arange(0, (2 * np.pi), 0.01)

def compute_length(radius, c1, c2):
    length = 0
    radii = circle_x_sine(radius, c1, c2)
    r1 = radii[0]
    r2 = radii[0]
    for i in range(len(rads)-1):
        r1 = r2
        r2 = radii[i+1]
        length = length + np.sqrt(r1 * r1 + r2 * r2 - 2 * r1 * r2 * cos(0.01))
    return length

def compute_and_format_length(radius, c1, c2):
    return "{:.2f}".format(compute_length(radius, c1, c2))

def find_best_radius(radius, c1, c2, target):
    cur_length = compute_length(radius, c1, c2)
    prev_length = cur_length
    cur_radius = radius
    prev_radius = radius
    in_between = False
    #We iterate until we find two radii that make lengths such that the target is in between
    while not in_between:
        if prev_length > target:
            cur_radius = prev_radius/2
            cur_length = compute_length(cur_radius, c1, c2)
            if cur_length < target:
                in_between = True
            else:
                prev_length = cur_length
                prev_radius = cur_radius
        else:
            cur_radius = prev_radius*2
            cur_length = compute_length(cur_radius, c1, c2)
            if cur_length > target:
                in_between = True
            else:
                prev_length = cur_length
                prev_radius = cur_radius
    #we reorganize the lengths
    small_radius = min(prev_radius, cur_radius)
    big_radius = max(prev_radius, cur_radius)
    #divide the interval, keep interval containing target, continue until new length is close enough to target
    while (True):
        new_radius = (big_radius + small_radius)/2
        new_length = compute_length(new_radius, c1, c2)
        if abs(new_length - target) < 1e-2:
            return new_radius
        elif new_length < target:
            small_radius = new_radius
        else:
            big_radius = new_radius


axis_color = 'lightgoldenrodyellow'

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')

# Adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.25, bottom=0.25)

c1_0 = 0
c2_0 = 0
rad_0 = 1

# Draw the initial plot
# The 'line' variable is used for modifying the line later
[line] = ax.plot(rads, circle_x_sine(rad_0, c1_0, c2_0), linewidth=2, color='red')
ax.set_ylim([0, 5])
# Add two sliders for tweaking the parameters

# Define an axes area and draw a slider in it
c1_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axis_color)
c1_slider = Slider(c1_slider_ax, 'c1', -1, 1.0, valinit=c1_0)

# Draw another slider
c2_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor=axis_color)
c2_slider = Slider(c2_slider_ax, 'c2', -1, 1, valinit=c2_0)

# Draw another slider
radius_slider_ax = fig.add_axes([0.25, 0.05, 0.65, 0.03], facecolor=axis_color)
radius_slider = Slider(radius_slider_ax, 'radius', 0.1, 10, valinit=rad_0)

#length display
length_ax = fig.add_axes([0.1, 0.5, 0.15, 0.15])
length_text_box = TextBox(length_ax, 'length', initial = compute_and_format_length(rad_0, c1_0, c2_0))

# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
    line.set_ydata(circle_x_sine(radius_slider.val, c1_slider.val, c2_slider.val))
    length_text_box.set_val(compute_and_format_length(radius_slider.val, c1_slider.val, c2_slider.val))
    fig.canvas.draw_idle()
c1_slider.on_changed(sliders_on_changed)
c2_slider.on_changed(sliders_on_changed)
radius_slider.on_changed(sliders_on_changed)

# Add a button for resetting the parameters
reset_button_ax = fig.add_axes([0.8, 0., 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')
def reset_button_on_clicked(mouse_event):
    c2_slider.reset()
    c1_slider.reset()
    radius_slider.reset()
reset_button.on_clicked(reset_button_on_clicked)

#target display
target_ax = fig.add_axes([0.1, 0.4, 0.15, 0.15])
target_text_box = TextBox(target_ax, 'target', initial = "6.28")
opt_ax = fig.add_axes([0.1, 0.3, 0.15, 0.15])
opt_button = Button(opt_ax, 'optimize', color=axis_color)
def opt_button_on_clicked(mouse_event):
    radius_slider.set_val(find_best_radius(radius_slider.val, c1_slider.val, c2_slider.val, float(target_text_box.text)))
opt_button.on_clicked(opt_button_on_clicked)

plt.show()
