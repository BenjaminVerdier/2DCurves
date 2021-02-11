from numpy import pi, sin, cos
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
import math

def circle_x_sine(radius, amp, freq):
    return radius * (1 + amp * sin(rads * np.floor(freq)))

rads = np.arange(0, (2 * np.pi), 0.01)

def compute_length(radius, amp, freq):
    length = 0
    radii = circle_x_sine(radius, amp, freq)
    r1 = radii[0]
    r2 = radii[0]
    for i in range(len(rads)-1):
        r1 = r2
        r2 = radii[i+1]
        length = length + np.sqrt(r1 * r1 + r2 * r2 - 2 * r1 * r2 * cos(0.01))
    return length

def compute_and_format_length(radius, amp, freq):
    return "{:.2f}".format(compute_length(radius, amp, freq))

def find_best_radius(radius, amp, freq, target):
    cur_length = compute_length(radius, amp, freq)
    prev_length = cur_length
    cur_radius = radius
    prev_radius = radius
    in_between = False
    #We iterate until we find two radii that make lengths such that the target is in between
    while not in_between:
        if prev_length > target:
            cur_radius = prev_radius/2
            cur_length = compute_length(cur_radius, amp, freq)
            if cur_length < target:
                in_between = True
            else:
                prev_length = cur_length
                prev_radius = cur_radius
        else:
            cur_radius = prev_radius*2
            cur_length = compute_length(cur_radius, amp, freq)
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
        new_length = compute_length(new_radius, amp, freq)
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

amp_0 = 0
freq_0 = 0
rad_0 = 1

# Draw the initial plot
# The 'line' variable is used for modifying the line later
[line] = ax.plot(rads, circle_x_sine(rad_0, amp_0, freq_0), linewidth=2, color='red')
ax.set_ylim([0, 5])
# Add two sliders for tweaking the parameters

# Define an axes area and draw a slider in it
amp_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axis_color)
amp_slider = Slider(amp_slider_ax, 'Amp', -1, 1.0, valinit=amp_0)

# Draw another slider
freq_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor=axis_color)
freq_slider = Slider(freq_slider_ax, 'Freq', 0.1, 30.0, valinit=freq_0, valfmt='%0.0f')

# Draw another slider
radius_slider_ax = fig.add_axes([0.25, 0.05, 0.65, 0.03], facecolor=axis_color)
radius_slider = Slider(radius_slider_ax, 'Radius', 0.1, 10, valinit=rad_0)

#length display
length_ax = fig.add_axes([0.1, 0.5, 0.15, 0.15])
length_text_box = TextBox(length_ax, 'Length', initial = compute_and_format_length(rad_0, amp_0, freq_0))

# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
    line.set_ydata(circle_x_sine(radius_slider.val, amp_slider.val, freq_slider.val))
    length_text_box.set_val(compute_and_format_length(radius_slider.val, amp_slider.val, freq_slider.val))
    fig.canvas.draw_idle()
amp_slider.on_changed(sliders_on_changed)
freq_slider.on_changed(sliders_on_changed)
radius_slider.on_changed(sliders_on_changed)

# Add a button for resetting the parameters
reset_button_ax = fig.add_axes([0.8, 0., 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')
def reset_button_on_clicked(mouse_event):
    freq_slider.reset()
    amp_slider.reset()
    radius_slider.reset()
reset_button.on_clicked(reset_button_on_clicked)

#target display
target_ax = fig.add_axes([0.1, 0.4, 0.15, 0.15])
target_text_box = TextBox(target_ax, 'Target', initial = "6.28")
opt_ax = fig.add_axes([0.1, 0.3, 0.15, 0.15])
opt_button = Button(opt_ax, 'find Radius', color=axis_color)
def opt_button_on_clicked(mouse_event):
    radius_slider.set_val(find_best_radius(radius_slider.val, amp_slider.val, freq_slider.val, float(target_text_box.text)))
opt_button.on_clicked(opt_button_on_clicked)

plt.show()
