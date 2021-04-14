from numpy import pi, sin, cos
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox
import math
from functools import partial
from scipy.optimize import minimize
import utils

# Globals

c1_0 = 0
c2_0 = 0
rad_0 = 1

theta = np.arange(0, (2 * pi), 0.01)
radii = utils.radius_SC(theta, rad_0, c1_0, c2_0)

# Utility functions

# computes and truncates the length of the curve to 2 decimal places.
def compute_and_format_length(radius, c1, c2):
    return "{:.2f}".format(utils.approximate_integral_SC_length(radius, c1, c2))

# function used to in the optimization, necessary for fixing paramters.
# During the optimization setup, we will create a partial function with fixed containing the fixed parameters.
def optimizable_function(fixed, params):
    param_index = 0
    radius, c1, c2 = (0,0,0)
    if 'radius' in fixed:
        radius = fixed['radius']
    else:
        radius = params[param_index]
        param_index = param_index + 1
    if 'c1' in fixed:
        c1 = fixed['c1']
    else:
        c1 = params[param_index]
        param_index = param_index + 1
    if 'c2' in fixed:
        c2 = fixed['c2']
    else:
        c2 = params[param_index]
        param_index = param_index + 1
    return abs(fixed['target_radius'] - utils.approximate_integral_SC_length(radius, c1, c2))

# Display setup

axis_color = 'lightgoldenrodyellow'

# The window and plot
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='polar')

# Adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.25, bottom=0.25)

# Draw the initial plot
# The 'line' variable is used for modifying the line later
[line] = ax.plot(theta, radii, linewidth=2, color='red')
ax.set_ylim([0, 5])

# Add three sliders for tweaking the parameters
c1_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axis_color)
c1_slider = Slider(c1_slider_ax, 'c1', -1, 1.0, valinit=c1_0)

c2_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor=axis_color)
c2_slider = Slider(c2_slider_ax, 'c2', -1, 1, valinit=c2_0)

radius_slider_ax = fig.add_axes([0.25, 0.05, 0.65, 0.03], facecolor=axis_color)
radius_slider = Slider(radius_slider_ax, 'Radius', 0.1, 10, valinit=rad_0)

# length display
length_ax = fig.add_axes([0.1, 0.5, 0.15, 0.05])
length_text_box = TextBox(length_ax, 'Length', initial = compute_and_format_length(rad_0, c1_0, c2_0))

# Height display
height_ax = fig.add_axes([0.1, 0.8, 0.15, 0.05])
height_text_box = TextBox(height_ax, 'Height', initial = '10')

# Bounding box display
bb_ax = fig.add_axes([0.1, 0.6, 0.15, 0.05])
bb_text_box = TextBox(bb_ax, 'BB', initial = '5')

# Width display
width_ax = fig.add_axes([0.1, 0.7, 0.15, 0.05])
width_text_box = TextBox(width_ax, 'Width', initial = '.5')

# Target length display
target_ax = fig.add_axes([0.1, 0.4, 0.15, 0.05])
target_text_box = TextBox(target_ax, 'Target', initial = "6.28")

# Optimization button
opt_ax = fig.add_axes([0.1, 0.3, 0.15, 0.05])
opt_button = Button(opt_ax, 'Optimize', color=axis_color)

# Fixed parameters checkboxes
checkbuttons_ax = fig.add_axes([0.1, 0.2, 0.15, 0.1])
checkbuttons_fix = CheckButtons(checkbuttons_ax, ['c1', 'c2', 'R'])

# Sanity check button
check_button_ax = fig.add_axes([0.4, 0., 0.1, 0.04])
check_button = Button(check_button_ax, 'Check', color=axis_color, hovercolor='0.975')

# Button for generating stl
stl_button_ax = fig.add_axes([0.8, 0., 0.1, 0.04])
stl_button = Button(stl_button_ax, 'STL', color=axis_color, hovercolor='0.975')

# Callbacks
def sliders_on_changed(val):
    global radii
    radii = utils.radius_SC(theta, radius_slider.val, c1_slider.val, c2_slider.val)
    line.set_ydata(radii)
    length_text_box.set_val(compute_and_format_length(radius_slider.val, c1_slider.val, c2_slider.val))
    fig.canvas.draw_idle()
c1_slider.on_changed(sliders_on_changed)
c2_slider.on_changed(sliders_on_changed)
radius_slider.on_changed(sliders_on_changed)

def get_stl_name():
    return 'r_' + "{:.2f}".format(radius_slider.val) + '_c1_' + "{:.2f}".format(c1_slider.val) + '_c2_' + "{:.2f}".format(c2_slider.val) + '.stl'

def stl_button_on_clicked(mouse_event):
    # In order: we get the carthesian coordiantes of the curve, offset it outward then make it into an STL file
    utils.lineToSTL(utils.offset_curve(np.array(utils.polarToCart(theta, radii)), -float(width_text_box.text)/2), get_stl_name(), float(height_text_box.text))
stl_button.on_clicked(stl_button_on_clicked)


def opt_button_on_clicked(mouse_event):
    # x0 contains the parameters we want to optimize, in the order: R, c1, c2.
    # partial_params contains the target length and the fixed parameters.
    x0 = []
    partial_params = {'target_radius':float(target_text_box.text)}
    # We check the values of the checkboxes and add the corresponding parameters to x0 (resp. partial_params) if unchecked, aka we want to optimize them (resp. if we want to keep them as is)
    fixed_params = checkbuttons_fix.get_status()
    if fixed_params[2]:
        partial_params['radius'] = radius_slider.val
    else:
        x0.append(radius_slider.val)
    if fixed_params[0]:
        partial_params['c1'] = c1_slider.val
    else:
        x0.append(c1_slider.val)
    if fixed_params[1]:
        partial_params['c2'] = c2_slider.val
    else:
        x0.append(c2_slider.val)
    # We create a partial function with the fixed parameters and the length.
    # Then we try to minimize it.
    res = minimize(partial(optimizable_function, partial_params), np.array(x0), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    result_index = 0
    # We assign each value inside the result to its slider.
    if not fixed_params[2]:
        radius_slider.set_val(res.x[result_index])
        result_index = result_index + 1
    if not fixed_params[0]:
        c1_slider.set_val(res.x[result_index])
        result_index = result_index + 1
    if not fixed_params[1]:
        c2_slider.set_val(res.x[result_index])
opt_button.on_clicked(opt_button_on_clicked)


def check_button_on_clicked(mouse_event):
    utils.sanity_check_verbose(theta, radii, float(width_text_box.text), float(bb_text_box.text))
check_button.on_clicked(check_button_on_clicked)

plt.show()
