from numpy import pi, sin, cos
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox
import math
import linetostl
from functools import partial
from scipy.optimize import minimize
import integral
import curves_overlap

c1_0 = 0
c2_0 = 0
rad_0 = 1

def circle_x_sine(radius = 1, c1 = 0, c2 = 0):
    return radius * (1 + c1 * cos(4 * theta) + c2 * cos(8 * theta))

theta = np.arange(0, (2 * pi), 0.01)
radii = circle_x_sine(rad_0, c1_0, c2_0)

def compute_and_format_length(radius, c1, c2):
    return "{:.2f}".format(integral.approximate_integral_length(radius, c1, c2))

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
    return abs(fixed['target_radius'] - integral.approximate_integral_length(radius, c1, c2))

axis_color = 'lightgoldenrodyellow'

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='polar')

# Adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.25, bottom=0.25)

# Draw the initial plot
# The 'line' variable is used for modifying the line later
[line] = ax.plot(theta, radii, linewidth=2, color='red')
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
radius_slider = Slider(radius_slider_ax, 'Radius', 0.1, 10, valinit=rad_0)

#length display
length_ax = fig.add_axes([0.1, 0.5, 0.15, 0.05])
length_text_box = TextBox(length_ax, 'Length', initial = compute_and_format_length(rad_0, c1_0, c2_0))

#height display
height_ax = fig.add_axes([0.1, 0.8, 0.15, 0.05])
height_text_box = TextBox(height_ax, 'Height', initial = '10')

#bounding box display
bb_ax = fig.add_axes([0.1, 0.6, 0.15, 0.05])
bb_text_box = TextBox(bb_ax, 'BB', initial = '5')

#width display
width_ax = fig.add_axes([0.1, 0.7, 0.15, 0.05])
width_text_box = TextBox(width_ax, 'Width', initial = '.5')


def check_for_neg_rad():
    if np.amin(radii) < 0.01:
        return True
    return False

def checkBB():
    bb = float(bb_text_box.text)
    pts = linetostl.polarToCart(theta, radii)
    if np.amax(pts) > bb or np.amin(pts) < -bb:
        return True
    return False

def check_overlap():
    pts = np.array(linetostl.polarToCart(theta, radii))
    width = float(width_text_box.text)
    return curves_overlap.check_for_intersection(pts,width)

# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
    global radii
    radii = circle_x_sine(radius_slider.val, c1_slider.val, c2_slider.val)
    line.set_ydata(radii)
    length_text_box.set_val(compute_and_format_length(radius_slider.val, c1_slider.val, c2_slider.val))
    fig.canvas.draw_idle()
c1_slider.on_changed(sliders_on_changed)
c2_slider.on_changed(sliders_on_changed)
radius_slider.on_changed(sliders_on_changed)

def getStlName():
    return 'r_' + "{:.2f}".format(radius_slider.val) + '_c1_' + "{:.2f}".format(c1_slider.val) + '_c2_' + "{:.2f}".format(c2_slider.val) + '.stl'

# Add a button for generating stl
stl_button_ax = fig.add_axes([0.8, 0., 0.1, 0.04])
stl_button = Button(stl_button_ax, 'STL', color=axis_color, hovercolor='0.975')
def stl_button_on_clicked(mouse_event):
    linetostl.lineToSTL(curves_overlap.offset_curve(np.array(linetostl.polarToCart(theta, radii)), -float(width_text_box.text)/2), getStlName(), float(height_text_box.text))
    #linetostl.lineToSTL(linetostl.polarToCart(theta, radii), 'original_' + getStlName(), float(height_text_box.text))
stl_button.on_clicked(stl_button_on_clicked)

#fixing arguments
checkbuttons_ax = fig.add_axes([0.1, 0.2, 0.15, 0.1])
checkbuttons_fix = CheckButtons(checkbuttons_ax, ['c1', 'c2', 'R'])

#opt display
target_ax = fig.add_axes([0.1, 0.4, 0.15, 0.05])
target_text_box = TextBox(target_ax, 'Target', initial = "6.28")
opt_ax = fig.add_axes([0.1, 0.3, 0.15, 0.05])
opt_button = Button(opt_ax, 'Optimize', color=axis_color)
def opt_button_on_clicked(mouse_event):
    x0 = []
    partial_params = {'target_radius':float(target_text_box.text)}
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
    res = minimize(partial(optimizable_function, partial_params), np.array(x0), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    result_index = 0
    if not fixed_params[2]:
        radius_slider.set_val(res.x[result_index])
        result_index = result_index + 1
    if not fixed_params[0]:
        c1_slider.set_val(res.x[result_index])
        result_index = result_index + 1
    if not fixed_params[1]:
        c2_slider.set_val(res.x[result_index])
opt_button.on_clicked(opt_button_on_clicked)


# Add a button for checking validity
check_button_ax = fig.add_axes([0.4, 0., 0.1, 0.04])
check_button = Button(check_button_ax, 'Check', color=axis_color, hovercolor='0.975')
def check_button_on_clicked(mouse_event):
    problem = False
    if checkBB():
        print("Curve exceeds bounding box!")
        problem = True
    if check_for_neg_rad():
        print("Curve has negative radius!")
        problem = True
    if check_overlap():
        print("Curve self-overlaps!")
        problem = True
    if not problem:
        print("The curve is valid.")
check_button.on_clicked(check_button_on_clicked)

plt.show()
