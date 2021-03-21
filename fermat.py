from numpy import pi, sin, cos
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox
from math import sqrt
import linetostl
from functools import partial
from scipy.optimize import minimize
import integral
import curves_overlap

class data_linewidth_plot():
    def __init__(self, x, y, **kwargs):
        self.ax = kwargs.pop("ax", plt.gca())
        self.fig = self.ax.get_figure()
        self.lw_data = kwargs.pop("linewidth", 1)
        self.lw = 1
        self.fig.canvas.draw()

        self.ppd = 72./self.fig.dpi
        self.trans = self.ax.transData.transform
        self.linehandle, = self.ax.plot([],[],**kwargs)
        if "label" in kwargs: kwargs.pop("label")
        self.line, = self.ax.plot(x, y, **kwargs)
        self.line.set_color(self.linehandle.get_color())
        self._resize()
        self.cid = self.fig.canvas.mpl_connect('draw_event', self._resize)

    def _resize(self, event=None):
        lw =  ((self.trans((1, self.lw_data))-self.trans((0, 0)))*self.ppd)[1]
        if lw != self.lw:
            self.line.set_linewidth(lw)
            self.lw = lw
            self._redraw_later()

    def _redraw_later(self):
        self.timer = self.fig.canvas.new_timer(interval=10)
        self.timer.single_shot = True
        self.timer.add_callback(lambda : self.fig.canvas.draw_idle())
        self.timer.start()

type = 'fermat'

a = 5
max_theta = 4 * pi

def make_spiral(a, max_theta, type):
    theta = np.arange(0, max_theta, 0.01)
    theta_trunc = np.flip(np.arange(0.01, max_theta + pi, 0.01))
    double_theta = np.append(theta_trunc, theta)
    if type == 'fermat':
        double_radii = np.append(-a * np.sqrt(theta_trunc), a * np.sqrt(theta))
        return double_theta, double_radii
    if type == 'arch':
        double_radii = np.append(-a * theta_trunc, a * theta)
        return double_theta, double_radii
    if type == 'mixed':
        pi_squared = pi*pi
        radii = [(max(pi_squared-theta, 0)*pi*sqrt(theta) + min(theta, pi_squared)*theta)/pi_squared for theta in theta]
        radii_trunc = [(max(pi_squared-theta, 0)*pi*sqrt(theta) + min(theta, pi_squared)*theta)/pi_squared for theta in theta_trunc]
        double_radii = np.append(-a * np.array(radii_trunc), a * np.array(radii))
        return double_theta, double_radii


theta, radii = make_spiral(a, max_theta, type)

axis_color = 'lightgoldenrodyellow'

fig = plt.figure(1)
ax = fig.add_subplot(111)

# Adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.25, bottom=0.25)

# Draw the initial plot
# The 'line' variable is used for modifying the line later
pts = np.array(linetostl.polarToCart(theta, radii))

line =  data_linewidth_plot(pts[:,0], pts[:,1], ax=ax, linewidth=.6, color='red')
ax.set_xlim([-19, 19])
ax.set_ylim([-19, 19])
ax.set_aspect('equal')

# Add two sliders for tweaking the parameters

# theta
max_theta_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axis_color)
max_theta_slider = Slider(max_theta_slider_ax, 'max_theta', pi, 8*pi, valinit=max_theta)

# a
a_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor=axis_color)
a_slider = Slider(a_slider_ax, 'a', 0, 10, valinit=a)

def compute_fermat_length_from_scratch(a, max_theta):
    length1 = integral.fermat_length(a, max_theta)
    length2 = integral.fermat_length(a, max_theta + pi)
    length_between_ends = a * (sqrt(max_theta + pi) - sqrt(max_theta))
    return float(length1 + length2) + length_between_ends

def is_inside_bb_fermat(a, max_theta, BB):
    return a * (sqrt(max_theta + pi) + sqrt(max_theta)) - BB

def compute_arch_length_from_scratch(a, max_theta):
    length1 = integral.arch_length(a, max_theta)
    length2 = integral.arch_length(a, max_theta + pi)
    length_between_ends = a * pi
    return float(length1 + length2) + length_between_ends

def is_inside_bb_arch(a, max_theta, BB):
    return a * (2 * max_theta + pi) - BB

def compute_mixed_length_from_scratch(a, max_theta):
    theta, radii = make_spiral(a, max_theta, 'mixed')
    pts = np.array(linetostl.polarToCart(theta, radii))
    length = integral.length_from_points(pts)
    return length

def is_inside_bb_mixed(a, max_theta, BB):
    pi_squared = pi*pi
    r1 = (max(pi_squared-max_theta, 0)*pi*sqrt(max_theta) + min(max_theta, pi_squared)*max_theta)/pi_squared
    r2 = (max(pi_squared-max_theta+pi, 0)*pi*sqrt(max_theta+pi) + min(max_theta+pi, pi_squared)*(max_theta+pi))/pi_squared
    return a * (r1 + r2) - BB

def compute_and_format_length(a, max_theta):
    if type == "fermat":
        return "{:.2f}".format(compute_fermat_length_from_scratch(a, max_theta))
    elif type == "arch":
        return "{:.2f}".format(compute_arch_length_from_scratch(a, max_theta))
    elif type == "mixed":
        length = integral.length_from_points(pts)
        return "{:.2f}".format(length)

#length display
length_ax = fig.add_axes([0.1, 0.5, 0.15, 0.05])
length_text_box = TextBox(length_ax, 'Length', initial = compute_and_format_length(a, max_theta))

#height display
height_ax = fig.add_axes([0.1, 0.8, 0.15, 0.05])
height_text_box = TextBox(height_ax, 'Height', initial = '19')

#bounding box display
bb_ax = fig.add_axes([0.1, 0.6, 0.15, 0.05])
bb_text_box = TextBox(bb_ax, 'BB', initial = '38')

#width display
width_ax = fig.add_axes([0.1, 0.7, 0.15, 0.05])
width_text_box = TextBox(width_ax, 'Width', initial = '.6')


def check_for_neg_rad():
    if np.amin(radii) < 0.01:
        return True
    return False

def checkBB():
    bb = float(bb_text_box.text)
    if np.amax(pts) > bb or np.amin(pts) < -bb:
        return True
    return False

def check_overlap():
    width = float(width_text_box.text)
    return curves_overlap.check_for_intersection(pts,width)

# Define an action for modifying the line when any slider's value changes
def max_theta_slider_on_changed(val):
    global max_theta
    max_theta = val
    update_graph()


# Define an action for modifying the line when any slider's value changes
def a_slider_on_changed(val):
    global a
    a = val
    update_graph()

max_theta_slider.on_changed(max_theta_slider_on_changed)
a_slider.on_changed(a_slider_on_changed)

def update_graph():
    global pts, a, max_theta, type
    theta, radii = make_spiral(a, max_theta, type)
    pts = np.array(linetostl.polarToCart(theta, radii))
    line.line.set_xdata(pts[:,0])
    line.line.set_ydata(pts[:,1])
    length_text_box.set_val(compute_and_format_length(a_slider.val, max_theta_slider.val))
    fig.canvas.draw_idle()

fermat_button_ax = fig.add_axes([0.2, 0., 0.1, 0.04])
fermat_button = Button(fermat_button_ax, 'Fermat', color=axis_color, hovercolor='0.975')
def fermat_button_on_clicked(mouse_event):
    global type
    type = 'fermat'
    update_graph()
fermat_button.on_clicked(fermat_button_on_clicked)

arch_button_ax = fig.add_axes([0.4, 0., 0.1, 0.04])
arch_button = Button(arch_button_ax, 'Arch.', color=axis_color, hovercolor='0.975')
def arch_button_on_clicked(mouse_event):
    global type
    type = 'arch'
    update_graph()
arch_button.on_clicked(arch_button_on_clicked)

mixed_button_ax = fig.add_axes([0.6, 0., 0.1, 0.04])
mixed_button = Button(mixed_button_ax, 'Mixed', color=axis_color, hovercolor='0.975')
def mixed_button_on_clicked(mouse_event):
    global type
    type = 'mixed'
    update_graph()
mixed_button.on_clicked(mixed_button_on_clicked)


#optimization things

#fixing arguments
checkbuttons_ax = fig.add_axes([0.1, 0.2, 0.15, 0.1])
checkbuttons_fix = CheckButtons(checkbuttons_ax, ['a', 'max_theta'])

def optimizable_function(fixed, params):
    param_index = 0
    a, max_theta = (0,0)
    if 'a' in fixed:
        a = fixed['a']
    else:
        a = params[param_index]
        param_index = param_index + 1
    if 'max_theta' in fixed:
        max_theta = fixed['max_theta']
    else:
        max_theta = params[param_index]
        param_index = param_index + 1
    length = 0
    penalty = 0
    if 'fermat' in fixed:
        length = compute_fermat_length_from_scratch(a, max_theta)
        bb_diff = is_inside_bb_fermat(a, max_theta, fixed['BB'])
        if bb_diff > 0:
            penalty = 10*bb_diff
    if 'arch' in fixed:
        length = compute_arch_length_from_scratch(a, max_theta)
        bb_diff = is_inside_bb_arch(a, max_theta, fixed['BB'])
        if bb_diff > 0:
            penalty = 10*bb_diff
    if 'mixed' in fixed:
        length = compute_mixed_length_from_scratch(a, max_theta)
        bb_diff = is_inside_bb_mixed(a, max_theta, fixed['BB'])
        if bb_diff > 0:
            penalty = 10*bb_diff
    return abs(fixed['target_length'] - length) + penalty


target_ax = fig.add_axes([0.1, 0.4, 0.15, 0.05])
target_text_box = TextBox(target_ax, 'Target', initial = "240.667")
opt_ax = fig.add_axes([0.1, 0.3, 0.15, 0.05])
opt_button = Button(opt_ax, 'Optimize', color=axis_color)
def opt_button_on_clicked(mouse_event):
    x0 = []
    partial_params = {'target_length':float(target_text_box.text), type:True, "BB":float(bb_text_box.text)}
    fixed_params = checkbuttons_fix.get_status()
    if fixed_params[0]:
        partial_params['a'] = a_slider.val
    else:
        x0.append(a_slider.val)
    if fixed_params[1]:
        partial_params['max_theta'] = max_theta_slider.val
    else:
        x0.append(max_theta_slider.val)
    res = minimize(partial(optimizable_function, partial_params), np.array(x0), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    print(res.x)
    result_index = 0
    if not fixed_params[0]:
        a_slider.set_val(res.x[result_index])
        result_index = result_index + 1
    if not fixed_params[1]:
        max_theta_slider.set_val(res.x[result_index])
opt_button.on_clicked(opt_button_on_clicked)

def getStlName():
    return 'spiral_' + type + "_a_" + "{:.2f}".format(a_slider.val) + '_max_theta_' + "{:.2f}".format(max_theta_slider.val) + '.stl'

# Add a button for generating stl
stl_button_ax = fig.add_axes([0.8, 0., 0.1, 0.04])
stl_button = Button(stl_button_ax, 'STL', color=axis_color, hovercolor='0.975')
def stl_button_on_clicked(mouse_event):
    linetostl.lineToSTL(pts, getStlName(), float(height_text_box.text))
stl_button.on_clicked(stl_button_on_clicked)

plt.show()
