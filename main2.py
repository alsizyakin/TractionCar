import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox, RadioButtons
matplotlib.use('TkAgg')


def moving_average(a, nl=3):
    ret = np.cumsum(a, dtype=float)
    ret[nl:] = ret[nl:] - ret[:-nl]
    return ret[nl - 1:] / nl

class Motor:
    n_max = 10000
    torque = 350
    power = 80000

    def set_n_max(self,nm):
        self.n_max = nm

    def set_torque(self, t):
        self.torque = t

    def set_power(self, p):
        p = float(p)
        self.power = p

    def calc(self):
        self.n = np.arange(1, self.n_max, 1)
        self.n = np.append(self.n, self.n_max)
        self.trq = self.power / self.n * 60 / (2 * np.pi)
        self.trq[-1] = 0
        self.trq[np.where(self.trq > self.torque)] = self.torque
        return [self.trq, self.n]
    
    
class Vehicle:
    __i = 23
    __r_kol = 0.3
    __mass = 1500
    __cx = 0.3
    __area = 5.5
    __bat_capacity = 60

    def set_i(self, i):
        self.__i = i

    def set_r_kol(self, r):
        self.__r_kol = r

    def set_mass(self, m):
        self.__mass = m

    def set_cx(self, c):
        self.__cx = c

    def set_area(self, a):
        self.__area = a

    def set_bat_capacity(self, cap):
        self.__bat_capacity = cap

    def calc(self, motor_data:list):
        f = motor_data[0] * self.__i / self.__r_kol
        v = motor_data[1] / self.__i / 60 * 2 * np.pi * self.__r_kol * 3.6
        return [f, v]
        



M = Motor()
V = Vehicle()

acc_time100 = 0
n_max = 10000
mass = 1500
face_area = 5.5
load = 0
ramp = 0
r_kol = 0.3
amp = 0
power = 80000
#i = 3
n = 0
v = n

ind_resist = 0
ind_adhesion = 0
trans_k_adhesion = 1
cx = 0.55
F_resist = n
v_max = 0
box_x_pos = [0.25, 0.425, 0.6, 0.775]
box_y_pos = [0.2, 0.25, 0.3, 0.35, 0.4]

v_set = 150
wh_set = 60
road_tire_type = [[0.01, [0.6, 0.75, 0.8]],
                  [0.1, [0.3, 0.37, 0.42]],
                  [0.35, [0.275, 0.35, 0.4]]]

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.55, top=0.95, left=0.3)
fig.set_figheight(10)
fig.set_figwidth(10)
maximum_force, = ax.plot(0, 0, lw=2)
maximum_adhesion_force, = ax.plot(0, 0 * n, lw=2)
traction_resistance, = ax.plot(0, 0 * n, lw=2)
ax.grid()


def update():

    #motor_data = M.calc()
    veh_data = V.calc(M.calc())
    f = veh_data[0]
    v = veh_data[1]
    maximum_force.set_ydata(f)
    maximum_force.set_xdata(v)
    ax.set_ylim(0, max(f) * 1.1)
    ax.set_xlim(0, max(v) * 1.1)
    mass_sum = mass + load
    k_sop = road_tire_type[ind_resist][0]
    k_adhesion_rez = road_tire_type[ind_resist][1][ind_adhesion] * trans_k_adhesion
    f_adhesion = mass_sum * 9.81 * k_adhesion_rez
    f_sop = mass_sum * 9.81 * (k_sop * np.cos(ramp / 180 * np.pi) + np.sin(ramp / 180 * np.pi)) + \
        cx * face_area * (v / 3.6) ** 2
    f_sop /= 0.95  # kpd  of transmission
    maximum_adhesion_force.set_ydata(f_sop)
    maximum_adhesion_force.set_xdata(v)
    f_rez = f
    f_rez[np.where(f_rez > f_adhesion)] = f_adhesion
    traction_resistance.set_ydata(f_rez)
    traction_resistance.set_xdata(v)

    v_max_ind = (np.where(f_rez < f_sop))[0][0]
    v_max = v[v_max_ind]
    text_boxVout.set_val(str(round(v_max * 100) / 100))

    acc_m = (f_rez - f_sop) / mass_sum * 3.6
    acc_r = moving_average(acc_m, 2)
    dt2 = np.diff(v)
    dt2 = dt2 / acc_r
    v100_ind = (np.where(v < 100)[0])
    v_set_ind = (np.where(v < v_set)[0])
    if v_max >= 100:
        v100_ind = v100_ind[-1]
        acc_v_100 = np.cumsum(dt2)

        text_boxTout.set_val(str(round(acc_v_100[v100_ind] * 100) / 100))
    else:
        text_boxTout.set_val('--')
    if v_max >= v_set:
        v_set_ind = v_set_ind[-1]
        acc1502 = np.cumsum(dt2)

        text_boxTmout.set_val(str(round(acc1502[v_set_ind] * 100) / 100))
    else:
        text_boxTmout.set_val('--')

    if v_set > v_max:
        text_boxBatlifeout.set_val('--')
    else:
        lifetime = wh_set*1000/(f_sop[np.where(v >= v_set)][0] * v_set/3.6)
        liferange = lifetime * v_set
        text_boxBatlifeout.set_val(str(round(lifetime*100)/100))
        text_boxBatrangeout.set_val(str(round(liferange)))

    fig.canvas.draw_idle()

#Slider section############################################################################################
ax_ramp = fig.add_axes([0.25, 0.1, 0.65, 0.03])
ax_load = fig.add_axes([0.25, 0.15, 0.65, 0.03])

slider_ramp = Slider(
    ax_ramp, "Ramp, deg", 0, 45,
    valinit=0, valstep=0.1,
    initcolor='none'  # Remove the line marking the valinit position.
)

slider_load = Slider(
    ax_load, "Load, kg", 0, 5000,
    valinit=0, valstep=10,
    color="cyan"
)

def update_ramp(val):
    global ramp
    ramp = val

    text_boxRampProm.set_val(round(np.tan(ramp / 180 * np.pi) * 100000) / 100)
    update()

slider_ramp.on_changed(update_ramp)

def update_load(val):
    global load
    load = val
    update()

slider_load.on_changed(update_load)

#Result section########################################################################################################
axbox = fig.add_axes([box_x_pos[0], box_y_pos[0], 0.1, 0.035])
text_boxVout = TextBox(axbox, "Vmax", textalignment="center", color='lightgoldenrodyellow')

axbox = fig.add_axes([box_x_pos[1], box_y_pos[0], 0.1, 0.035])
text_boxTout = TextBox(axbox, "Acc100", textalignment="center", color='lightgoldenrodyellow')

axbox = fig.add_axes([box_x_pos[2], box_y_pos[0], 0.1, 0.035])
text_boxTmout = TextBox(axbox, "Acc_v_set", textalignment="center", color='lightgoldenrodyellow')

axbox = fig.add_axes([box_x_pos[3], box_y_pos[0], 0.1, 0.035])
text_boxRampProm = TextBox(axbox, "Ramp,%" + chr(1995), textalignment="center", color='lightgoldenrodyellow')
text_boxRampProm.set_val(0)

axbox = fig.add_axes([box_x_pos[0], box_y_pos[1], 0.1, 0.035])
text_boxBatlifeout = TextBox(axbox, "Bat_life", textalignment="center", color='lightgoldenrodyellow')
text_boxBatlifeout.set_val(0)

axbox = fig.add_axes([box_x_pos[1], box_y_pos[1], 0.1, 0.035])
text_boxBatrangeout = TextBox(axbox, "Bat_range", textalignment="center", color='lightgoldenrodyellow')
text_boxBatrangeout.set_val(0)

#Input section#######################################################################################################
def amplsub(expression):
    M.set_torque(float(expression))
    update()

axbox = fig.add_axes([box_x_pos[0], box_y_pos[2], 0.1, 0.035])
text_boxTrq = TextBox(axbox, "Torque", textalignment="center")
text_boxTrq.on_submit(amplsub)
text_boxTrq.set_val(350)

def powersub(expression):
    M.set_power(float(expression))
    update()

axbox = fig.add_axes([box_x_pos[1], box_y_pos[2], 0.1, 0.035])
text_boxPwr = TextBox(axbox, "Power", textalignment="center")
text_boxPwr.on_submit(M.set_power)
#text_boxPwr.on_submit(powersub)
text_boxPwr.set_val(80000)

def nmaxsub(expression):
    M.set_n_max(float(expression))
    update()

axbox = fig.add_axes([box_x_pos[2], box_y_pos[2], 0.1, 0.035])
text_boxNmax = TextBox(axbox, "n_max", textalignment="center")
text_boxNmax.on_submit(nmaxsub)
text_boxNmax.set_val(10000)

def isub(expression):
    V.set_i(float(expression))
    update()

axbox = fig.add_axes([box_x_pos[3], box_y_pos[4], 0.1, 0.035])
text_boxIred = TextBox(axbox, "i", textalignment="center")
text_boxIred.on_submit(isub)
text_boxIred.set_val(23)

def rkolsub(expression):
    V.set_r_kol(float(expression))
    update()

axbox = fig.add_axes([box_x_pos[2], box_y_pos[4], 0.1, 0.035])
text_boxRkol = TextBox(axbox, 'R_kol', textalignment="center")
text_boxRkol.on_submit(rkolsub)
text_boxRkol.set_val(0.3)

def masssub(expression):
    V.set_mass(float(expression))
    update()

axbox = fig.add_axes([box_x_pos[0], box_y_pos[3], 0.1, 0.035])
text_boxMass = TextBox(axbox, "Mass", textalignment="center")
text_boxMass.on_submit(masssub)
text_boxMass.set_val(1500)

def face_ar_sub(expression):
    V.set_area(float(expression))
    update()

axbox = fig.add_axes([box_x_pos[1], box_y_pos[4], 0.1, 0.035])
text_boxSquare = TextBox(axbox, "Face area", textalignment="center")
text_boxSquare.on_submit(face_ar_sub)
text_boxSquare.set_val(face_area)

def cxsub(expression):
    V.set_cx(float(expression))
    update()

axbox = fig.add_axes([box_x_pos[0], box_y_pos[4], 0.1, 0.035])
text_boxCx = TextBox(axbox, "Cx", textalignment="center")
text_boxCx.on_submit(cxsub)
text_boxCx.set_val(cx)

def Whsetsub(expression):
    V.set_bat_capacity(float(expression))
    update()

axbox = fig.add_axes([box_x_pos[2], box_y_pos[3], 0.1, 0.035])
text_boxWhset = TextBox(axbox, "кWh_set", textalignment="center")
text_boxWhset.on_submit(Whsetsub)
text_boxWhset.set_val(wh_set)

def Vsetsub(expression):
    global v_set
    v_set = float(expression)
    update()

axbox = fig.add_axes([box_x_pos[1], box_y_pos[3], 0.1, 0.035])
text_boxVset = TextBox(axbox, "v_set", textalignment="center")
text_boxVset.on_submit(Vsetsub)
text_boxVset.set_val(v_set)

#RadioButtons section#################################################################################
radioButtonColor = 'lightgoldenrodyellow'
rax = fig.add_axes([0.05, 0.8, 0.15, 0.15], facecolor=radioButtonColor)
radio = RadioButtons(rax, ('Asphault', 'Ground', 'Plow'))

def roadtype(label):
    roaddict = {'Asphault': 0, 'Ground': 1, 'Plow': 2}
    global ind_resist
    ind_resist = roaddict[label]
    update()

radio.on_clicked(roadtype)

rax = fig.add_axes([0.05, 0.6, 0.15, 0.15], facecolor=radioButtonColor)
radio2 = RadioButtons(rax, ('HighPres', 'LowPres', 'MudTire'))

def tiretype(label):
    tiredict = {'HighPres': 0, 'LowPres': 1, 'MudTire': 2}
    global ind_adhesion
    ind_adhesion = tiredict[label]
    update()

radio2.on_clicked(tiretype)

rax = fig.add_axes([0.05, 0.4, 0.15, 0.15], facecolor=radioButtonColor)
radio3 = RadioButtons(rax, ('AWD', '4x2', '6x4'))

def transtype(label):
    transdict = {'AWD': 1, '4x2': 0.6, '6x4': 0.75}
    global trans_k_adhesion
    trans_k_adhesion = transdict[label]
    update()

radio3.on_clicked(transtype)

#Buttons section############################################################################################
#ax_reset = fig.add_axes([0.05, box_y_pos[3], 0.1, 0.04])
#button0 = Button(ax_reset, 'Reset', hovercolor='0.975')

#def reset(event):
#    slider_ramp.reset()
#    slider_load.reset()
#    text_boxTrq.set_val(350)
#
#button0.on_clicked(reset)
#
#ax_save = fig.add_axes([0.05, box_y_pos[2], 0.1, 0.04])
#button1 = Button(ax_save, 'save', hovercolor='0.975')

# def save_data(event):
#     s = 1

# button1.on_clicked(save_data)

# ax_load = fig.add_axes([0.05, box_y_pos[1], 0.1, 0.04])
# button2 = Button(ax_load, 'load', hovercolor='0.975')

# def load_data(event):
#     s = 1

# button2.on_clicked(load_data)

plt.show()
