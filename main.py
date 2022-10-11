import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox, RadioButtons

matplotlib.use('TkAgg')

acc_time100 = 0
n_max = 10000
mass = 1500
face_area = 5.5
load = 0
ramp = 0
r_kol = 0.3
amp = 0
power = 80000
i = 3
n = 0
v = n
a0 = 10
f0 = 3
ind_resist = 0
ind_adhesion = 0
trans_k_adhesion = 1
cx = 0.55
F_resist = n
v_max = 0
box_x_pos = [0.25, 0.425, 0.6, 0.775]
box_y_pos = [0.2, 0.25, 0.3, 0.35, 0.4]

v_set = 10
wh_set = 60
road_tire_type = [[0.01, [0.6, 0.75, 0.75]],
                  [0.1, [0.3, 0.37, 0.42]],
                  [0.35, [0.275, 0.35, 0.4]]]

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.55, top=0.95, left=0.3)
fig.set_figheight(10)
fig.set_figwidth(10)
l1, = ax.plot(0, 0, lw=2)
l2, = ax.plot(0, 0 * n, lw=2)
l3, = ax.plot(0, 0 * n, lw=2)
ax.grid()


def update():
    global ramp
    global load
    global i
    global ax
    global v
    global trans_k_adhesion
    global v_max
    global n
    global l1
    global acc_time100
    n = np.arange(1, n_max, 1)
    n = np.append(n, n_max)
    trq = power / n * 60 / (2 * np.pi)
    trq[-1] = 0
    trq[np.where(trq > amp)] = amp
    trq *= i
    f = trq / r_kol
    v = n / i / 60 * 2 * np.pi * r_kol * 3.6
    l1.set_ydata(f)
    l1.set_xdata(v)
    ax.set_ylim(0, max(f) * 1.1)
    ax.set_xlim(0, max(v) * 1.1)
    mass_sum = mass + load
    k_sop = road_tire_type[ind_resist][0]
    k_adhesion_rez = road_tire_type[ind_resist][1][ind_adhesion] * trans_k_adhesion
    f_adhesion = mass_sum * 9.81 * k_adhesion_rez
    f_sop = mass_sum * 9.81 * (k_sop * np.cos(ramp / 180 * np.pi) + np.sin(ramp / 180 * np.pi)) + \
            cx * face_area * (v / 3.6) ** 2
    f_sop /= 0.95  # kpd  of transmission
    l2.set_ydata(f_sop)
    l2.set_xdata(v)
    f_rez = f
    f_rez[np.where(f_rez > f_adhesion)] = f_adhesion
    l3.set_ydata(f_rez)
    l3.set_xdata(v)
    time150 = 0
    time100 = 0

    v_max = v[-1]
    for ind in range(0, v.size):
        if f_rez[ind] < f_sop[ind]:
            v_max = v[ind]
            break
        else:
            if ind != 0:
                acc_prev = (f_rez[ind - 1] - f_sop[ind - 1]) / mass_sum * 3.6
                acc_act = (f_rez[ind] - f_sop[ind]) / mass_sum * 3.6
                acc = (acc_prev + acc_act) * 0.5
                dt = (v[ind] - v[ind - 1]) / acc
                if v[ind] <= 150:
                    time150 += dt
                if v[ind] <= 100:
                    time100 = time150

    if v_max < 150:
        text_boxTmout.set_val('--')
    else:
        text_boxTmout.set_val(str(round(time150 * 100) / 100))
    if v_max < 100:
        text_boxTout.set_val('--')
    else:
        text_boxTout.set_val(str(round(time100 * 100) / 100))
    text_boxVout.set_val(str(round(v_max)))

    if v_set > v_max:
        text_boxBatlifeout.set_val('--')
    else:
        lifetime = wh_set*1000/(f_sop[np.where(v >= v_set)][0] * v_set/3.6)
        liferange = lifetime * v_set
        #print(v_set, np.where(v >= v_set)[0][0], lifetime)
        #print(lifetime)
        text_boxBatlifeout.set_val(str(round(lifetime*100)/100))
        text_boxBatrangeout.set_val(str(round(liferange)))
    fig.canvas.draw_idle()


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



axbox = fig.add_axes([box_x_pos[0], box_y_pos[0], 0.1, 0.035])
text_boxVout = TextBox(axbox, "Vmax", textalignment="center", color='lightgoldenrodyellow')

axbox = fig.add_axes([box_x_pos[1], box_y_pos[0], 0.1, 0.035])
text_boxTout = TextBox(axbox, "Acc100", textalignment="center", color='lightgoldenrodyellow')

axbox = fig.add_axes([box_x_pos[2], box_y_pos[0], 0.1, 0.035])
text_boxTmout = TextBox(axbox, "Acc150", textalignment="center", color='lightgoldenrodyellow')

axbox = fig.add_axes([box_x_pos[3], box_y_pos[0], 0.1, 0.035])
text_boxRampProm = TextBox(axbox, "Ramp,pr", textalignment="center", color='lightgoldenrodyellow')
text_boxRampProm.set_val(0)

axbox = fig.add_axes([box_x_pos[0], box_y_pos[1], 0.1, 0.035])
text_boxBatlifeout = TextBox(axbox, "Bat_life", textalignment="center", color='lightgoldenrodyellow')
text_boxBatlifeout.set_val(0)

axbox = fig.add_axes([box_x_pos[1], box_y_pos[1], 0.1, 0.035])
text_boxBatrangeout = TextBox(axbox, "Bat_range", textalignment="center", color='lightgoldenrodyellow')
text_boxBatrangeout.set_val(0)


def amplsub(expression):
    global amp
    amp = float(expression)
    update()


axbox = fig.add_axes([box_x_pos[0], box_y_pos[2], 0.1, 0.035])
text_boxTrq = TextBox(axbox, "Torque", textalignment="center")
text_boxTrq.on_submit(amplsub)
text_boxTrq.set_val(350)


def powersub(expression):
    global power
    power = float(expression)
    update()


axbox = fig.add_axes([box_x_pos[1], box_y_pos[2], 0.1, 0.035])
text_boxPwr = TextBox(axbox, "Power", textalignment="center")
text_boxPwr.on_submit(powersub)
text_boxPwr.set_val(80000)


def nmaxsub(expression):
    global n_max
    n_max = float(expression)
    update()


axbox = fig.add_axes([box_x_pos[2], box_y_pos[2], 0.1, 0.035])
text_boxNmax = TextBox(axbox, "n_max", textalignment="center")
text_boxNmax.on_submit(nmaxsub)
text_boxNmax.set_val(10000)


def isub(expression):
    global i
    i = float(expression)
    update()


axbox = fig.add_axes([box_x_pos[3], box_y_pos[4], 0.1, 0.035])
text_boxIred = TextBox(axbox, "i", textalignment="center")
text_boxIred.on_submit(isub)
text_boxIred.set_val(23)


def rkolsub(expression):
    global r_kol
    r_kol = float(expression)
    update()


axbox = fig.add_axes([box_x_pos[2], box_y_pos[4], 0.1, 0.035])
text_boxRkol = TextBox(axbox, 'R_kol', textalignment="center")
text_boxRkol.on_submit(rkolsub)
text_boxRkol.set_val(0.3)


def masssub(expression):
    global mass
    mass = float(expression)
    update()


axbox = fig.add_axes([box_x_pos[0], box_y_pos[3], 0.1, 0.035])
text_boxMass = TextBox(axbox, "Mass", textalignment="center")
text_boxMass.on_submit(masssub)
text_boxMass.set_val(1500)


def face_ar_sub(expression):
    global face_area
    face_area = float(expression)
    update()


axbox = fig.add_axes([box_x_pos[1], box_y_pos[4], 0.1, 0.035])
text_boxSquare = TextBox(axbox, "Face area", textalignment="center")
text_boxSquare.on_submit(face_ar_sub)
text_boxSquare.set_val(face_area)


def cxsub(expression):
    global cx
    cx = float(expression)
    update()


axbox = fig.add_axes([box_x_pos[0], box_y_pos[4], 0.1, 0.035])
text_boxCx = TextBox(axbox, "Cx", textalignment="center")
text_boxCx.on_submit(cxsub)
text_boxCx.set_val(cx)


def Vsetsub(expression):
    global v_set
    v_set = float(expression)
    update()


axbox = fig.add_axes([box_x_pos[1], box_y_pos[3], 0.1, 0.035])
text_boxVset = TextBox(axbox, "v_set", textalignment="center")
text_boxVset.on_submit(Vsetsub)
text_boxVset.set_val(v_set)


def Whsetsub(expression):
    global wh_set
    wh_set = float(expression)
    update()


axbox = fig.add_axes([box_x_pos[2], box_y_pos[3], 0.1, 0.035])
text_boxWhset = TextBox(axbox, "Wh_set", textalignment="center")
text_boxWhset.on_submit(Whsetsub)
text_boxWhset.set_val(wh_set)


ax_reset = fig.add_axes([0.05, box_y_pos[3], 0.1, 0.04])
button0 = Button(ax_reset, 'Reset', hovercolor='0.975')


def reset(event):
    slider_ramp.reset()
    slider_load.reset()
    text_boxTrq.set_val(350)


button0.on_clicked(reset)

ax_save = fig.add_axes([0.05, box_y_pos[2], 0.1, 0.04])
button1 = Button(ax_save, 'save', hovercolor='0.975')


def save_data(event):
    s = 1


button1.on_clicked(save_data)

ax_load = fig.add_axes([0.05, box_y_pos[1], 0.1, 0.04])
button2 = Button(ax_load, 'load', hovercolor='0.975')


def load_data(event):
    s = 1


button2.on_clicked(load_data)

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

plt.show()
