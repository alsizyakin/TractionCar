import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox, RadioButtons
matplotlib.use('TkAgg')

mass = 1500
square = 5.5
load = 0
nakl = 0
rkol = 0.3
amp = 0
power = 80000
i = 3
n = np.arange(1, 10000, 1)
v = n
a0 = 10
f0 = 3
ind_sopr = 0
ind_scep = 0
trans_k_scep = 1
cx = 0.55
Fsop = n
vmin = 0


roadtiretype = [[0.01, [0.6, 0.75, 0.75]],
                [0.1, [0.2, 0.2, 0.25]],
                [0.35, [0.175, 0.2, 0.2]]]


fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.55, top=0.95, left=0.3)
fig.set_figheight(10)
fig.set_figwidth(10)
l, = ax.plot(n, 0 * n, lw=2)
l2, = ax.plot(n, 0 * n, lw=2)
l3, = ax.plot(n, 0 * n, lw=2)

ax_nakl = fig.add_axes([0.25, 0.1, 0.65, 0.03])
ax_load = fig.add_axes([0.25, 0.15, 0.65, 0.03])
ax.grid()

snakl = Slider(
    ax_nakl, "Naklon, deg", 0, 90,
    valinit=0, valstep=1,
    initcolor='none'  # Remove the line marking the valinit position.
)

sload = Slider(
    ax_load, "Load, kg", 0, 5000,
    valinit=0, valstep=10,
    color="cyan"
)

axbox = fig.add_axes([0.07, 0.2, 0.1, 0.035])
text_boxOut = TextBox(axbox, "Vmax", textalignment="center", color='red')


def update(val):
    global nakl
    global load
    global i
    global ax
    global v
    global trans_k_scep
    global vmin
    nakl = snakl.val
    load = sload.val
    trq = power / n * 60 / (2 * np.pi)
    trq[np.where(trq > amp)] = amp
    trq *= i
    f = trq/rkol
    v = n / i / 60 * 2 * np.pi * rkol * 3.6
    l.set_ydata(f)
    l.set_xdata(v)
    ax.set_ylim(0, max(f)*1.1)
    ax.set_xlim(0, max(v)*1.1)
    mass_sum = mass + load
    k_sop = roadtiretype[ind_sopr][0]
    k_scep_rez = roadtiretype[ind_sopr][1][ind_scep] * trans_k_scep
    f_scep = mass_sum * 9.81 * k_scep_rez
    f_sop = mass_sum * 9.81 * (k_sop * np.cos(nakl/180*np.pi) + np.sin(nakl/180*np.pi)) + cx * square * (v / 3.6) ** 2
    l2.set_ydata(f_sop)
    l2.set_xdata(v)
    f_rez = f
    f_rez[np.where(f_rez > f_scep)] = f_scep
    l3.set_ydata(f_rez)
    l3.set_xdata(v)

    for ind in range(0, v.size):
        if f_rez[ind] < f_sop[ind]:
            vmin = v[ind]
            break
        else:
            vmin = v[-1]

    text_boxOut.set_val(str(round(vmin)))

    fig.canvas.draw_idle()


snakl.on_changed(update)
sload.on_changed(update)


def amplsub(expression):
    global amp
    amp = float(expression)
    update(0)


axbox = fig.add_axes([0.25, 0.2, 0.1, 0.035])
text_boxTrq = TextBox(axbox, "Torque", textalignment="center")
text_boxTrq.on_submit(amplsub)
text_boxTrq.set_val(350)


def powersub(expression):
    global power
    power = float(expression)
    update(0)


axbox = fig.add_axes([0.5, 0.2, 0.1, 0.035])
text_boxPwr = TextBox(axbox, "Power", textalignment="center")
text_boxPwr.on_submit(powersub)
text_boxPwr.set_val(80000)


def isub(expression):
    global i
    i = float(expression)
    update(0)


axbox = fig.add_axes([0.75, 0.2, 0.1, 0.035])
text_boxIred = TextBox(axbox, "i", textalignment="center")
text_boxIred.on_submit(isub)
text_boxIred.set_val(23)


def rkolsub(expression):
    global rkol
    rkol = float(expression)
    update(0)


axbox = fig.add_axes([0.25, 0.25, 0.1, 0.035])
text_boxRkol = TextBox(axbox, 'R_kol', textalignment="center")
text_boxRkol.on_submit(rkolsub)
text_boxRkol.set_val(0.3)


def masssub(expression):
    global mass
    mass = float(expression)
    update(0)


axbox = fig.add_axes([0.5, 0.25, 0.1, 0.035])
text_boxMass = TextBox(axbox, "Mass", textalignment="center")
text_boxMass.on_submit(masssub)
text_boxMass.set_val(1500)


def squaresub(expression):
    global square
    square = float(expression)
    update(0)


axbox = fig.add_axes([0.75, 0.25, 0.1, 0.035])
text_boxSquare = TextBox(axbox, "Square", textalignment="center")
text_boxSquare.on_submit(squaresub)
text_boxSquare.set_val(square)

def cxsub(expression):
    global cx
    cx = float(expression)
    update(0)


axbox = fig.add_axes([0.75, 0.3, 0.1, 0.035])
text_boxCx = TextBox(axbox, "Cx", textalignment="center")
text_boxCx.on_submit(cxsub)
text_boxCx.set_val(cx)


ax_reset = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(ax_reset, 'Reset', hovercolor='0.975')


def reset(event):
    snakl.reset()
    sload.reset()


button.on_clicked(reset)
radioButtonColor = 'lightgoldenrodyellow'
rax = fig.add_axes([0.05, 0.8, 0.15, 0.15], facecolor=radioButtonColor)
radio = RadioButtons(rax, ('Asphault', 'Ground', 'Plow'))


def roadtype(label):
    roaddict = {'Asphault': 0, 'Ground': 1, 'Plow': 2}
    global ind_sopr
    ind_sopr = roaddict[label]
    update(0)


radio.on_clicked(roadtype)


rax = fig.add_axes([0.05, 0.6, 0.15, 0.15], facecolor=radioButtonColor)
radio2 = RadioButtons(rax, ('HighPres', 'LowPres', 'MudTire'))


def tiretype(label):
    tiredict = {'HighPres': 0, 'LowPres': 1, 'MudTire': 2}
    global ind_scep
    ind_scep = tiredict[label]
    update(0)


radio2.on_clicked(tiretype)

rax = fig.add_axes([0.05, 0.4, 0.15, 0.15], facecolor=radioButtonColor)
radio3 = RadioButtons(rax, ('AWD', '4x2', '6x4'))


def transtype(label):
    transdict = {'AWD': 1, '4x2': 0.6, '6x4': 0.75}
    global trans_k_scep
    trans_k_scep = transdict[label]
    update(0)


radio3.on_clicked(transtype)

plt.show()
