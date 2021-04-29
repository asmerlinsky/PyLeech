import os
import tkinter as tk
from PyQt5 import QtWidgets
import sys
import matplotlib
import numpy as np

class powNorm(matplotlib.colors.Normalize):
    def __init__(self, pow=2, vmin=None, vmax=None, vmid=0, clip=False):
        self.vmin = vmin # minimum value
        self.vmax = vmax # maximum value
        self.vmid = vmid
        self.pow = pow
        self.g = lambda x, vmin,vmax, vmid, pow: np.abs((x-vmid)/(vmax-vmid))**pow * np.sign((x-vmid)/(vmax-vmid))
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        r = self.g(value, self.vmin,self.vmax, self.vmid, self.pow)

        return np.ma.masked_array((r+1)/2)


def center(toplevel):
    toplevel.update_idletasks()

    # Tkinter way to find the screen resolution
    # screen_width = toplevel.winfo_screenwidth()
    # screen_height = toplevel.winfo_screenheight()

    # PyQt way to find the screen resolution
    app = QtWidgets.QApplication([])
    screen_width = app.desktop().screenGeometry().width()
    screen_height = app.desktop().screenGeometry().height()

    size = tuple(int(_) for _ in toplevel.geometry().split('+')[0].split('x'))
    x = screen_width/2 - size[0]/2
    y = screen_height/2 - size[1]/2

    toplevel.geometry("+%d+%d" % (x, y))
    toplevel.title("Centered!")

class MyDialog(tk.Toplevel):

    def __init__(self, parent, text):
        tk.Toplevel.__init__(self, parent)
        tk.Label(self, text=text).grid(row=0, column=0, columnspan=2, padx=50, pady=10)

        b_yes = tk.Button(self, text="Ok", command=self.yes, width=8)
        b_yes.grid(row=1, column=0, padx=10, pady=10)
        b_no = tk.Button(self, text="Cancel", command=self.no, width=8)
        b_no.grid(row=1, column=1, padx=10, pady=10)
        self.attributes('-topmost', 'true')
        self.answer = None
        self.protocol("WM_DELETE_WINDOW", self.no)

    def yes(self):
        self.answer = True
        self.destroy()

    def no(self):
        self.answer = False
        self.destroy()

def popup(root, delay=60):
    d = MyDialog(root, "Pc will go to sleep in less than a minute")
    center(d)
    root.after(delay * 1000, d.yes)
    d.lift()
    root.wait_window(d)
    root.destroy()
    if sys.platform != 'linux':
        if d.answer:
            os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
        else:
            print("Ok, won't go to sleep")
    else:
        if d.answer:
            os.system("systemctl suspend")
        else:
            print("Ok, won't go to sleep")

        


def sleepPc(delay=60):
    if delay > 60:
        print("Pc will go to sleep in %i min" % int(round(delay / 60)))
    else:
        print("Pc will go to sleep in 60 s")
        delay = 60
    root = tk.Tk()
    popup(root=root, delay=delay)

    root.mainloop()



# root.after(10000, lambda: root.destroy())

