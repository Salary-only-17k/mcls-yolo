# -*- coding: UTF-8 -*-
__author__ = 'zy'
__time__ = '2019/8/31 16:55'
import matplotlib.pyplot as plt
 
from matplotlib import cm
from numpy import linspace
 
start = 0.0
stop = 1.0
number_of_lines= 7
cm_subsection = linspace(start, stop, number_of_lines)
 
print(cm_subsection)
 
colors = [ cm.jet(x) for x in cm_subsection ]
 
print(colors)
for i, color in enumerate(colors):
    plt.axhline(i, color=color)
 
plt.ylabel('Line Number')
plt.show()