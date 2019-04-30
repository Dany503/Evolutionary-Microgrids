# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 10:53:49 2019

@author: arodriguez
"""
import matplotlib.pyplot as plt
import numpy as np

PV = np.array([0, 0, 0, 0, 0, 0, 0, 0, 6, 10, 15, 20, 30, 40, 40, 20, 15, 10, 8, 2, 0, 0, 0, 0])
WT = np.array([51, 51, 58, 51, 64, 51, 44, 51, 44, 51, 51, 46, 81, 74, 65, 65, 65, 51, 39, 63, 38, 66, 74, 74])
DM = np.array([67, 67, 90, 114, 120, 130, 150, 190, 200, 206, 227, 227, 250, 250, 200, 180, 160, 160, 190, 150, 100, 50, 20, 20])
P_dem = DM - PV - WT
# Batería
SOC_ini = 140
P_BT_max = 120
SOC_BT_max = 280
SOC_BT_min = 70
eta_c = 0.9
eta_d = 0.9
# DE
P_DE_min = 5
P_DE_max = 80
# Mt
P_MT_min = 20
P_MT_max = 210

def Battery_evolution(BT, SOC_ini):
    SOC = np.zeros(24)
    SOC[0] = SOC_ini
    for i in range(1,SOC.size):
        if BT[i] > 0:
            SOC[i] = SOC[i-1] - BT[i]/eta_d
        else:
            SOC[i] = SOC[i-1] - BT[i]*eta_c   
    return SOC


ind = np.array([  0.      ,     0.    ,       0.        ,  10.0025626,   10.00319704,
  10.00002718,   0.        ,  10.00295035,  10.00099501,  10.0070019,
  10.01046192,  10.00752336,  10.00071034,  10.0089759 ,  10.0037145,
  10.00130892,   0.        ,  10.00314119,  10.00345408,  10.0014105,
   0.        ,   0.        ,   0.        ,   0.        ,   0.,
   0.        ,   0.        ,  91.71570751,  62.49262169, 181.01548035,
   0.        , 149.06269822, 140.84432171, 168.11233525, 145.99180696,
 141.86662166, 161.71009274, 161.54026538,  97.28845174,  83.15913257,
   0.        ,  88.38348104, 134.66142325,  79.64367904,   0.,
   0.        ,   0.    ,       0.        ])

width = 0.7
DE = ind[:24]
MT = ind[24:]
BT = DM - PV - WT - DE - MT
BTpos = (BT>0)*BT
BTneg = (BT<0)*BT

R_up = (P_DE_max - ind[:24]) + (P_MT_max - ind[24:]) 
R_down = (ind[:24] - P_DE_min) + (ind[24:] - P_MT_min) 
for index,r in enumerate(R_down):
    if r < 0:
        R_down[index] = 0
    


# Resultados
plt.subplot()
p1 = plt.bar(np.arange(24), PV, width, color = 'r')
p2 = plt.bar(np.arange(24), WT, width, bottom = PV, color = 'b')
p3 = plt.bar(np.arange(24), DE, width, bottom = PV+WT, color = 'k')
p4 = plt.bar(np.arange(24), MT, width, bottom = PV+WT+DE, color = 'c')
p5 = plt.plot(np.arange(24), DM, '*--')
p6 = plt.bar(np.arange(24), BTpos, width, bottom = PV+WT+DE+MT, color = 'y')
plt.bar(np.arange(24), BTneg, width, color = 'y')
plt.grid()
plt.ylabel('Energy (kWh)')
plt.title('Unit commitment')
plt.legend((p1[0], p2[0], p3[0], p4[0], p6[0], p5[0]), ('PV', 'WT', 'DE', 'MT', 'Battery', 'Demand'))
plt.show()
plt.savefig("solution3.png", dpi = 100)
    
# Evolución de la batería

fig, ax1 = plt.subplots()

SOC = Battery_evolution(BT, SOC_ini)
p1 = ax1.plot(np.arange(24), SOC_BT_min*np.ones(24), 'k')
p2 = ax1.plot(np.arange(24), SOC_BT_max*np.ones(24), 'k')
p3 = ax1.bar(np.arange(24), SOC - SOC_BT_min, width, bottom = SOC_BT_min)
ax2 = ax1.twinx()
ax1.set_ylim(bottom = SOC_BT_min - 10, top = SOC_BT_max + 10)
ax2.set_ylim(bottom = -P_BT_max, top = P_BT_max)
p4 = ax2.plot(np.arange(24), BT, 'r*--')
plt.grid(True)
plt.ylabel('Energy (kWh)')
plt.title('Battery management')
plt.show()
fig.savefig("Battery.eps", dpi = 100)