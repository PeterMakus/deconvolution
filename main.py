#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:45:22 2019

@author: pm
"""
import os
import SignalProcessingTB as sptb
from obspy import read
import deconvolve
import synthetic as syn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
import matplotlib as mpl
from obspy.signal import filter

# plotting properties
#dirFile = os.path.dirname('main.py')

plt.style.use('PaperDoubleFig.mplstyle')

# Make some style choices for plotting
colourWheel =['#329932',
            '#ff6961',
            'b',
            '#6a3d9a',
            '#fb9a99',
            '#e31a1c',
            '#fdbf6f',
            '#ff7f00',
            '#cab2d6',
            '#6a3d9a',
            '#ffff99',
            '#b15928',
            '#67001f',
            '#b2182b',
            '#d6604d',
            '#f4a582',
            '#fddbc7',
            '#f7f7f7',
            '#d1e5f0',
            '#92c5de',
            '#4393c3',
            '#2166ac',
            '#053061']
dashesStyles = [[3,1],
            [1000,1],
            [2,1,10,1],
            [4, 1, 1, 1, 1, 1]]


# test program
M = 12 #number of seismic discontinuities
N2 = 4000
dt = 0.1 #sampling interval
SNR = 0 #SNR=0 -> noisefree #This is the ratio of maximal amplitudes
stdv = 25

st = read("NO.NC204.2013-04-16T10:52:17.749575Z.mseed")
#
#
#st[0].data = filter.bandpass(st[0].data,0.1,1.5,st[0].stats.sampling_rate)
#st[1].data = filter.bandpass(st[1].data,0.1,1.5,st[0].stats.sampling_rate)
#st[2].data = filter.bandpass(st[2].data,0.1,1.5,st[0].stats.sampling_rate)
#
#shift = 30
#h = st[1].data #radial component
#v = st[2].data #vertical component
#h = h/max(v)
#v = v/max(v)
#dt = st[1].stats.delta
#N = st[1].stats.npts
#t = np.linspace(-shift,N*dt-shift,N)
# save seismogram
#st.plot(outfile='NO.NC.pdf',title=False,show_y_UTC_label=False,title_size=0)
        
# create reflection series
R = syn.create_R(N2,M)



# create seismograms
[v,h,t] = syn.synthetic(N2,dt,R,SNR,stdv)

 #Find time shift = first value higher than maximal noise level in v
if SNR:
    shift = (v>max(v)/SNR).argmax()*dt
else:
    shift = (v!=0).argmax()*dt



# Deconvolution
[rf,it,IR] = deconvolve.it(v,h,dt,shift,2.5,0.5)


[rf_gen,IR_gen,it_gen,rej] = deconvolve.gen_it(v,h,dt,.1,shift,1,5,0.5,4000)


#[rf_mult,var] = deconvolve.multitaper(v,h,1)


# Nromalise the outputs
#R = R/max(abs(R))
#IR = IR/max(abs(IR))
#IR_gen = IR_gen/max(abs(IR_gen))
#


plt.close('all')
fig, ax = plt.subplots(1,1)#3,1)

#ax[0].plot(R[0:4000],'k')
ax.plot(IR_gen,"r")
[rf_gen,IR_gen,it_gen,rej] = deconvolve.gen_it(v,h,dt,.1,shift,1,5,0.5,400)
ax.plot(IR_gen,"k")
#ax[0].set_title('Reflection series')
#ax[0].set_title('Iterative Deconvolution')
#ax[1].set_title('Gen. Iterative Deconvolution')
ax.set_xlabel('N')
#ax[0].set_ylabel('normalised Amplitude')
#ax[0].set_xticklabels([])
#ax[1].set_xticklabels([])
#for x in ax:
x = ax
x.yaxis.set_major_formatter(ScalarFormatter())
x.yaxis.major.formatter._useMathText = True
x.yaxis.set_minor_locator(  AutoMinorLocator(5))
x.xaxis.set_minor_locator(  AutoMinorLocator(5))
x.yaxis.set_label_coords(0.63,1.01)
x.yaxis.tick_right()
#nameOfPlot = 'noisefree, M=12'
#plt.ylabel(nameOfPlot,rotation=0)
ax.legend(frameon=False, loc='upper left',ncol=2,handlelength=4)
plt.legend(['it_max=4000','it_max=400'])
plt.savefig(os.path.join('stdv25_4000.pdf'),dpi=300)
plt.show()

