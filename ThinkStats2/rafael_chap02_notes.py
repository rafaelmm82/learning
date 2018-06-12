# -*- coding: utf-8 -*-
"""
Notes and tries on Chaper 02 (Think Stats 2, Allen B. Downey)

Self-study on statistics using pyhton
@author: Github: @rafaelmm82
"""

import thinkstats2
import thinkplot
import nsfg
import math
import matplotlib.pyplot as plt
import numpy as np


hist = thinkstats2.Hist([1, 2, 2, 3, 5])
hist

hist.Freq(2)

hist[2]

hist.Freq(4)

hist.Values()

for val in sorted(hist.Values()):
    print(val, hist[val])


for val, freq in hist.Items():
    print(val, freq)

thinkplot.Hist(hist)
thinkplot.Config(xlabel='value', ylabel='frequency', legend='false')

preg = nsfg.ReadFemPreg()
live = preg[preg.outcome == 1]

# 2.1
hist = thinkstats2.Hist(live.birthwgt_lb, label='birthwgt_lb')
thinkplot.Hist(hist)
thinkplot.Config(xlabel='pounds', ylabel='frequency')

# 2.2
hist = thinkstats2.Hist(live.birthwgt_oz, label='birthwgt_oz')
thinkplot.Hist(hist)
thinkplot.Config(xlabel='ounces', ylabel='frequency')

# 2.3
hist = thinkstats2.Hist(np.floor(live.agepreg), label='agepreg')
thinkplot.Hist(hist)
thinkplot.Config(xlabel='agepreg', ylabel='frequency')

# 2.4
hist = thinkstats2.Hist(live.prglngth, label='prglength')
thinkplot.Hist(hist)
thinkplot.Config(xlabel='pregnancy weeks', ylabel='frequency')


for weeks, freq in hist.Smallest(10):
    print(weeks, freq)

for weeks, freq in hist.Largest(10):
    print(weeks, freq)

# First Babies

firsts = live[live.birthord == 1]
others = live[live.birthord != 1]

firsts_hist = thinkstats2.Hist(firsts.prglngth)
others_hist = thinkstats2.Hist(others.prglngth)


width = 0.45
thinkplot.PrePlot(2)
thinkplot.Hist(firsts_hist, align='right', width=width)
thinkplot.Hist(others_hist, align='left', width=width)
thinkplot.Config(xlabel='weeks', ylabel='Count', xlim=[27, 46])



plt.hist(firsts_hist, 20, color='r')
plt.hist(others_hist, 20, color='b')
plt.show()

def CohenEffectSize(group1, group2):
    diff = group1.mean() - group2.mean()

    var1 = group1.var()
    var2 = group2.var()
    n1, n2 = len(group1), len(group2)

    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / math.sqrt(pooled_var)
    return d

cohendiff = CohenEffectSize(firsts.prglngth, others.prglngth)


"""
 Exerc 2.1
 Based on about nine thousands pregnancies records from a great american census
 , a group of data scientits conducteded some analysis to understanding if
 there are any traces or elements to affirm or not that firsts babies come
 or not early them others. The conclusions, based on statistical derivative
 measurements, the fact graph analysis and standard and numerical accepted
 procedures, they concluded that there are no effective evidence that first
 babies tend to born too early ro too later them others.

"""