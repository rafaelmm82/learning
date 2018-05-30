# -*- coding: utf-8 -*-
"""
Notes and tries on Chaper 03 (Think Stats 2, Allen B. Downey)

Self-study on statistics using pyhton
@author: Github: @rafaelmm82
"""

import thinkstats2
import thinkplot
import nsfg
import math
import first
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pmf= thinkstats2.Pmf([1, 2, 2, 3, 5])
pmf

pmf.Prob(2)

pmf[2]

pmf.Incr(2, 0.2)
pmf.Prob(2)

pmf.Mult(2, 0.5)
pmf

pmf.Total()

pmf.Normalize()
pmf.Total()
pmf


live, firsts, others = first.MakeFrames()

first_pmf = thinkstats2.Pmf(firsts.prglngth, label='first')
other_pmf = thinkstats2.Pmf(others.prglngth, label='other')

width = 0.45
thinkplot.PrePlot(2, cols=2)
thinkplot.Hist(first_pmf, align='right', width=width)
thinkplot.Hist(other_pmf, align='left', width=width)
thinkplot.Config(xlabel='weeks',
                 ylabel='probability',
                 axis=[27, 46, 0, 0.6])
thinkplot.PrePlot(2)
thinkplot.SubPlot(2)
thinkplot.Pmfs([first_pmf, other_pmf])
thinkplot.Show(xlabel='weeks',
               axis=[27, 46, 0, 0.6])

weeks = range(35, 46)
diffs = []
for week in weeks:
    p1 = first_pmf.Prob(week)
    p2 = other_pmf.Prob(week)
    diff = 100 * (p1 - p2)
    diffs.append(diff)

thinkplot.Bar(weeks, diffs)


d = {7: 8, 12: 8, 17: 14, 22: 4, 
     27:6, 32: 12, 37: 8, 42: 3,47: 2}

pmf = thinkstats2.Pmf(d, label='actual')
print('mean', pmf.Mean())

def BiasPmf(pmf, label=''):
    new_pmf = pmf.Copy(label=label)

    for x, p in pmf.Items():
        new_pmf.Mult(x, x)
        
    new_pmf.Normalize()
    return new_pmf


biased_pmf = BiasPmf(pmf, label='observed')
thinkplot.PrePlot(2)
thinkplot.Pmfs([pmf, biased_pmf])
thinkplot.Show(xlabel='class size', ylabel='PMF')

def UnbiasPmf(pmf, label):
    new_pmf = pmf.Copy(label=label)

    for x, p in pmf.Items():
        new_pmf.Mult(x, 1.0/x)
        
    new_pmf.Normalize()
    return new_pmf

unbiased_pmf = UnbiasPmf(pmf, label='better_values')
thinkplot.PrePlot(2)
thinkplot.Pmfs([pmf, unbiased_pmf])
thinkplot.Show(xlabel='class size', ylabel='PMF')


array = np.random.randn(4, 2)
df = pd.DataFrame(array)
df

columns = ['A', 'B']

df = pd.DataFrame(array, columns=columns)
df

index = ['a', 'b', 'c', 'd']
df = pd.DataFrame(array, columns=columns, index=index)
df

df.loc['a']

df.iloc[0]

indices = ['a', 'c']
df.loc[indices]

df['a':'c']

df[0:2]