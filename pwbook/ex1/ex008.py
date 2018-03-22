#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# Calculates the total weight of a order
# @author: rafael magalhaes

# the weight of produts
# widget weight in grams
WW = 75
# gizmos weight in grams
GW = 112

# Getting the quantities
widgets = int(input("How many widgets was ordered: "))
gizmos = int(input("How many gizmos was ordered: "))

# doing calculation
total_weight = int((widgets * WW) + (gizmos * GW))

# display the result
print("\nThe total weight of the order is: " + str(total_weight) + " grams.\n")
