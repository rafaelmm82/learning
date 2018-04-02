#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# Free Fall - Determine the final velocity that a dropped object touch at the ground
# @author: rafael magalhaes

# importing the necessary functions
from math import sqrt

# reading the information from user
print("\nWelcome to final velocity calculator. Please, informe the data..\n")
d = float(input("The height that object was dropped (meters): "))
print("Because it was dropped we assume that initial velocity is 0 m/s")
print("And because we are at Earth planet we assume acceleration as 9.8 m/s2")

# doing calculation
final_velocity = sqrt(0**2 + 2*9.8*d)

# Displaying the result
print("\nThe velocity that the object touch at the ground is {:.2f} m/s\n".
      format(final_velocity))
