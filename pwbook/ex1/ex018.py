#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# Calculates the Volume of a Cylinder
# @author: rafael magalhaes

# necessary libraries
from math import pi

# Read the radius and the height of a cylinder by user input
print("\nHello, let's calculate the volume of a Cylinder..\n")
height = float(input("The height of the cylinder is: "))
r = float(input("The radius of the cylinder is: "))

# doing calculation
area = 2 * pi * r**2
volume = area * height

# Displaying the result
print("\nThe Volume of the cilinder is {:.1f} cubic units\n".format(volume))
