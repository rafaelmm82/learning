##!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# Calculates the area of a circle and sphere
# @author: rafael magalhaes

from math import pi

# Read the information of the radius
r = float(input("\nInform the 'r' radius lenght: "))

# calculating and displaying information
circle_area = pi*pow(r, 2)
print("\nThe area of a cirle with 'r' radius is: %.2f square units"
      % circle_area)
sphere_area = (4/3) * pi * pow(r, 3)
print("The area of a sphere with 'r' radius is: %.2f cubic units\n"
      % sphere_area)
