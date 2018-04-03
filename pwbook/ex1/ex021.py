#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# Area of a Triangle
# @author: rafael magalhaes

# reading the data from user
print("\nLet's calculate an area of a triangle..\n")
b = float(input("What is the lenght of the base: "))
h = float(input("What is the height of the triangle: "))

# calculation and showing the result
area = (b * h)/2
print("\nThe triangle area is equals to {:.2f} square units\n".format(area))
