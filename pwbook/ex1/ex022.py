#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# Area of a Triangle (Again, without base and height knoweledge)
# @author: rafael magalhaes

# importing necessary function
from math import sqrt

# Getting information from user
print("\nLet's calculate a triangle area without base and height information.")
print("Please, inform the side lenghts of the triangle:\n")
s1 = float(input("Lenght of side 1: "))
s2 = float(input("Lenght of side 2: "))
s3 = float(input("Lenght of side 3: "))

# calculating and showing the result
s = (s1 + s2 + s3)/2
area = sqrt(s * (s - s1) * (s - s2) * (s - s3))

print("\nThe area of this triangle is equals to %.2f square unites." % area)
