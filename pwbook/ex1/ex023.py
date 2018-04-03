#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# Area of a regular polygono
# @author: rafael magalhaes

# importing necesseray function
from math import tan, pi

# getting data from user
print("\nWelcome to regular polygono area calculator.")
n = int(input("Please, inform how number of sides have your polygono: "))
size = float(input("Please, inform the lenght of the sides: "))

# doing calculation and displaying the result
area = (n * size**2)/(4 * tan(pi/n))

print("\nThe area of your poygono is equals to %.2f square units.\n" % area)
