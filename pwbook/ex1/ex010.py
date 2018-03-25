#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# A program that execute a lot of math operations on integers given by user
# @author: rafael magalhaes

# importing the necessary functions from libraries
from math import log10

# Reading the integer from user
print("Welcome to math operations. Please, input two integer numbers...\n")
a = int(input("a = "))
b = int(input("b = "))

# Computing and displaying the results
print("\nThe sum of 'a' and 'b' is = %i" % (a+b))
print("The difference when 'b' is subtracted from 'a' = %i" % (a - b))
print("The product of 'a' and 'b' is = %i" % (a * b))
print("The quotient when 'a' is divided by 'b' is = %i" % (a // b))
print("The reminder when 'a' is divided by 'b' is = %i" % (a % b))
print("The result of 'log10(a)' is = %f" % (log10(a)))
print("The result of (a)ˆ2 is = %i" % (a ** b))
print("\n")
