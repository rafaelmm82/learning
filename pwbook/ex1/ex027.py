#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# Body Mass Index Calculator
# @author: rafael magalhaes

# getting data from user
print("\nWelcome to BMI (Body Mass Index) Calculator.")
print("Please, give the necessary information:\n")

height = float(input("Inform your Height (centimeters): "))
weight = float(input("Inform your weight (kilograms): "))

# change unit of height from centimeters by meters
height = height/100

# doing calculation and displaying the result
bmi = (weight)/(height**2)
print("\nYour BMI index is equals to %.2f." % bmi)
