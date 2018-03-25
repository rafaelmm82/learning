#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# A program that calculate an area of a room with measures given by user
# @author: rafael magalhaes


# Asking the operands and doing the calculus
print("Welcome to area calculator.")
width = input("Input the width of the room (meters): ")
length = input("Input the length of the room (meters): ")
area = float(width) * float(length)

# Two options to print the result

# First: Format before print
# area = "{:.2f}".format(area)
# print("The area (in meters) of the room is: " + str(area) + " square meters")

# Second: Format inline printing
print("The area (in meters) of the room is: {:.2f} square meters".format(area))
