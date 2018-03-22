#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# A program that calculate an area of a farm with measures given by user
# The result is show in Acres
# @author: rafael magalhaes

SQFT_PER_ACRES = 43560

# Obtain the input from the user
print("Hello farmer, lets calculte your farmer area in acres.")
width = float(input("Input the 'width' of your property in 'feets': "))
length = float(input("Input the 'length' of you property in 'feets': "))

# Calculate the area and convert to Acres units
area = width * length
acres = area/SQFT_PER_ACRES

# Display the result
print("What a great property! It has an area of {:.2f} acres.".
      format(acres))
