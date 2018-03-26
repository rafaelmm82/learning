#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# Distance Units Conversion
# @author: rafael magalhaes

FEET_IN_INCHE = 12
FEET_IN_YARDS = 1/3
FEET_IN_MILES = 0.000189394

# Reading the data from user
feet = int(input("Inform the distance in feets: "))

# Displaying the results
print("\nThis distance is equivalent to: " + str(feet * FEET_IN_INCHE) +
      " Inches")
print("This distance is equivalent to: " + str(feet * FEET_IN_YARDS) +
      " Yards")
print("This distance is equivalent to: " + str(feet * FEET_IN_MILES) +
      " Miles")
print("\n")
