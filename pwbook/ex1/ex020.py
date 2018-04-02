#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# Ideal Gas Law
# @author: rafael magalhaes

# constant of ideal gas in Joules/(mol * K)
R = 8.314

# getting the user data information
print("\nWelcome to Gas Amount calculator (ideal gas law).")
print("Please, inform the asked necessary data...\n")
preassure = float(input("The preassure in Pascals: "))
volume = float(input("The volume in Litters: "))
temperature = float(input("The temperature em Celcius: "))

# doing convertions and calculations
kelvin = temperature + 273.15
n = (preassure * volume)/(R * kelvin)

# displaying the results
print("\nThe amount of gas on these conditions are %.2f moles.\n" % n)
