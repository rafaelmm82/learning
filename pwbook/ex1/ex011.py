#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# Calculates the Fuel Efficiency From USA to Canada Display Units Convention
# @author: rafael magalhaes

# Units converstions
# Litters in a Gallon
L_in_G = 3.78541
# Metters in a Mile
Me_in_Mi = 1609.34

# Asking for data from user
print("\nWelcome to Fuel Efficiency translator (USA -> CA)\n")
mpg = float(input("How many milles your car run with 1 gallon (MPG)? "))

# convertion to miles per litters
millesPL = mpg/L_in_G

# convertion to metters per litters
mettersPL = millesPL * Me_in_Mi

# Litters per 100km (100000 metters)
lp100km = 100000/mettersPL

# Display result
print("So, your car needs {:.2f} Litters to run 100km!\n".format(lp100km))
