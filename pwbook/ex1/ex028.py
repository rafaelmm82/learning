#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# Wind Chill Index Calculator
# @author: rafael magalhaes

# - Ta temperature of the air
# - V velocity of the wind
# Obs.: This formula is valid only on temperatures less than 10 Celsius
# dregrees and winds faster than 4.8 km/h

print("\nWelcome to WCI - Wind Chill Index Calculator.")
print("Please, input the necessery information...\n")

Ta = float(input("Whats is the value of temperature (Celcius): "))
V = float(input("Whats it the volicity of the wind (km/h): "))

# calculating and showing the result

WCI = 13.12 + 0.6215*Ta - 11.37*pow(V, 0.16) + 0.3965*Ta*pow(V, 0.16)
print("\nThw WCI - Wind Chill Index is equals to %i." % round(WCI))
