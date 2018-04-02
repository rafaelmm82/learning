#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# Heat Capacity - Energy necessery to change water temperature
# @author: rafael magalhaes

# C - Water Heat Capacity (Joules/(gram*degree[Celcius])
C = 4.186
# KWH - KiloWatt-Hour in Joules
KWH = 3600000
# PWH - Price of one KiloWatt-Hour in US$ cents
PKWH = 8.9

# Read the mass and the delta temperature
mass = float(input("\nHow much water do you have? (milliliters): "))
delta_t = float(input("How much degrees do you want to change (+-Celcius): "))

# Energy calculation
q = mass * C * delta_t
# Cost of Energy in dollars
cost = (PKWH * (q/KWH)) / 100

# Display the result
print("\nYou will need {:.2f} Joules...".format(q))
print("to change {:.2f} millitres of water...".format(mass))
print("by {:.2f} degrees Celcius.\n".format(delta_t))

print("It will cost about US${:.2f} dollars to heat this!".format(cost))
print("* considering the price of 8.9 cents per killowatts hour.\n")
