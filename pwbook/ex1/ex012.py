#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# Georeference distance calculator
# @author: rafael magalhaes

# import math functions
from math import *

# Getting the latitute and longitude points from user
print("\nWelcome to Earth distance calculator! (degrees only)\n")
lat_1 = float(input("Inform the Latitude of the origin point: "))
lon_1 = float(input("Inform the Longitude of the origin point: "))
lat_2 = float(input("\nInform the Latitude of the destination point: "))
lon_2 = float(input("Inform the Longitude of the destination point: "))

# converting degrees to radians
lat_1 = radians(lat_1)
lon_1 = radians(lon_1)
lat_2 = radians(lat_2)
lon_2 = radians(lon_2)

# distance calculation
distance = 6371.01 * acos(sin(lat_1)*sin(lat_2) + cos(lat_1)*cos(lat_2) *
                          cos(lon_1 - lon_2))
# Displey the results
print("\nThe distance between points is: {:.2f} kilometers".format(distance))
print("\n")
