#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# Height Units Conversion
# @author: rafael magalhaes

# Getting the information in Royal Units
print("Welcome to height measure conversion (from Royal to IS)")
print("Please inform your real height in feets and inches...")
print("\nExample: \nHeight in feets: 6\nAnd Inches: 3\n")
feets = int(input("Heights in feets: "))
inches = int(input("And Inches: "))

# conversion and display information
height_in_centimeters = ((feets*12)+inches)*2.54

print("\nYou have exactly {:.2f} centimeters!".format(height_in_centimeters))
