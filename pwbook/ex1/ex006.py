#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# Calculates and display a restaurant bill with tax and tips
# @author: rafael magalhaes


# Tax Rate in percent
TAX = 8/100

# Tip percentage
TIP = 18/100

# Getting the meal value
print("Welcome to grand total restaurant app.")
meal_value = float(input("How much was the ordered total meal: "))

# calculate and display the grand total amount
tax_value = meal_value * TAX
tip_value = meal_value * TIP
grand_total = meal_value + tax_value + tip_value

print("\n\nGreat Meal! Your meal and total amount is:\n")
print("${:10,.2f} - Net meal ordered value".format(meal_value))
print("${:10,.2f} - State TAX of meal".format(tax_value))
print("${:10,.2f} - Suggested TIP of the meal".format(tip_value))
print("---------------------")
print("${:10,.2f} - Grand TOTAL\n".format(grand_total))
