#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# Change Machine to auto checkout shopper
# @author: rafael magalhaes

# Getting the amount of cents does the user have
print("Welcome to self checkout machine and changer.\n")
print("Please, inform how much do you have to receive in cents...\n")
total = int(input("total amount in cents: "))

# creating variables
toonies = 0
loonies = 0
halfs = 0
quarters = 0
dimes = 0
nickels = 0
pennies = 0

# doing calculation
if total >= 200:
    toonies = total // 200
    total = total % 200

if total >= 100:
    loonies = total // 100
    total = total % 100

if total >= 50:
    halfs = total // 50
    total = total % 50

if total >= 25:
    quarters = total // 25
    total = total % 25

if total >= 10:
    dimes = total // 10
    total = total % 10

if total >= 5:
    nickels = total // 5
    total = total % 5

if total >= 1:
    pennies = total

# Display results
print("Your change would be like...\n")
print("%i (CA$ 2.00) Toonies" % toonies)
print("%i (CA$ 1.00) Lonnies" % loonies)
print("%i (CA$ 0.50) Halfs" % halfs)
print("%i (CA$ 0.25) Quarters" % quarters)
print("%i (CA$ 0.10) Dimes" % dimes)
print("%i (CA$ 0.05) Nickles" % nickels)
print("%i (CA$ 0.01) Pennies" % pennies)
print("\nThank you for buying with us!\n")
