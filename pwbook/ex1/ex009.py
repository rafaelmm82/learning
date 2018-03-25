#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# Financial calculator of compound interest
# @author: rafael magalhaes

# interest rate in percent
IR = 4/100

# Getting the information of intial investment value
money = float(input("How much money would you like to invest: "))


# Calculate and display the invest evolution through three years
interest = money * IR

print("\n ====== SUMARY OF INVESTMENT ======")
print("{:10.2f} - Initial investment amount".format(money))
print("{:10.2f} - Interest earn of 1st year".format(interest))
money += interest
print("{:10.2f} - First year new Balance".format(money))
print("\n ----------------------------------")
interest = money * IR
money += interest
print("{:10.2f} - Interest earn of 2sd year".format(interest))
print("{:10.2f} - Second year new Balance".format(money))
print("\n ----------------------------------")
interest = money * IR
money += interest
print("{:10.2f} - Interest earn of 3th year".format(interest))
print("{:10.2f} - Third year new Balance".format(money))

print("\n\nThank you for saving your money with us!\n")
