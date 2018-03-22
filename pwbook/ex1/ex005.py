#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# Informs the refund value to Bottles Deposits
# @author: rafael magalhaes

# Refund for each container
# smal container price (1L or less)
SCP = 0.10
# big container price (more than 1L)
BCP = 0.25

# Getting the containers size and quantities
print("Welcome to containers refund calculator.")
small_ones = int(input("How many small containers have you bring? "))
big_ones = int(input("How many big containers have you bring? "))

# Show me the money
total_ammount = (small_ones * SCP) + (big_ones * BCP)

# Display the total refound amount
print("Congratulations, you will receive ${:.2f} dollars as refund.".
      format(total_ammount))
print("Thanks to help the world getting more sustainable!")
