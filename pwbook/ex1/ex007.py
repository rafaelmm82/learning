#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# Display the sum of the first 'n' positive integer
# @author: rafael magalhaes

# get the 'n' value
n = int(input("\nInform the 'n' integer value to make the sum: "))
total_sum = int((n*(n+1))/2)
print("\nThe total sum of integers between 1 and 'n' is = " + str(total_sum)
      + "\n")
