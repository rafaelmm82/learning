#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# Units of Time - Convert a duration time in seconds
# @author: rafael magalhaes

# getting data from user
print("\nWelcome to time duration convertion.")
print("Please, informe the duration time as asked below..")
days = int(input("How many Days: "))
hours = int(input("How many hours: "))
minutes = int(input("How many minutes: "))
seconds = int(input("How many seconds: "))

# calculation and result display
total = (days * 86400) + (hours * 3600) + (minutes * 60) + seconds
print("\nThe total time in seconds is equals to %i seconds." % total)
