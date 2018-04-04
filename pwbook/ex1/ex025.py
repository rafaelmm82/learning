#!/usr/local/bin/python
# -*- coding: utf-8 -*-
##
# Units of Time (again) - convert a duration in seconds to D:HH:MM:SS
# @author: rafael magalhaes

# constants already known about time
SECONDS_IN_DAY = 86400
SECONDS_IN_HOUR = 3600
SECONDS_IN_MINUTE = 60

# getting the amount of seconds from user
print("\nWelcome to time duration convertion.")
total = int(input("Please, inform the number of seconds in duration time: "))

# transformation and result display
dd = total // SECONDS_IN_DAY
total = total % SECONDS_IN_DAY

hh = total // SECONDS_IN_HOUR
total = total % SECONDS_IN_HOUR

mm = total // SECONDS_IN_MINUTE
total = total % SECONDS_IN_MINUTE

ss = total

print("\nThe duration time is equivalent to %i:%02i:%02i:%02i." % (dd, hh, mm,
      ss))
