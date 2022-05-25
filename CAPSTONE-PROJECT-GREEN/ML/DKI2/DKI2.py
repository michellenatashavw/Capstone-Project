import imp
import d1pm10 as pm10
import pandas as pd
import d1pm25 as pm25
#import d1so2 as so2
#import d1co as co
#import d1o3 as o3
#import d1no2 as no2
input = '2021-11-12'
print(pm10.pm10(input))
print(pm25.pm25(input))
#so2.so2(input)
#co.co(input)
#o3.o3(input)
#no2.no2(input)