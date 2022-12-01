
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats



def parse_file(lines):
	
csv_filename = "stats.csv"

with open(csv_filename) as csv_file:
	csv_lines = [i.strip() for i in csv_file.readlines()[1:]]



