#!/usr/bin/python
import csv
if __name__ == '__main__':
    with open("params.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['theta0', 'theta1'])
        writer.writeheader()
        writer.writerow({'theta0': 0, 'theta1': 0})
        csvfile.close()
