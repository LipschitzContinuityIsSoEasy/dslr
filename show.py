#!/usr/bin/env python3

import sys
import pandas as pd

def histogram(file_name):
    all_data = pd.read_csv(file_name)
    print(all_data.head())

def main():
    args = sys.argv
    if len(args) < 2:
        print("Usage: ./histogram.py <dataset.csv>")
        exit(1)

    file_name = args[1]
    try:
        histogram(file_name)

    except Exception as e:
        print(f"Erreur inattendue : {e}")
        exit(1)

if __name__ == "__main__":
    main()
