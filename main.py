# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from tabular_lib import *
from pathlib import Path

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = Path('.')
    tabdata = TabData(path/'TrainAndValid.csv', 'SalePrice')
    print(tabdata.data.head(5))
    tabdata.handling_dates('saledate')
    tabdata.data

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
