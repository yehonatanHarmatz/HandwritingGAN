# This is a sample Python script.
import torch
import cv2
import pandas as pd
import lmdb
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    print("this is i told u so ron file")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(torch.__version__)
    print(lmdb.__version__)
    print(pd.DataFrame(columns=[*[f"img{i}" for i in range(15)], "writer"]))
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
