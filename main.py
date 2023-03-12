# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import EOS
P = 14.8; T = 323.15; P_c = 48.08; T_c = 305.3; w = 0.1
z, v = EOS.virial(P, T, P_c, T_c, w)
print(f"Compressibility factor: {z:.4f}")
print(f"Molar volume: {v:.5f} [m3/mol]")