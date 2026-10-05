import numpy as np

np.set_printoptions(threshold=np.inf, suppress=True)

arr = np.loadtxt(r"C:\Users\yugoy\OneDrive\Desktop\Coding\AutoHotKey\CarInputInfo.txt",delimiter="\t",encoding="utf-8-sig")

np.set_printoptions()

fLow = arr[0:20,:]
fHigh = arr[20:40,:]
wotLow = arr[40:60,:]
wotHigh = arr[60:80,:]
ignLow = arr[80:100,:]
ignHigh = arr[100:120,:]


# print(fLow ,"\n")
# print(fHigh ,"\n")
# print(fHigh,"\n")
# print(wotLow ,"\n")
# print(wotHigh ,"\n")
# print(ignLow, "\n")
# print(ignHigh ,"\n")

def lesser(array):
    for i in range(len(array)):
        for j in range(len(array[i]) - 1):
            if array[i][j] > array[i][j+1]:
                return False
    return True

def greater(array):
    for i in range(len(array)):
        for j in range(len(array[i]) - 1):
            if array[i][j] < array[i][j+1]:
                print(i,j,array[i][j],array[i][j+1])
                return False
    return True

print(lesser(fLow))
print(lesser(fHigh))
print(greater(ignLow))
print(greater(ignHigh))

