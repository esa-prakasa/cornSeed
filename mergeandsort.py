import numpy as np

x = [7, 1, 4, 6, 3, 2, 5]
y = [21, 3, 12, 18, 9, 6, 15]

b = []
for i in range(7):
    b.append([x[i],y[i]])

print(b)

# print(x)
# print(y)

# print(len(x))
# print(len(y))


a = np.array([[9, 2], [4, 5], [7, 0]])
a2 = np.array(b)

print(a)
print("-------sorted----------")
a = np.array(sorted(a, key=lambda a_entry: a_entry[0]))
print (a)
print("-----------------------")
print("-------sorted----------")

print("-----------------------")
a2 = np.array(sorted(a2, key=lambda a_entry: a_entry[0]))
print (a2)