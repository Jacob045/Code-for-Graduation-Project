n = 42
k = 2
s = 0
while int(n / k) >= 0:
    s += int(n / k)
    n = n-int(n/k)*k
    if(int(n / k) == 0):
        s += n
        break
    print(s)
print(s)
