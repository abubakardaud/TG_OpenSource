from sys import stdin, stdout

n = stdin.readline()
n = int(n)
arr = []
line = stdin.readline()
while line:
    arr.append(int(line))
    line = stdin.readline()

answer = 0
visit = dict()
arr_index = [*enumerate(arr)]
for num in range(n):
    visit[num] = False

arr_index.sort(key=lambda x: x[1])

for i in range(n):

    if visit[i] or arr_index[i][0] == i:
        continue
    cz = 0
    j = i
    node = visit[j]
    while not node:
        visit[j] = True
        j = arr_index[j][0]
        node = visit[j]
        cz += 1

    if cz > 0:
        answer += (cz - 1)

print(answer)



