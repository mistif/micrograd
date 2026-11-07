scores = [("ana", 90), ("ben", 72), ("cal", 85), ("ana", 70), ("ben", 88)]

def averageScore(name, scores):
    total = 0
    count = 0
    for x,y  in scores:
        if(name == x):
            total += y
            count += 1
    return total / count

print(averageScore("ana", scores))
print(averageScore("ben", scores))
print(averageScore("cal", scores))

grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
def gridSum (grid):
    row = []
    col = []

    for x in grid:
        total = 0
        for i in x:
            total += i
        row.append(total)

    for y in range(len(grid[0])):
        total = 0
        for i in range(len(grid)):
            total += grid[i][y]
        col.append(total)
    return row, col
print(gridSum(grid))


def normalize(list):
    big = 0
    for i in list:
        if(i > big):
            big = i
    listR = []
    for i in list:
        listR.append(i / big) 
    return listR

nums = [2, 4, 8, 10]
print(normalize(nums))
print(nums)