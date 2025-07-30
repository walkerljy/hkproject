import csv
list1=[]

# 打开 CSV 文件
with open('D:/program development/hkproject/python/data.csv', mode='r', encoding='utf-8') as file:
    # 创建 csv.reader 对象
    csv_reader = csv.reader(file)
    # print(csv_reader)
    for row in csv_reader:
        list1.append(row)

# print(list[1][2])
list2=[]
for i in range(round(len(list1))-1):
    lt=[list1[i+1][0]]
    sum=0
    for j in range(5):
            sum+=float(list1[i+1][j+2])
    for j in range(5):
        if sum==0:
            ans=0
        else:
            ans=float(list1[i+1][j+2])/sum
        lt.append(ans)
    list2.append(lt)

with open('D:/program development/hkproject/python/output2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(list2)  # 写入多行