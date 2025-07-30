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
for i in range(round(len(list1)/8)-1):
    lt=[list1[i*8+8][0]]
    for j in range(5):
        if float(list1[i*8+8][j+2])*float(list1[i*8+1][j+2])==0:
            ans=1
        else:
            ans=float(list1[i*8+8][j+2])/float(list1[i*8+1][j+2])
        lt.append(ans)
    list2.append(lt)

with open('D:/program development/hkproject/python/output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(list2)  # 写入多行