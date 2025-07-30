import csv
file1=[]
file2=[]
cities={}
with open('python/sj+fp/1.csv', mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        file1.append(row)
with open('python/sj+fp/2.csv', mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        file2.append(row)
for i in range(round((len(file1)-3)/8)):
    row=i*8+3
    name=str(file1[row][0])
    city={}
    for j in range(8):
        row=i*8+j+2
        year=int(file1[row][1])
        data=file1[row][2:9]
        city[year]=data
    cities[name]=city
for row in range(1,len(file2)):
    name=file2[row][0]
    try:
        cities[name]["info"]=file2[row][1:6]
    except Exception:
        print(str(Exception))
ans=[]
for i in cities.items():
    # print(i)
    try:
        if len(i[1])==9:
            ans.append([i[0]]+i[1]["info"]+[0.1,0.1,0.1])
            for j in range(2009,2017):
                ans.append(["",]+i[1][j])
    except Exception:
        print(i)
with open('python/sj+fp/output.csv', 'w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerows(ans)  # 写入多行