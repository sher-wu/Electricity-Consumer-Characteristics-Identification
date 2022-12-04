import numpy as np
import math
import csv

MAX = 10000
ID_ST = 1000
In = "D:/cer_electricity/File1.txt/File1.txt"
Out = "D:/cer_electricity/F+C_Set/Feature_File1.csv"
St_Date = 194  # start from Monday
Ed_Date = 732
Data_Len = 1005  # 1005 for 1-5, 1450 for 6


def readin(data):
    total = 0
    with open(In) as file:
        while True:
            if total % 1000000 == 0:
                print("Readin " + str(total))
            line = file.readline()
            if not line:
                break
            a, b, c = line.split(" ")
            # if int(a) >= 2000 or int(int(b) / 100) >= 800 or int(b) % 100 >= 50:
            #     print(a + ' ' + b + ' ' + c + ' ' + str(total))
            #     sys.exit()

            id = int(a)
            day = int(int(b) / 100)
            hour = int(b) % 100
            if hour > 48:
                day += 1
                hour -= 48
                if hour > 48:
                    day += 1
                    hour -= 48

            data[id - ID_ST][day][hour] = float(c)
            total += 1
    print("Readin Finished")


def func(data, feature):

    for i in range(Data_Len):

        feature[i][74][0] = i + ID_ST

        if i % 10 == 0:
            print("Func " + str(i))

        valid = 0
        week_temp = -2  # -1 means now_week is incomplete, -2 means initial
        week_data = []

        feature[i][4][0] = MAX
        for j in range(24, 34):
            feature[i][j][0] = MAX

        for j in range(St_Date, Ed_Date):
            if (j + 2) % 7 < 5:  # 2009/1/1 is Thursday, wd=0 weekdays
                wd = 0
            else:
                wd = 1

            if (j + 2) % 7 == 0:
                if week_temp >= 0:
                    feature[i][2][0] += week_temp
                    feature[i][2][1] += 7
                    week_data.append(week_temp)
                    if week_temp > feature[i][3][0]:
                        feature[i][3][0] = week_temp
                    if week_temp < feature[i][4][0]:
                        feature[i][4][0] = week_temp
                week_temp = -2

            temp = 0
            for k in range(1, 49):
                if data[i][j][k] == -1:
                    temp = -1
                    break
                else:
                    temp += data[i][j][k]

            if temp != -1:
                feature[i][wd][0] += temp
                feature[i][wd][1] += 1
                data[i][j][0] = temp
                valid = 1

                if week_temp != -1:
                    if week_temp == -2:
                        week_temp = 0
                    week_temp += temp

                if temp > feature[i][wd + 14][0]:
                    feature[i][wd + 14][0] = temp

                if temp < feature[i][wd + 24][0]:
                    feature[i][wd + 24][0] = temp
            else:
                data[i][j][0] = -1
                week_temp = -1

            count_st = [5, 13, 23, 37]
            count_ed = [11, 21, 29, 45]

            for period in range(4):
                temp = 0
                for k in range(count_st[period], count_ed[period]):
                    if data[i][j][k] == -1:
                        temp = -1
                        break
                    else:
                        temp += data[i][j][k]
                if temp != -1:
                    feature[i][wd + 6 + period * 2][0] += temp
                    feature[i][wd + 6 + period * 2][1] += 1

                    if temp > feature[i][wd + 16 + period * 2][0]:
                        feature[i][wd + 16 + period * 2][0] = temp

                    if temp < feature[i][wd + 26 + period * 2][0]:
                        feature[i][wd + 26 + period * 2][0] = temp

        if valid == 0:
            feature[i][0][0] = -1
            continue

        for j in [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13]:
            feature[i][j][0] = np.divide(feature[i][j][0], feature[i][j][1])

        feature[i][36][0] = np.divide(feature[i][2][0], feature[i][3][0])
        for j in range(37, 45):
            feature[i][j][0] = np.divide(feature[i][j - 31][0],
                                         feature[i][j - 21][0])

        feature[i][45][0] = np.divide(feature[i][24][0], feature[i][0][0])
        feature[i][46][0] = np.divide(feature[i][25][0], feature[i][1][0])
        feature[i][47][0] = np.divide(feature[i][12][0], feature[i][10][0])
        feature[i][48][0] = np.divide(feature[i][13][0], feature[i][11][0])
        feature[i][49][0] = np.divide(feature[i][8][0], feature[i][10][0])
        feature[i][50][0] = np.divide(feature[i][9][0], feature[i][11][0])
        feature[i][51][0] = np.divide(feature[i][6][0], feature[i][10][0])
        feature[i][52][0] = np.divide(feature[i][7][0], feature[i][11][0])
        feature[i][53][0] = np.divide(feature[i][0][0], feature[i][2][0])
        feature[i][54][0] = np.divide(feature[i][1][0], feature[i][2][0])

        Count = 0
        GE_half = 0
        GE_1 = 0
        GE_2 = 0
        GT_mean = 0
        halfhour_average = feature[i][2][0] / 48
        max_value = -1
        min_value = MAX

        for j in range(St_Date, Ed_Date):
            for k in range(1, 49):
                if data[i][j][k] != -1:
                    Count += 1
                    if data[i][j][k] >= 0.5:
                        GE_half += 1
                        if data[i][j][k] >= 1:
                            GE_1 += 1
                            if data[i][j][k] >= 2:
                                GE_2 += 1
                    if data[i][j][k] > halfhour_average:
                        GT_mean += 1

                    if data[i][j][k] > max_value:
                        max_value = data[i][j][k]
                        feature[i][61][0] = feature[i][62][0] = j * 100 + k
                    elif data[i][j][k] == max_value:
                        feature[i][62][0] = j * 100 + k

                    if data[i][j][k] < min_value:
                        min_value = data[i][j][k]
                        feature[i][63][0] = feature[i][64][0] = j * 100 + k
                    elif data[i][j][k] == min_value:
                        feature[i][64][0] = j * 100 + k

        feature[i][57][0] = GE_half / Count
        feature[i][58][0] = GE_1 / Count
        feature[i][59][0] = GE_2 / Count
        feature[i][60][0] = GT_mean / Count

        week_average = feature[i][2][0] * 7
        two_central = 0
        three_central = 0
        four_central = 0
        week_round = []
        week_len = len(week_data)

        if week_len == 0:
            continue

        feature[i][72][0] = np.corrcoef(week_data[1:],
                                        week_data[:week_len - 1])[0, 1]
        # feature[i][72][0] = np.correlate(week_data[1:] - np.mean(week_data[1:]), week_data[:week_len - 1] - np.mean(week_data[:week_len - 1]))[0] / np.std(week_data[1:]) / np.std(week_data[:week_len - 1]) / (week_len - 1)

        week_data.sort()

        for j in week_data:
            two_central += (j - week_average)**2
            three_central += (j - week_average)**3
            four_central += (j - week_average)**4
            week_round.append(int(j / 5))

        feature[i][65][0] = two_central / week_len

        temp = int((week_len + 1) / 4) - 1
        feature[i][66][0] = week_data[temp if temp >= 0 else 0]

        temp = int((week_len + 1) * 3 / 4) - 1
        feature[i][67][0] = week_data[temp if temp >= 0 else 0]

        temp = int((week_len + 1) / 2) - 1
        feature[i][68][0] = week_data[temp if temp >= 0 else 0]

        feature[i][69][0] = three_central / week_len / (
            (two_central / week_len)**1.5)

        feature[i][70][0] = four_central / week_len / (
            (two_central / week_len)**2)

        pre = -1
        temp_num = 0
        entropy_5 = 0

        for j in week_round:
            if j != pre and temp_num != 0:
                entropy_5 += -temp_num / week_len * math.log(
                    temp_num / week_len)
                pre = j
                temp_num = 1
            else:
                temp_num += 1

        feature[i][71][0] = entropy_5

    for i in range(Data_Len):
        for j in range(75):
            if math.isinf(feature[i][j][0]) or math.isnan(feature[i][j][0]):
                feature[i][j][0] = 0
        if feature[i][4][0] == MAX:
            feature[i][4][0] = 0
        for j in range(24, 34):
            if feature[i][j][0] == MAX:
                feature[i][j][0] = 0

    print("Func Finished")


def printout(feature):
    # required = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 57, 58, 59, 60, 61, 62, 63, 64,
    #  65, 66, 67, 68, 69, 70, 71, 72, 73]
    with open(Out, 'w', encoding='utf-8', newline="") as file:
        csv_writer = csv.writer(file)
        for i in range(Data_Len):
            if i % 100 == 0:
                print("Printout " + str(i))

            if feature[i][0][0] != -1:
                csv_writer.writerow(
                    np.array(feature[i][74:] + feature[i][:5] +
                             feature[i][6:34] + feature[i][36:55] +
                             feature[i][57:73])[:, 0])
    print("Printout Finished")


def main():
    data = [[[-1 for k in range(50)] for j in range(800)]
            for i in range(Data_Len)]
    feature = [[[0 for k in range(2)] for j in range(75)]
               for i in range(Data_Len)]
    readin(data)
    func(data, feature)
    printout(feature)


if __name__ == '__main__':
    main()
