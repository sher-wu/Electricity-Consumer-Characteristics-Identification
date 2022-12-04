import numpy as np
import csv


def main():
    with open("D:/cer_electricity/CER_Electricity_Data/Survey data - CSV format/Smart meters Residential pre-trial survey data.csv", 'r') as from_file:
        with open("D:/cer_electricity/F+C_Set/Characteristic_Set.csv", 'w', newline="") as to_file:
            with open("D:/cer_electricity/Feature_Set/Total.csv", 'r') as F_file:
                with open("D:/cer_electricity/F+C_Set/F+C_Set.csv", 'w', newline="") as F_plus_C_file:
                    from_reader = csv.reader(from_file)
                    to_writer = csv.writer(to_file)
                    F_reader = csv.reader(F_file)
                    F_plus_C_writer = csv.writer(F_plus_C_file)
                    _ = next(from_reader)

                    for row in from_reader:
                        temp = [row[0]]

                        now = int(row[2])  # 1
                        if 1 <= now <= 2:
                            temp.append(1)
                        elif 3 <= now <= 5:
                            temp.append(2)
                        elif now == 6:
                            temp.append(3)
                        else:
                            temp.append(0)

                        now = int(row[3])  # 2
                        if 1 <= now <= 3:
                            temp.append(1)
                        elif 4 <= now <= 7:
                            temp.append(2)
                        else:
                            temp.append(0)

                        now = int(row[3])  # 3
                        if now == 6:
                            temp.append(1)
                        elif 1 <= now <= 7:
                            temp.append(2)
                        else:
                            temp.append(0)

                        now = int(row[4])  # 4
                        if now == 1:
                            temp.append(1)
                        elif 2 <= now <= 3:
                            temp.append(2)
                        elif now == 4:
                            temp.append(3)
                        else:
                            temp.append(0)

                        now = int(row[9])  # 5
                        if now == 3:
                            temp.append(1)
                        elif 1 <= now <= 2:
                            temp.append(2)
                        else:
                            temp.append(0)

                        now = int(row[9])  # 6
                        if now == 1:
                            temp.append(1)
                        elif 2 <= now <= 3:
                            temp.append(2)
                        else:
                            temp.append(0)

                        now = 0  # 7
                        if row[10] != " ":
                            now += int(row[10])
                        if row[12] != " ":
                            now += int(row[12])
                        if now <= 2:
                            temp.append(1)
                        else:
                            temp.append(2)

                        now = int(row[34])  # 8
                        if now == 3 or now == 5:
                            temp.append(1)
                        elif now == 2 or now == 4:
                            temp.append(2)
                        else:
                            temp.append(0)

                        now = int(row[35])  # 9
                        if now == 1 or now == 2:
                            temp.append(1)
                        elif now == 3 or now == 4:
                            temp.append(2)
                        else:
                            temp.append(0)

                        now = int(row[40])  # 10
                        if now == 1 or now == 2:
                            temp.append(1)
                        elif now == 3:
                            temp.append(2)
                        elif now == 4 or now == 5:
                            temp.append(3)
                        else:
                            temp.append(0)

                        now = int(row[36])  # 11
                        if now == 9999:
                            now = int(row[37])
                            if now <= 2:
                                temp.append(1)
                            elif now == 3:
                                temp.append(2)
                            else:
                                temp.append(3)
                        elif now >= 1999:
                            temp.append(1)
                        elif 1979 <= now < 1999:
                            temp.append(2)
                        elif 1000 <= now < 1979:
                            temp.append(3)
                        else:
                            temp.append(0)

                        now = int(row[59])  # 12
                        if now == 1:
                            temp.append(1)
                        elif 2 <= now <= 4:
                            temp.append(2)
                        else:
                            temp.append(0)

                        now = -10  # 13
                        for j in range(80, 90):
                            now += int(row[j])
                        if now <= 5:
                            temp.append(1)
                        elif 5 < now <= 8:
                            temp.append(2)
                        else:
                            temp.append(3)

                        now = -5  # 14
                        for j in range(95, 100):
                            now += int(row[j])
                        if now <= 3:
                            temp.append(1)
                        elif 3 < now <= 5:
                            temp.append(2)
                        else:
                            temp.append(3)

                        now = int(row[117])  # 12
                        if 1 <= now <= 3:
                            temp.append(1)
                        elif 4 <= now <= 5:
                            temp.append(2)
                        else:
                            temp.append(0)

                        to_writer.writerow(np.array(temp))
                        F_now = next(F_reader)
                        F_plus_C_writer.writerow(np.array(F_now + temp[1:]))


if __name__ == '__main__':
    main()
