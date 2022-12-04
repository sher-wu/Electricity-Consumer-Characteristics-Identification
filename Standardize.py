import csv

Feature = 27
total = [0 for i in range(Feature + 1)]
variance = [0 for i in range(Feature + 1)]


def main():
    num = 0
    for i in range(1, 7):
        with open("D:/cer_electricity/Train_Set/Retailer" + str(i) + "_PPPCA.csv", 'r') as input_file:
            input_reader = csv.reader(input_file)
            for row in input_reader:
                for j in range(1, Feature + 1):
                    total[j] += float(row[j])
                num += 1

    for j in range(1, Feature + 1):
        total[j] = total[j] / num

    for i in range(1, 7):
        with open("D:/cer_electricity/Train_Set/Retailer" + str(i) + "_PPPCA.csv", 'r') as input_file:
            input_reader = csv.reader(input_file)
            for row in input_reader:
                for j in range(1, Feature + 1):
                    variance[j] += (float(row[j]) - total[j]) ** 2

    for j in range(1, Feature + 1):
        variance[j] = variance[j] / num
        variance[j] = variance[j] ** 0.5

    for i in range(1, 7):
        with open("D:/cer_electricity/Train_Set/Retailer" + str(i) + "_PPPCA.csv", 'r') as input_file:
            with open("D:/cer_electricity/Train_Set/Retailer" + str(i) + "_PPPCA_S.csv", 'w', newline="") as output_file:
                input_reader = csv.reader(input_file)
                output_writer = csv.writer(output_file)
                for row in input_reader:
                    tmp = [0 for j in range(Feature)]
                    for j in range(1, Feature + 1):
                        tmp[j - 1] = (float(row[j]) - total[j]) / variance[j]
                    output_writer.writerow(row[0:1] + list(map(str, tmp)) + row[Feature + 1:])


if __name__ == '__main__':
    main()
