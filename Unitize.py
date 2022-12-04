import csv

Feature = 27
minn = [10000000 for i in range(Feature + 1)]
maxn = [0 for i in range(Feature + 1)]


def main():
    for i in range(1, 7):
        with open("D:/cer_electricity/Train_Set/Retailer" + str(i) + "_PPPCA.csv", 'r') as input_file:
            input_reader = csv.reader(input_file)
            for row in input_reader:
                for j in range(1, Feature + 1):
                    if float(row[j]) > maxn[j]:
                        maxn[j] = float(row[j])
                    if float(row[j]) < minn[j]:
                        minn[j] = float(row[j])

    for i in range(1, 7):
        with open("D:/cer_electricity/Train_Set/Retailer" + str(i) + "_PPPCA.csv", 'r') as input_file:
            with open("D:/cer_electricity/Train_Set/Retailer" + str(i) + "_PPPCA_U.csv", 'w', newline="") as output_file:
                input_reader = csv.reader(input_file)
                output_writer = csv.writer(output_file)
                for row in input_reader:
                    tmp = [0 for j in range(Feature)]
                    for j in range(1, Feature + 1):
                        tmp[j - 1] = (float(row[j]) - minn[j]) / (maxn[j] - minn[j])
                    output_writer.writerow(row[0:1] + list(map(str, tmp)) + row[Feature + 1:])


if __name__ == '__main__':
    main()
