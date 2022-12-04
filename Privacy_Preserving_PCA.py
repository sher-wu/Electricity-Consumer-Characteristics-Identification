import csv
import numpy as np


Feature = 27


def func1():

    total_a = [[0 for i in range(68)] for j in range(68)]
    total_b = [0 for i in range(68)]
    total_c = 0

    for i in range(1, 6):
        with open("D:/cer_electricity/Train_Set/Retailer" + str(i) + ".csv", 'r') as input_file:
            input_reader = csv.reader(input_file)
            tmp_a = [[0 for i in range(68)] for j in range(68)]
            tmp_b = [0 for i in range(68)]
            tmp_c = 0

            for row in input_reader:
                tmp = np.array(list(map(float, row[1:69]))).reshape(1, 68)
                tmp_a += np.dot(tmp.T, tmp)
                tmp_b += tmp
                tmp_c += 1

            total_a += tmp_a
            total_b += tmp_b
            total_c += tmp_c

    return [total_a, total_b, total_c]


def func2(t):
    for i in range(1, 7):
        with open("D:/cer_electricity/Train_Set/Retailer" + str(i) + ".csv", 'r') as input_file:
            with open("D:/cer_electricity/Train_Set/Retailer" + str(i) + "_PPPCA.csv", 'w', newline="") as output_file:
                input_reader = csv.reader(input_file)
                output_writer = csv.writer(output_file)

                for row in input_reader:
                    tmp = np.array(list(map(float, row[1:69]))).reshape(1, 68)
                    tmp_2 = np.dot(tmp, t)
                    output_writer.writerow(row[0:1] + list(map(str, tmp_2[0])) + row[69:])


def main():
    a, b, c = func1()
    cov = a - (np.dot(b.T, b) / c)
    u, sigma, vt = np.linalg.svd(cov)
    func2(u[:, :Feature])
    with open("D:/cer_electricity/Train_Set/ProjectionMatrix_PPPCA.csv", 'w', newline="") as output_file:
        output_writer = csv.writer(output_file)
        output_writer.writerows(u[:, :Feature])


if __name__ == '__main__':
    main()
