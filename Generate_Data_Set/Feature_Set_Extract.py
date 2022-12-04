import csv


def main():
    Target_file = open("D:/cer_electricity/CER_Electricity_Data/Survey data - CSV format/Smart meters Residential pre-trial survey data.csv")
    target = []
    csv_reader = csv.reader(Target_file)
    _ = next(Target_file)
    for row in csv_reader:
        target.append(row[0])
    Target_file.close()

    target.append(0)
    k = 0

    with open("D:/cer_electricity/F+C_Set/Feature_Total.csv", 'w', newline="") as file:
        csv_writer = csv.writer(file)
        for i in range(1, 7):
            Deal_file = open("D:/cer_electricity/F+C_Set/Feature_File" + str(i) + ".csv")
            csv_reader2 = csv.reader(Deal_file)
            for row in csv_reader2:
                if int(float(row[0])) == int(float(target[k])):
                    csv_writer.writerow(row)
                    k += 1
            Deal_file.close()
            print("File" + str(i) + " Finished")
    print(k)


if __name__ == '__main__':
    main()
