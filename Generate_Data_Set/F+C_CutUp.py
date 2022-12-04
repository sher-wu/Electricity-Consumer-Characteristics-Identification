import csv


def main():
    with open("D:/cer_electricity/F+C_Set/F+C_Set.csv", 'r') as from_file:
        from_reader = csv.reader(from_file)

        for i in range(1, 7):
            with open("D:/cer_electricity/Train_Set/Retailer" + str(i) + ".csv", 'w', newline="") as to_file:
                to_writer = csv.writer(to_file)
                if i < 6:
                    total = 680
                else:
                    total = 832
                for j in range(total):
                    now = next(from_reader)
                    to_writer.writerow(now)


if __name__ == '__main__':
    main()
