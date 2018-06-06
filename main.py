import csv
from datetime import datetime
from matplotlib import pyplot

O_YEAR = 1
O_MONTH = 2
O_DAY = 3
O_HOUR = 4
O_DEW_POINT = 6
O_PM = 5
O_TEMP = 7
O_PRESS = 8
O_LWS = 10
O_LS = 11
O_LR = 12
O_WINDIR = 9

def parseDate(x):
    return datetime.strptime(str(x), '%Y/%m/%d %H')

def saveToFile(array):
    with open('new.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar=',', quoting=csv.QUOTE_MINIMAL)
        for r in newRows:
            spamwriter.writerow(r)

def plot(arr):
    temp = []
    for x in arr:
        temp.append(float(x[1]))

    temp2 = []
    for x in arr:
        temp2.append(x[0])

    pyplot.plot(temp2, temp, '-', linewidth=2.0)
    pyplot.gcf().autofmt_xdate()
    pyplot.ylabel('pollution')
    pyplot.show()

def getColumnValues(reader, index):
    column = []
    for row in reader:
        if row[index] == 'NA': continue
        column.append(row[index])

    return column

def writeWindDirection(row, wind):
    if wind == 'NE':
        row.append(1)
    else:
        row.append(0)

    if wind == 'NW':
        row.append(1)
    else:
        row.append(0)

    if wind == 'SE':
        row.append(1)
    else:
        row.append(0)

def maxValueInColumn(column):
    max = 0
    for value in column:
        if float(value) > max:
            max = float(value)
    return max

def minValueInColumn(column):
    min = 10000
    for value in column:
        if float(value) < min:
            min = float(value)
    return min


newRows = []

with open('air.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    next(reader, None)  # skip the headers

    maxDew = maxValueInColumn(getColumnValues(reader, O_DEW_POINT));
    csvFile.seek(0)
    next(reader, None)  # skip the headers

    minDew = minValueInColumn(getColumnValues(reader, O_DEW_POINT));
    csvFile.seek(0)
    next(reader, None)  # skip the headers


    maxPM = maxValueInColumn(getColumnValues(reader, O_PM));
    csvFile.seek(0)
    next(reader, None)  # skip the headers

    minPM = minValueInColumn(getColumnValues(reader, O_PM));
    csvFile.seek(0)
    next(reader, None)  # skip the headers


    maxTemp = maxValueInColumn(getColumnValues(reader, O_TEMP));
    csvFile.seek(0)
    next(reader, None)  # skip the headers

    minTemp = minValueInColumn(getColumnValues(reader, O_TEMP));
    csvFile.seek(0)
    next(reader, None)  # skip the headers


    maxPress = maxValueInColumn(getColumnValues(reader, O_PRESS));
    csvFile.seek(0)
    next(reader, None)  # skip the headers

    minPress = minValueInColumn(getColumnValues(reader, O_PRESS));
    csvFile.seek(0)
    next(reader, None)  # skip the headers


    maxLWS = maxValueInColumn(getColumnValues(reader, O_LWS));
    csvFile.seek(0)
    next(reader, None)  # skip the headers

    minLWS = minValueInColumn(getColumnValues(reader, O_LWS));
    csvFile.seek(0)
    next(reader, None)  # skip the headers


    maxLS = maxValueInColumn(getColumnValues(reader, O_LS));
    csvFile.seek(0)
    next(reader, None)  # skip the headers

    minLS = minValueInColumn(getColumnValues(reader, O_LS));
    csvFile.seek(0)
    next(reader, None)  # skip the headers


    maxLR = maxValueInColumn(getColumnValues(reader, O_LR));
    csvFile.seek(0)
    next(reader, None)  # skip the headers

    minLR = minValueInColumn(getColumnValues(reader, O_LR));
    csvFile.seek(0)
    next(reader, None)  # skip the headers

    maxPM = maxValueInColumn(getColumnValues(reader, O_PM));
    csvFile.seek(0)
    next(reader, None)  # skip the headers

    minPM = minValueInColumn(getColumnValues(reader, O_PM));
    csvFile.seek(0)
    next(reader, None)  # skip the headers

    newRows.append(['DATE', 'DEW', 'PM', 'TEMP', 'PRESS', 'LWS', 'LS', 'LR', 'NE', 'NW', 'SE'])
    for row in reader:
        tempRow = []
        tempRow.append(parseDate(row[O_YEAR] + "/" + row[O_MONTH] + "/" + row[O_DAY] + " " + row[O_HOUR]))
        if row[O_PM] == 'NA': continue

        tempRow.append((float(row[O_DEW_POINT]) - minDew) / (maxDew - minDew))
        tempRow.append((float(row[O_PM]) - minPM) / (maxPM - minPM))
        tempRow.append((float(row[O_TEMP]) - minTemp) / (maxTemp - minTemp))
        tempRow.append((float(row[O_PRESS]) - minPress) / (maxPress - minPress))
        tempRow.append((float(row[O_LWS]) - minLWS) / (maxLWS - minLWS))
        tempRow.append((float(row[O_LS]) - minLS) / (maxLS - minLS))
        tempRow.append((float(row[O_LR]) - minLR) / (maxLR - minLR))
        writeWindDirection(tempRow, row[O_WINDIR])
        newRows.append(tempRow)

    saveToFile(newRows)
    #plot(newRows)

csvFile.close()