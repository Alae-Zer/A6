def readIntoList(filename, linesToSkip=0):
    """
    Reads a file into a list of strings, one line per string.

    Caller must handle IO exceptions.

    Args:
        fileName: the path to the file to read
        linesToSkip: the number of lines to skip at the beginning, default = 0

    Returns:
        a list of strings, with line breaks removed, one line per list element
    """
    fin = open(filename, 'r')
    # print(linesToSkip)
    lines = []
    for i in range(linesToSkip):
        fin.readline()
    for line in fin:
        lines.append(line.strip())
    fin.close()
    return lines


def writeListToFile(lines, fileName):
    """
    Writes a list of strings to a file, one string per line.

    Note that a new file is created; so if the specified file
    already exists, it is over-written.

    Args:
        lines: a list of strings to be written to the file
        fileName: the name of the file to create
    """
    fout = open(fileName, 'w')
    for line in lines:
        fout.write(line + "\n")
    fout.close()
