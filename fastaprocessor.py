
import csv
from Bio import SeqIO


def main():
    fastafile = "benchmark_set.fasta"
    fastarecords = generateFastaRecord(fastafile)
    writeCSV(fastarecords)

def generateFastaRecord(fastafile):
    fastarecords = []
    with open(fastafile, "rU") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            recordlength = int(len(record.seq) / 2)
            record.seq = record.seq[0:recordlength]
            if record.name.__contains__("NO_SP"):
                fastarecords.append([str(record.name.split("|")[0]),str(record.seq),"NO_SP"])
            elif record.name.__contains__("SP"):
                fastarecords.append([str(record.name.split("|")[0]),str(record.seq),"SP"])
        return fastarecords

def writeCSV(fastarecords):
    with open('processed_data_test.csv', mode='w') as processed_data:
        processed_writer = csv.writer(processed_data
                                      , delimiter=','
                                      , quotechar='"'
                                      , quoting=csv.QUOTE_MINIMAL)

        processed_writer.writerow(["ID", "Sequence", "Signal Peptide"])
        for record in fastarecords:
            processed_writer.writerow(record)

if __name__=="__main__":
	main()