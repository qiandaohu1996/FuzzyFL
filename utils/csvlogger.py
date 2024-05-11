import csv

class CSVLogger:
    def __init__(self, filename=None, mode='w', **kwargs):
        self.filename = filename
        self.mode = mode
        self.csv_writer = None
        self.file = None
        self.kwargs = kwargs

    def __enter__(self):
        self.file = open(self.filename, self.mode, **self.kwargs)
        self.csv_writer = csv.writer(self.file)
        return self.csv_writer

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
            
    def openfile(self, filename, mode='w', **kwargs):
        self.filename = filename
        self.mode = mode
        self.kwargs = kwargs
        self.file = open(self.filename, self.mode, **self.kwargs)
        self.csv_writer = csv.writer(self.file)
        return self.csv_writer
    
    def writerow(self, row):
        if self.csv_writer:
            self.csv_writer.writerow(row)
            
    def close(self):
        if self.file:
            self.file.close()