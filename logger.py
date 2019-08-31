import logging
import csv
import io
import os

class CsvFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.output = io.StringIO()
        self.writer = csv.writer(self.output, quoting=csv.QUOTE_ALL)

    def format(self, record):
        self.writer.writerow([record.levelname, record.msg])
        data = self.output.getvalue()
        self.output.truncate(0)
        self.output.seek(0)
        return data.strip()

class Logger():
    def __init__(self, log_path):
        if not os.path.isdir(os.path.dirname(log_path)):
            os.mkdir(os.path.dirname(log_path))

        logging.basicConfig(level=logging.INFO, filename=log_path)
        self.logger = logging.getLogger(__name__)
        logging.root.handlers[0].setFormatter(CsvFormatter())

    def log(self, str):
        print(str)
        self.logger.info(str)
