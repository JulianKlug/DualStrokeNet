def csv_callback(file_descriptor):
    headers_printed = False
    def inner(metrics):
        nonlocal headers_printed
        if not headers_printed:
            file_descriptor.write(','.join(metrics.keys()))
            file_descriptor.write('\n')
            headers_printed = True
        file_descriptor.write(','.join(str(x) for x in metrics.values()))
        file_descriptor.write('\n')
        file_descriptor.flush()
    return inner
