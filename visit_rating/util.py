import gzip

def readGz(f):
    for l in gzip.open(f):
        yield eval(l)