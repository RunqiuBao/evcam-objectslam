import lmdb
import numpy

class LmdbWriter(): #lmdb is multi read single write.
    def __init__(self, write_path, map_size=1099511627776):
        self.write_path = write_path
        self.map_size = map_size
        self.env = lmdb.open(write_path, map_size)
        self.txn = self.env.begin(write=True)

    def write(self, key, dataunit):
        self.txn.put(key=key, value=dataunit)

    def commitchange(self):
        # commit change before ram is full
        self.txn.commit()

    def endwriting(self):
        self.env.close()

    class OneResult():
        def __init__(self, key, img):
            self.key = key
            self.img = img.astype(numpy.int8)