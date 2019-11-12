import caffe
import time
import csv

caffe_root = ''

caffe.set_mode_cpu()
solver = caffe.SGDSolver('prototxt/lenet_auto_solver.prototxt')

TRAIN_DATA_SIZE = 60000
BATCH_SIZE = 64
EPOCH = 2

test_interval = TRAIN_DATA_SIZE // BATCH_SIZE
niter = test_interval * EPOCH


class History:
    def __init__(self):
        self.times = []
        self.loss = []
        self.acc = []


history = History()
sum_loss = 0
epoch_start_time = None
for i in range(niter):
    if i % test_interval == 0:
        epoch_start_time = time.time()

    solver.step(1)

    sum_loss += solver.net.blobs['loss'].data
    if i % test_interval == 0:
        print "epoch", i // test_interval, 'testing...'
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        epoch_acc = correct / 10000
        epoch_loss = sum_loss / test_interval
        history.times.append(time.time()-epoch_start_time)
        history.loss.append(epoch_loss)
        history.acc.append(epoch_acc)


f = open("record.csv", "w")
f_writer = csv.writer(f)
f_writer.writerow(['time', 'loss', 'acc'])
for i in range(len(history.times)):
    row = [history.times[i], history.loss[i], history.acc[i]]
    f_writer.writerow(row)
f.close()
