import caffe
import time
import csv

caffe_root = ''

# Parameters
TRAIN_DATA_SIZE = 60000
BATCH_SIZE = 64          # Batch Size
EPOCH_NUM = 20           # Training Epoch

test_interval = TRAIN_DATA_SIZE // BATCH_SIZE
niter = test_interval * EPOCH_NUM

# device (CPU or GPU)
caffe.set_mode_cpu()
# caffe.set_mode_gpu()
# caffe.set_device(0)

# net definition
solver = caffe.SGDSolver('prototxt/lenet_auto_solver.prototxt')


# storing evaluation
class History:
    def __init__(self):
        self.times = []
        self.loss = []
        self.acc = []

# Training
if __name__ == "__main__":
    history = History()
    sum_loss = 0
    epoch_start_time = None
    for i in range(niter):
        # one batch size
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
            epoch_acc = correct / 10000.0
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
