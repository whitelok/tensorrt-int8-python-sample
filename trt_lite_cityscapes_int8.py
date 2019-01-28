import glob
from random import shuffle
import numpy as np
from PIL import Image

import tensorrt as trt

import calibrator

MEAN = (71.60167789, 82.09696889, 72.30508881)
MODEL_DIR = './data/mnist/'
CITYSCAPES_DIR = './data/mnist/'
TEST_IMAGE = CITYSCAPES_DIR + '1.pgm.png'
CALIBRATION_DATASET_LOC = CITYSCAPES_DIR + '*.png'

CLASSES = 10
CHANNEL = 1
HEIGHT = 28
WIDTH = 28


def sub_mean_chw(data):
    return data


def color_map(output):
    output = output.reshape(CLASSES, HEIGHT, WIDTH)
    out_col = np.zeros(shape=(HEIGHT, WIDTH), dtype=(np.uint8, 3))
    for x in range(WIDTH):
        for y in range(HEIGHT):
            out_col[y, x] = labels.id2label[labels.trainId2label[np.argmax(output[:, y, x])].id].color
    return out_col


def create_calibration_dataset():
    calibration_files = glob.glob(CALIBRATION_DATASET_LOC)
    shuffle(calibration_files)
    return calibration_files


def main():
    print("Loading image files...")
    calibration_files = create_calibration_dataset()
    batchstream = calibrator.ImageBatchStream(
        5, calibration_files, sub_mean_chw)
    print("Map image data from float to int8...")
    int8_calibrator = calibrator.PythonEntropyCalibrator(["data"], batchstream)

    # create a TensorRT engine
    engine = trt.lite.Engine(framework="c1",
                             deployfile=MODEL_DIR + "mnist.prototxt",
                             modelfile=MODEL_DIR + "mnist.caffemodel",
                             max_batch_size=1,
                             max_workspace_size=(256 << 20),
                             input_nodes={"data": (CHANNEL, HEIGHT, WIDTH)},
                             output_nodes=["prob"],
                             preprocessors={"data": sub_mean_chw},
                             data_type=trt.infer.DataType.INT8,
                             # here is the int8 calibrator
                             calibrator=int8_calibrator,
                             logger_severity=trt.infer.LogSeverity.INFO)

    test_data = calibrator.ImageBatchStream.read_image_chw(TEST_IMAGE)

    print("Doing inference...")
    out = engine.infer(test_data.reshape((1,1,HEIGHT,WIDTH)))[0]

    print("reading image %s" % TEST_IMAGE)

    result = list(out.reshape(CLASSES,))
    print("The raw print out result is:%s" % str(result))

    print("The inference expect result is: %s" % result.index(max(result)))

if __name__ == '__main__':
    main()
