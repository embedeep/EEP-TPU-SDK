#!/bin/bash
basepath=$(cd `dirname $0`; pwd)
exepath=${basepath}"/../../"
cd ${exepath}
./eeptpu_compiler --prototxt ${basepath}/Lenet-5.prototxt --caffemodel ${basepath}/lenet_iter_10000.caffemodel --image ${basepath}/00001_2.pgm --norm "0.00390625" --output ${basepath}/
cd - > /dev/null

