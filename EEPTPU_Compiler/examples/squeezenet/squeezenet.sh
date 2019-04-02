#!/bin/bash
basepath=$(cd `dirname $0`; pwd)
exepath=${basepath}"/../../"
cd ${exepath}
./eeptpu_compiler --prototxt ${basepath}/squeezenet_v1.1.prototxt --caffemodel ${basepath}/squeezenet_v1.1.caffemodel --image ${basepath}/dog-Husky_248.jpg --mean "104.0, 117.0, 123.0" --norm "1.0,1.0,1.0" --mean_mode 0 --output ${basepath}/
cd - > /dev/null

