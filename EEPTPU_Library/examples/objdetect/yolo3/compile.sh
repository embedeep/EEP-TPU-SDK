# change the eeptpu library path to your own
echo "Note: Copy to ARM platform and compile it. Should manual modify the library path in this script."
echo "Compiling..."
arm-linux-gnueabihf-g++ -o eeplib_test main.cpp yolo3_detection_output.cpp -I./ -I../../../libeeptpu_arm/include -L../../../libeeptpu_arm/lib/ -Wl,-rpath,./:../../../libeeptpu_arm/lib/ -leeptpu -lopencv_core -lopencv_highgui -lopencv_imgproc

