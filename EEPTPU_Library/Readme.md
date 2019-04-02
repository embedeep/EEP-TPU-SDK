## EEP-TPU API Library
This is the library for EEP-TPU and FREE-TPU.

For Free-TPU users, you could use the pre-compiled binary files and free-tpu IP, please refer: https://github.com/embedeep/Free-TPU

For commercial users, you could use eeptpu_compiler to compile your own 'eeptpu.bin' file, and run it on commercial version EEP-TPU IP.

libeeptpu_arm: arm version. Such as Xilinx 7020/7035, EEP-TPU is in PL.

libeeptpu_x86: x86 version. Such as Xilinx VC707, EEP-TPU is in a PCIE card.

### User Guide
A simple user guide for you, please refer 'doc' folder.

### Examples
Please refer 'examples' folder.

### Notes
- Currently the library only supports to load and use one eeptpu.bin file in program. Using more than one bin file will be supported later.
- Built by cross-compiler tool: arm-linux-gnueabihf, version 6.3.1, compatible with raspberry pi file system.

