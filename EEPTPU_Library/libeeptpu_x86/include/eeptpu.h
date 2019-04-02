#ifndef _EEPTPU_H
#define _EEPTPU_H

#include <vector>
#include "nmat.h"

enum {
    eepInterfaceType_DDR   = 1,
    eepInterfaceType_PCIE,
    eepInterfaceType_USB,
    eepInterfaceType_EOF,  /* dummy data. */
};

enum {
    eepMeanMode_Mean2Norm = 0,
    eepMeanMode_Norm2Mean,
};

int eeptpu_set_interface(int interface_type);
int eeptpu_set_interface_info_pcie(const char* dev_reg, const char* dev_h2c, const char* dev_c2h);
int eeptpu_set_interface_addr_pcie(unsigned int mem_base_addr, unsigned int reg_base_addr);

int eeptpu_load_bin(const char* path_bin);

int eeptpu_get_input_shape(int* c, int* h, int* w);

int eeptpu_set_mean(float* mean, float* norm, int channel, int mode);

int eeptpu_set_input(unsigned char* cvdata_resized, int channel, int height, int width);

int eeptpu_forward(ncnn::Mat& blob_out, int async = 0, unsigned int timeout_ms = 2000);
int eeptpu_forward(std::vector< ncnn::Mat > & blobs_out, int async = 0, unsigned int timeout_ms = 2000);

int eeptpu_wait_writable(unsigned int timeout_ms = 2000);
int eeptpu_wait_result_fetched(int timeout_ms);
int eeptpu_read_forward_result(ncnn::Mat& blob_out);
int eeptpu_read_forward_result(std::vector< ncnn::Mat > & blobs_out);
unsigned int eeptpu_get_tpu_forward_time();

void eeptpu_close();
void eeptpu_terminate();

char* eeptpu_get_lib_version();
char* eeptpu_get_tpu_version();
char* eeptpu_get_tpu_info();

// Error Code
#define succ                        (0)
#define eeperr_Fail                 (-1)
#define eeperr_Param                (-2)
#define eeperr_NotSupport           (-3)
#define eeperr_FileOpen             (-4)
#define eeperr_FileRead             (-5)
#define eeperr_FileWrite            (-6)
#define eeperr_Malloc               (-7)
#define eeperr_Timeout              (-8)

#define eeperr_LoadBin              (-11)
#define eeperr_TpuInit              (-12)
#define eeperr_ImgType              (-13)
#define eeperr_PixelType            (-14)
#define eeperr_Mean                 (-15)

#define eeperr_BlobSrc              (-16)
#define eeperr_BlobFmt              (-17)
#define eeperr_BlobSize             (-18)
#define eeperr_BlobData             (-19)
#define eeperr_BlobOutput           (-20)

#define eeperr_MemAddr              (-21)
#define eeperr_MemWr                (-22)
#define eeperr_MemRd                (-23)
#define eeperr_MemMap               (-24)

#define eeperr_DevOpen              (-25)
#define eeperr_DevNotInit           (-26)
#define eeperr_InterfaceType        (-27)
#define eeperr_InterfaceInit        (-28)
#define eeperr_InterfaceOperate     (-29)

#define eeperr_EEPBinTooOld         (-40)
#define eeperr_EEPLibTooOld         (-41)
#define eeperr_TpuVerTooOld         (-42)

#endif
