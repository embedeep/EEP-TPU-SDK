#include <stdio.h>
#include <vector>
#include <sys/time.h>
#include "eeptpu.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;

static int image_read_to_cv_mat(char* img_path, int dc, int dh, int dw, cv::Mat& cv_img_orig, cv::Mat& cv_img_resized);
static int get_topk(ncnn::Mat& blob, int topk, std::vector< std::pair<float, int> >& top_list);
static double get_current_time();


static int eeptpu_init(int interface_type, char* path_bin)
{
    int ret; 
    
    ret = eeptpu_set_interface(interface_type);
    if (ret < 0) return ret;
    
    printf("EEPTPU library  version: %s\n", eeptpu_get_lib_version());
    printf("EEPTPU hardware version: %s\n", eeptpu_get_tpu_version());
    printf("EEPTPU hardware info   : %s\n", eeptpu_get_tpu_info());
    
    ret = eeptpu_load_bin(path_bin);
    if (ret < 0) {
        printf("Load bin fail, ret=%d\n", ret);
        return ret;
    }
    printf("Load bin ok\n");

    #if 1
    // lenet5
    float mean[] = {0.f};
    float norm[] = {1.0/256.0f};
    ret = eeptpu_set_mean(mean, norm, 1, eepMeanMode_Mean2Norm);
    #endif
    
    #if 0
    // mobilenet v1/v2
    float mean[] = {103.94f, 116.78f, 123.68f};
    float norm[] = {0.017f,0.017f,0.017f};
    ret = eeptpu_set_mean(mean, norm, 3, eepMeanMode_Mean2Norm);
    #endif
    
    if (ret < 0) {
        printf("Set mean fail\n");
        return ret;
    }
    
    printf("EEPTPU init ok\n");
    
    return 0;
}

static int eeptpu_write_input(char* path_image, int in_c, int in_h, int in_w)
{
    int ret;
    cv::Mat cvimg_orig;
    cv::Mat cvimg_resized;
    ret = image_read_to_cv_mat(path_image, in_c, in_h, in_w, cvimg_orig, cvimg_resized);
    if (ret < 0) {
        printf("Read image fail\n");
        return ret;
    }
    printf("Read image ok\n");
    
    ret = eeptpu_wait_writable();
    if (ret < 0) 
    {
        printf("Wait writable timeout. ret=%d\n", ret);
        return ret;
    }
    ret = eeptpu_set_input(cvimg_resized.data, in_c, in_h, in_w);
    if (ret < 0) {
        printf("Set EEPTPU input fail\n");
        return ret;
    }
    printf("Set EEPTPU input ok\n");
    
    cvimg_orig.release();
    cvimg_resized.release();
    
    return 0;
}

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        printf("Usage: %s eeptpu_bin_path image_path\n", argv[0]);
        return -1;
    }
    
    char *path_bin = argv[1];
    char *path_image = argv[2];
    int ret;
    
    printf("\n# Embedeep EEPTPU Library Example #\n\n");

    ret = eeptpu_init(eepInterfaceType_DDR, path_bin);
    if (ret < 0)
    {
        printf("EEPTPU init fail.\n");
        return ret;
    }
    
    int in_c, in_h, in_w;
    ret = eeptpu_get_input_shape(&in_c, &in_h, &in_w);
    if (ret < 0) 
    {
        printf("Can not get input shape, please check if you had loaded the tpu binary file correctly.\n");
        return ret;
    }
    printf("Network input shape: [c,h,w] = [%d,%d,%d]\n", in_c, in_h, in_w);
    
    // ====== eeptpu init ok, only init one time in program.
    
    // ====== loop write input and forward if you have many images.
    
    ret = eeptpu_write_input(path_image, in_c, in_h, in_w);
    if (ret < 0)
    {
        printf("EEPTPU write input fail\n");
        return ret;
    }
    
    ncnn::Mat blob_out;
    double st = get_current_time();
    ret = eeptpu_forward(blob_out);
    if (ret < 0) {
        printf("EEPTPU forward fail\n");
        return ret;
    }
    double et = get_current_time();
    printf("EEPTPU forward ok, cost time(hw+sw): %.3f ms\n", et-st);    
    unsigned int hwus = eeptpu_get_tpu_forward_time();
    printf("EEPTPU hw cost: %.3f ms\n", (float)hwus/1000);
    
    std::vector< std::pair<float, int> > top_list;
    ret = get_topk(blob_out, 5, top_list);
    if (ret < 0) return ret;
    
    printf("Result (top 5): \n");
    for (int i = 0; i < 5; i++)
    {
        printf("  [%3d] %.3f\n", top_list[i].second, top_list[i].first);
    }
    
    
    // all images done, close eeptpu.
    eeptpu_close();    
    top_list.clear();
    blob_out.release();
    
    printf("\nAll done\n");
    
    return 0;
}



static int image_read_to_cv_mat(char* img_path, int dc, int dh, int dw, cv::Mat& cv_img_orig, cv::Mat& cv_img_resized)
{
    if (dc == 1)
    {
        cv_img_orig = cv::imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
    }
    else if (dc == 3)
    {
        cv_img_orig = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
    }
    if (cv_img_orig.empty())
    {
        printf("opencv read file %s failed\n", img_path);
        return -1;
    }
    
    cv_img_resized = cv_img_orig;
    if ( (cv_img_resized.rows != dh) && (cv_img_resized.cols != dw) )
    {
        cv::resize(cv_img_resized, cv_img_resized, cv::Size(dw, dh)); 
    }
    
    return 0;
}


static int get_topk(const std::vector<float>& cls_scores, unsigned int topk, std::vector< std::pair<float, int> >& top_list)
{
    if (cls_scores.size() < topk) topk = cls_scores.size();

    int size = cls_scores.size();
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i=0; i<size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater< std::pair<float, int> >());
    
    top_list.resize(topk);
    for (unsigned int i=0; i<topk; i++)
    {
        top_list[i].first = vec[i].first;
        top_list[i].second = vec[i].second;
    }

    return 0;
}

static int get_topk(ncnn::Mat& blob, int topk, std::vector< std::pair<float, int> >& top_list)
{
    std::vector<float> cls_scores;
    cls_scores.resize(blob.w*blob.h*blob.c);

    int c = 0;
    for (int i = 0; i < blob.c; i++)
    {
        float* ch = (float*)blob.channel(i).data;
        for (int k = 0; k < blob.h * blob.w; k++)
        {
            cls_scores[c++] = *ch++;
        }
    }

    return get_topk(cls_scores, topk, top_list);
}

static double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}


