#include <stdio.h>
#include <vector>
#include <sys/time.h>
#include "eeptpu.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;

static int image_read_to_cv_mat(char* img_path, int dc, int dh, int dw, cv::Mat& cv_img_orig, cv::Mat& cv_img_resized);
static double get_current_time();

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    char obj_class_name[32];
};
static std::vector<Object> g_objects;
static char text[256];

static int post_process_obj_detect(cv::Mat& cvimg, ncnn::Mat& blob_out);
static cv::Mat draw_objects(const cv::Mat &bgr, const std::vector<Object> &objects);
static void clear_objects();

extern int yolo3_detection_output_forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs);

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

    // mobilenet yolo_v3_lite
    float mean_vals_mobilenet_yolov3_lite[3] = {1.0f, 1.0f, 1.0f};
    float norm_vals_mobilenet_yolov3_lite[3] = {1.0/255, 1.0/255, 1.0/255};
    ret = eeptpu_set_mean(mean_vals_mobilenet_yolov3_lite, norm_vals_mobilenet_yolov3_lite, 3, eepMeanMode_Norm2Mean);
    if (ret < 0) {
        printf("Set mean fail\n");
        return ret;
    }
    
    printf("EEPTPU init ok\n");
    
    return 0;
}

static int eeptpu_write_input(char* path_image, int in_c, int in_h, int in_w, cv::Mat& cvimg_orig)
{
    int ret;
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

    cvimg_resized.release();
    
    return 0;
}

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        printf("Usage: sudo %s eeptpu_bin_path image_path\n", argv[0]);
        return -1;
    }
    
    char *path_bin = argv[1];
    char *path_image = argv[2];
    int ret;
    
    printf("\n# Embedeep EEPTPU Library Example - Mobilenet YoloV3 #\n\n");

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
    
    cv::Mat cvimg_orig;
    ret = eeptpu_write_input(path_image, in_c, in_h, in_w, cvimg_orig);
    if (ret < 0)
    {
        printf("EEPTPU write input fail\n");
        return ret;
    }
    
    vector<ncnn::Mat> blobs_out;
    double st = get_current_time();
    ret = eeptpu_forward(blobs_out);
    //ret = eeptpu_forward(blobs_out, 1);   // async mode, should manual read output result(eeptpu_read_forward_result(blobs_out)). For example, use another thread to read result.
    if (ret < 0) {
        printf("EEPTPU forward fail\n");
        return ret;
    }
    
    // the last layer: yolo3_detection_output
    vector<ncnn::Mat> top_blobs;
    ret = yolo3_detection_output_forward(blobs_out, top_blobs);
    
    double et = get_current_time();
    printf("EEPTPU forward ok, cost time(hw+sw): %.3f ms\n", et-st);    
    unsigned int hwus = eeptpu_get_tpu_forward_time();
    printf("EEPTPU hw cost: %.3f ms\n", (float)hwus/1000);
    
    if (top_blobs.size() == 0)
    {
        printf("Nothing detected. ret=%d",ret);
    }
    else
    {
        ret = post_process_obj_detect(cvimg_orig, top_blobs[0]);
        if (ret < 0) return ret;
        
        for (unsigned int a = 0; a < top_blobs.size(); a++) top_blobs[a].release();
        top_blobs.clear();
        
        // draw object
        cv::Mat final_image = draw_objects(cvimg_orig, g_objects);
        
        // show final image.
        cv::imshow("EEPTPU Mobilenet YoloV3 Example", final_image);
        cv::waitKey(0);
        if (final_image.empty() == false) final_image.release();        
    }
    
    // all images done, close eeptpu.
    eeptpu_close();    
    for (unsigned int a = 0; a < blobs_out.size(); a++) blobs_out[a].release();
    blobs_out.clear();
    cvimg_orig.release();
    clear_objects();
    
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

static double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// ====== object detect process ======

static const char *class_names_default[] = {"background",
                                    "aeroplane", "bicycle", "bird", "boat",
                                    "bottle", "bus", "car", "cat", "chair",
                                    "cow", "diningtable", "dog", "horse",
                                    "motorbike", "person", "pottedplant",
                                    "sheep", "sofa", "train", "tvmonitor"};
char** class_names = (char **)class_names_default;

static void clear_objects()
{
    g_objects.clear();
}

static int post_process_obj_detect(cv::Mat& cvimg, ncnn::Mat& blob_out)
{
    clear_objects();
    
    if (blob_out.c * blob_out.h * blob_out.w == 0) 
    {
        printf("Nothing detected !\n");
        return 0;
    }
    
    class_names = (char**)class_names_default;
    
    for (int i = 0; i < blob_out.h; i++)
    {
        const float *values = blob_out.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * cvimg.cols;
        object.rect.y = values[3] * cvimg.rows;
        object.rect.width = values[4] * cvimg.cols - object.rect.x;
        object.rect.height = values[5] * cvimg.rows - object.rect.y;
        strcpy(object.obj_class_name, class_names[object.label]);
        
        if (object.rect.x < 0) { object.rect.width += object.rect.x; object.rect.x = 0; }
        if (object.rect.y < 0) { object.rect.height+= object.rect.y; object.rect.y = 0; }

        g_objects.push_back(object);
    }
    
    blob_out.release();

    printf("Detection result: \n");
    for (unsigned int i = 0; i < g_objects.size(); i++)
    {
        const Object &obj = g_objects[i];
        printf("    [%d] %d = %.5f at %.2f %.2f %.2f x %.2f   (%s)\n", i, obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height, class_names[obj.label]);
    }
        
    return 0;
}


static cv::Mat draw_objects(const cv::Mat &bgr, const std::vector<Object> &objects)
{    
    cv::Mat image = bgr.clone();

    double font_size = 0.6;
    int font_bold = 1;

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object &obj = objects[i];

        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);        
        int baseLine = 0;
        cv::rectangle(image, obj.rect, cv::Scalar(0, 255, 0), font_bold);        
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_size, font_bold, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (x + label_size.width > image.cols) x = image.cols - label_size.width;

        cv::Point point;

        point = cv::Point(x, y + label_size.height);
        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(128, 128, 0), CV_FILLED);
        cv::putText(image, text, point, cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar(0, 255, 0), font_bold);
    }

    return image;
}

