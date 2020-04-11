// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencl_kernels_core.hpp"

#define CHANNELS_RGB 3
#define CHANNELS_ALPHA 4
#define ALPHA_CHANNEL_INDEX 3

#define TYPE_24 CV_MAKE_TYPE(24, CHANNELS_RGB)
#define TYPE_32 CV_MAKE_TYPE(32, CHANNELS_ALPHA)

namespace cv
{
    namespace
    {
        struct Overlay : public ParallelLoopBody
        {
            Overlay(Mat _image, Mat _watermark, const Rect& _rect, int x, int y)
                : image(_image)
                , watermark(_watermark)
                , rect(_rect)
                , offset(Point(x, y))
            { }

            virtual void operator() (const Range& r) const
            {
                const Mat& result = image; // alter the original image
                const int endX = rect.x + rect.width;

                for (int y = r.start; y < r.end; y++)
                {
                    for (int x = rect.x; x < endX; x++)
                    {
                        size_t wm_index = (y - offset.y) * watermark.step + (x - offset.x) * watermark.channels();

                        size_t image_index = y * image.step + x * image.channels();

                        uchar alpha = watermark.data[wm_index + ALPHA_CHANNEL_INDEX];

                        double opacity = (double) alpha / 255.0;

                        for (int c = 0; c < image.channels(); ++c)
                        {
                            uchar foreground_px = image.data[image_index + c];

                            uchar watermark_px = watermark.data[wm_index + c];

                            double result_px = foreground_px * (1.0 - opacity) + watermark_px * opacity;

                            result.data[image_index + c] = saturate_cast<uchar>(result_px);
                        }
                    }
                }
            }

        private:
            Mat image;
            Mat watermark;
            Rect rect;
            Point offset; // watermark offset
        };
    }

    static void overlay_op(Mat image, Mat watermark, int x, int y)
    {
        if (!(
            image.type() == TYPE_24 || image.type() == TYPE_32
            ))
            throw
            std::runtime_error("Wrong type of background (must be 24 or 32-bit depth)");

        if (
            watermark.type() != TYPE_32
            )
            throw
            std::runtime_error("Wrong type of watermark (must be 32-bit depth)");

        // watermark rect + offset
        Rect wm(0, 0, watermark.cols, watermark.rows);
        wm.x += x;
        wm.y += y;

        // get the result rect
        Rect rc(0, 0, image.cols, image.rows);
        rc &= wm;

        if (rc.area() == 0)
            throw
            std::runtime_error("The final rect is empty");

        parallel_for_(Range(rc.y, rc.y + rc.height), Overlay(image, watermark, rc, x, y));
    }

    void overlay(InputOutputArray image, InputArray watermark, int x, int y)
    {
        overlay_op(image.getMatRef(), watermark.getMat(), x, y);
    }
}
