// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#pragma once
#ifndef OPENCV_OVERLAY_HPP
#define OPENCV_OVERLAY_HPP

#include "cvdef.h"

namespace cv
{
    CV_EXPORTS_W void overlay(InputOutputArray image, InputArray watermark, int x, int y);
} // namespace cv

#endif // OPENCV_OVX_HPP
