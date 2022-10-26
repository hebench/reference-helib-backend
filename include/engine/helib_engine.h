
// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hebench/api_bridge/cpp/hebench.hpp"
#include "helib/helib.h"
#include "helib_types.h"
#include <memory>
#include <string>
#include <vector>

class HELIBEngine : public hebench::cpp::BaseEngine
{
public:
    HEBERROR_DECLARE_CLASS_NAME(HELIBEngine)
    static HELIBEngine *create();
    static void destroy(HELIBEngine *p);

    ~HELIBEngine() override;

protected:
    HELIBEngine();

    void init() override;
};
