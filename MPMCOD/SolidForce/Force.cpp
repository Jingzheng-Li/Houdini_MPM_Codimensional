//
// This file is part of the libWetCloth open source project
//
// Copyright 2018 Yun (Raymond) Fei, Christopher Batty, Eitan Grinspun, and
// Changxi Zheng
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Force.h"

Force::~Force() {}

void Force::addAngularHessXToTotal(const VectorXs& x, const VectorXs& v,
                                   const VectorXs& m, 
                                   TripletXs& hessE,
                                   int hessE_index, const scalar& dt) {}

int Force::numAngularHessX() { return 0; }

void Force::postCompute(VectorXs& v, const scalar& dt) {}

bool Force::parallelized() const { return false; }
