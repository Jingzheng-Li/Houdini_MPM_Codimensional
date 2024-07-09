//
// This file is part of the libWetCloth open source project
//
// Copyright 2018 Yun (Raymond) Fei, Christopher Batty, Eitan Grinspun, and
// Changxi Zheng
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SIMPLE_GRAVITY_FORCE_H
#define SIMPLE_GRAVITY_FORCE_H

#include <Eigen/Core>
#include <iostream>

#include "Force.h"

class SimpleGravityForce : public Force {
 public:
	SimpleGravityForce(const Vector3s& gravity);

	virtual ~SimpleGravityForce();

	virtual void addEnergyToTotal(const VectorXs& x, const VectorXs& v,
							const VectorXs& m, 
							scalar& E);

	virtual void addGradEToTotal(const VectorXs& x, const VectorXs& v,
							const VectorXs& m, 
							VectorXs& gradE);

	virtual void addHessXToTotal(const VectorXs& x, const VectorXs& v,
							const VectorXs& m, 
							TripletXs& hessE,
							int hessE_index, const scalar& dt);

	virtual void updateMultipliers(const VectorXs& x, const VectorXs& vplus,
							const VectorXs& m, 
							const scalar& dt);

	virtual void preCompute();

	virtual void updateStartState();

	virtual Force* createNewCopy();

	virtual int numHessX();

	virtual int flag() const;

 private:
	Vector3s m_gravity;
};

#endif
