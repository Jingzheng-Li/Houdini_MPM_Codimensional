
#include "SimpleGravityForce.h"

#include "Utils/ThreadUtils.h"
#include "Utils/Sorter.h"
#include "Solver/SIMManager.h"

SimpleGravityForce::SimpleGravityForce(const Vector3s& gravity)
		: Force(), m_gravity(gravity) {
	assert((m_gravity.array() == m_gravity.array()).all());
	assert((m_gravity.array() != std::numeric_limits<scalar>::infinity()).all());
}

SimpleGravityForce::~SimpleGravityForce() {}

void SimpleGravityForce::addEnergyToTotal(const VectorXs& x, const VectorXs& v,
										const VectorXs& m,
										scalar& E) {
	assert(x.size() == v.size());
	assert(x.size() == m.size());
	assert(x.size() % 4 == 0);

	// Assume 0 potential is at origin
	for (int i = 0; i < x.size() / 4; ++i)
		E -= m(4 * i) * m_gravity.dot(x.segment<3>(4 * i));
}

void SimpleGravityForce::addGradEToTotal(const VectorXs& x, const VectorXs& v,
										const VectorXs& m, 
										VectorXs& gradE) {
	const int num_elasto = gradE.size() / 4;
	threadutils::for_each(0, num_elasto, [&](int i) {
		gradE.segment<3>(4 * i) -= m(4 * i) * m_gravity;
	});
}

void SimpleGravityForce::addHessXToTotal(const VectorXs& x, const VectorXs& v,
										const VectorXs& m, 
										TripletXs& hessE,
										int hessE_index, const scalar& dt) {
	assert(x.size() == v.size());
	assert(x.size() == m.size());

	assert(x.size() % 4 == 0);
	// Nothing to do.
}

void SimpleGravityForce::updateMultipliers(
		const VectorXs& x, const VectorXs& vplus, const VectorXs& m, const scalar& dt) {}

int SimpleGravityForce::numHessX() { return 0; }

void SimpleGravityForce::preCompute() {}

void SimpleGravityForce::updateStartState() {}

Force* SimpleGravityForce::createNewCopy() {
	return new SimpleGravityForce(*this);
}

int SimpleGravityForce::flag() const { return 3; }
