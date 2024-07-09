
#include "SIMCore.h"

#include "Utils/MemUtilities.h"
#include "Utils/TimingUtilities.h"

SIMCore::SIMCore(const std::shared_ptr<SIMManager>& scene, const std::shared_ptr<SceneStepper>& scene_stepper)
	: m_scene(scene), m_scene_stepper(scene_stepper), m_current_step(0) {
	timing_buffer.resize(15);
	timing_buffer.assign(15, 0.0);

	memset(&m_info, 0, sizeof(Info));
}

SIMCore::~SIMCore() {}

const SIMCore::Info& SIMCore::getInfo() const { return m_info; }

const std::shared_ptr<SceneStepper>& SIMCore::getSceneStepper() const {
	return m_scene_stepper;
}

int SIMCore::getCurrentTime() const { return m_current_step; }

const std::shared_ptr<SIMManager>& SIMCore::getScene() const {
	return m_scene;
}

const std::vector<scalar>& SIMCore::getTimingStatistics() const {
	return timing_buffer;
}

/*
 * This is the main function where time stepping happens
 */
void SIMCore::stepSystem(const scalar& dt) {
	assert(m_scene != NULL);
	assert(m_scene_stepper != NULL);

	VectorXs oldpos = m_scene->getX();
	VectorXs oldvel = m_scene->getV();

	const scalar max_elasto_vel = m_scene->getMaxVelocity();

	const scalar dx = m_scene->getCellSize();
//   const scalar max_elasto_dt = std::min(dx / std::max(1e-63, max_elasto_vel) / 3.0, 1.0 / 30.0);  // 1/6 CFLs
	const scalar max_elasto_dt = std::min(dx / std::max(1e-8, max_elasto_vel), 1.0 / 30.0); 
	const scalar max_dt = std::min(max_elasto_dt, 1e9);

	const int num_substeps = std::max(1, (int)ceil(dt / max_dt));
	const scalar sub_dt = dt / (scalar)num_substeps;

	m_info.m_historical_max_vel = std::max(m_info.m_historical_max_vel, max_elasto_vel);

	std::cout << "[step system max vel: (" << max_elasto_vel << " <"
						<< "), # sub-step: (" << num_substeps << "), sub-dt: " << sub_dt
						<< "]" << std::endl;

	// Start the possible sub-steps
	for (int k = 0; k < num_substeps; ++k) {
		scalar cur_time = (scalar)m_current_step * dt + k * sub_dt;
		std::cout << "[(" << cur_time << " s) start substep: " << k << "/" << num_substeps << "]" << std::endl;

		scalar t0 = timingutils::seconds();
		scalar t1;

		// Update Viscous Parameter for Elastic Rods
		m_scene->updateStrandParamViscosity(sub_dt);

		// Setup Scripting for Kinematic Objects
		m_scene->stepScript(sub_dt, cur_time);
		m_scene->applyScript(sub_dt);

		// Create Grid around Particles
		m_scene->updateParticleBoundingBox();
		m_scene->rebucketizeParticles();
		m_scene->resampleNodes();
		t1 = timingutils::seconds();
		timing_buffer[1] += t1 - t0;  // build Grid
		t0 = t1;

		// Update Particle-Node Weight
		m_scene->computeWeights(sub_dt);

		// Update Solid Stress
		m_scene->computedEdFe();

		m_scene->updateManifoldOperators();

		// Update the Orientation Field
		// m_scene->updateOrientation();

		// Here's the precomputation of some forces lay
		m_scene->updateStartState();

		// Update the Distance Function for Kinematic Objects
		m_scene->updateSolidPhi();

		// Update the Weight on Grid (see [Batty et al. 2007] for details) for
		// Kinematic Objects
		m_scene->updateSolidWeights();

		// Save Current Velocity
		m_scene->saveParticleVelocity();

		t1 = timingutils::seconds();
		timing_buffer[2] += t1 - t0;  // Compute Weight, Solid Stress, and Distance Field (all above)
		t0 = t1;

		// Map the Liquid Particles and Elastic Vertices onto Grid
		m_scene->mapParticleNodesAPIC();

		t1 = timingutils::seconds();
		timing_buffer[3] += t1 - t0;  // APIC Mapping & Computing the Fields (all above)
		t0 = t1;

		// Explicitly Integrate the Elastic and Liquid Velocity
		m_scene_stepper->stepVelocity(*m_scene, sub_dt);
		t1 = timingutils::seconds();
		timing_buffer[4] += t1 - t0;  // Velocity Prediction
		t0 = t1;

		if (m_scene->getSIMInfo().solve_solid) {
			// Implicitly Integrate the Elastic Objects
			m_scene_stepper->stepImplicitElasto(*m_scene, sub_dt);
			t1 = timingutils::seconds();
			timing_buffer[6] += t1 - t0;  // Solve solid velocity
			t0 = t1;
		}

		// Update the Current Velocity with the Solved Ones
		m_scene_stepper->acceptVelocity(*m_scene);


		// Transfer Velocity Back to Particles and Elastic Vertices
		m_scene->mapNodeParticlesAPIC();
		t1 = timingutils::seconds();
		timing_buffer[9] += t1 - t0;  // APIC Map Particle Back
		t0 = t1;

		// Update the Multipliers applied on Geometric Stiffness
		// (refer to the supplemental material of [Fei et al. 2017] for details)
		m_scene->updateMultipliers(sub_dt);

		// Advection of Liquid Particles and Elastic Vertices
		m_scene_stepper->advectScene(*m_scene, sub_dt);

		// Kinematic Projection of the Elastic Vertices at the Boundary (as Fail-safe)
		m_scene->solidProjection(sub_dt);
		t1 = timingutils::seconds();
		timing_buffer[10] += t1 - t0;  // Particle Advection
		t0 = t1;

		// Update the Velocity Displacement
		m_scene->updateVelocityDifference();

		// Update the Acceleration of Liquid on Elastic Vertices
		m_scene->updateGaussAccel();

		// Update the Variables on Elements
		// We denote elements as 'Gauss' since they are computed at the Gaussian
		// Quadrature Point (1-Point).
		std::cout << "[update gauss system and plasticity]" << std::endl;
		m_scene->updateGaussSystem(sub_dt);
		m_scene->updatePlasticity(sub_dt);
		t1 = timingutils::seconds();
		timing_buffer[14] += t1 - t0;  // update Deformation Gradient
		t0 = t1;
	}

	// Summarize Memory Usage
	size_t cur_usage = memutils::getCurrentRSS();

	m_info.m_mem_usage_accu += (scalar)cur_usage;
	m_info.m_num_particles_accu += (scalar)m_scene->getNumParticles();
	m_info.m_num_elements_accu += (scalar)m_scene->getNumGausses();


	++m_current_step;
}