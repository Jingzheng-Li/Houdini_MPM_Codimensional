

#include "Solver/LinearizedImplicitEuler.h"

#include <unordered_map>

#include "Utils/MathUtilities.h"
#include "Utils/ThreadUtils.h"
#include "Utils/Sorter.h"

#include <boost/sort/sort.hpp>

LinearizedImplicitEuler::LinearizedImplicitEuler(
	const scalar& criterion, const scalar& pressure_criterion,
	const scalar& quasi_static_criterion, const scalar& viscous_criterion,
	int maxiters, int manifold_substeps, int viscosity_substeps,
	int surf_tension_substeps)
	: SceneStepper(),
		m_pcg_criterion(criterion),
		m_maxiters(maxiters) {}

LinearizedImplicitEuler::~LinearizedImplicitEuler() {}

void LinearizedImplicitEuler::performInvLocalSolve(
		const SIMManager& scene, const std::vector<VectorXs>& node_rhs_x,
		const std::vector<VectorXs>& node_rhs_y,
		const std::vector<VectorXs>& node_rhs_z,
		const std::vector<VectorXs>& node_inv_mass_x,
		const std::vector<VectorXs>& node_inv_mass_y,
		const std::vector<VectorXs>& node_inv_mass_z,
		std::vector<VectorXs>& out_node_vec_x,
		std::vector<VectorXs>& out_node_vec_y,
		std::vector<VectorXs>& out_node_vec_z) {
	const Sorter& buckets = scene.getParticleBuckets();

	buckets.for_each_bucket([&](int bucket_idx) {
		if (!scene.isBucketActivated(bucket_idx)) return;

		const int num_nodes = scene.getNumNodes(bucket_idx);

		const VectorXs& bucket_node_masses_x = node_inv_mass_x[bucket_idx];
		const VectorXs& bucket_node_masses_y = node_inv_mass_y[bucket_idx];
		const VectorXs& bucket_node_masses_z = node_inv_mass_z[bucket_idx];

		const VectorXs& bucket_node_rhs_x = node_rhs_x[bucket_idx];
		const VectorXs& bucket_node_rhs_y = node_rhs_y[bucket_idx];
		const VectorXs& bucket_node_rhs_z = node_rhs_z[bucket_idx];

		VectorXs& bucket_out_node_vec_x = out_node_vec_x[bucket_idx];
		VectorXs& bucket_out_node_vec_y = out_node_vec_y[bucket_idx];
		VectorXs& bucket_out_node_vec_z = out_node_vec_z[bucket_idx];

		for (int i = 0; i < num_nodes; ++i) {
			bucket_out_node_vec_x[i] = bucket_node_rhs_x[i] * bucket_node_masses_x[i];
			bucket_out_node_vec_y[i] = bucket_node_rhs_y[i] * bucket_node_masses_y[i];
			bucket_out_node_vec_z[i] = bucket_node_rhs_z[i] * bucket_node_masses_z[i];
		}

		assert(!std::isnan(bucket_out_node_vec_x.sum()));
		assert(!std::isnan(bucket_out_node_vec_y.sum()));
		assert(!std::isnan(bucket_out_node_vec_z.sum()));
	});
}

void LinearizedImplicitEuler::performLocalSolve(const SIMManager& scene,
												const VectorXs& rhs,
												const VectorXs& m,
												VectorXs& out) {
	const int num_elasto = scene.getNumSoftElastoParticles();

	threadutils::for_each(0, num_elasto * 4, [&](int pidx) {
		out(pidx) = rhs(pidx) / std::max(1e-20, m(pidx));
	});
}

void LinearizedImplicitEuler::performLocalSolveTwist(const SIMManager& scene,
													const VectorXs& rhs,
													const VectorXs& m,
													VectorXs& out) {
	const int num_elasto = scene.getNumSoftElastoParticles();

	threadutils::for_each(0, num_elasto, [&](int pidx) {
		out(pidx) = rhs(pidx) / std::max(1e-20, m(pidx * 4 + 3));
	});
}

void LinearizedImplicitEuler::performLocalSolve(
		const SIMManager& scene, const std::vector<VectorXs>& node_rhs_x,
		const std::vector<VectorXs>& node_rhs_y,
		const std::vector<VectorXs>& node_rhs_z,
		const std::vector<VectorXs>& node_masses_x,
		const std::vector<VectorXs>& node_masses_y,
		const std::vector<VectorXs>& node_masses_z,
		std::vector<VectorXs>& out_node_vec_x,
		std::vector<VectorXs>& out_node_vec_y,
		std::vector<VectorXs>& out_node_vec_z) {
	const Sorter& buckets = scene.getParticleBuckets();

	buckets.for_each_bucket([&](int bucket_idx) {
		if (!scene.isBucketActivated(bucket_idx)) return;

		const int num_nodes = scene.getNumNodes(bucket_idx);

		const VectorXs& bucket_node_masses_x = node_masses_x[bucket_idx];
		const VectorXs& bucket_node_masses_y = node_masses_y[bucket_idx];
		const VectorXs& bucket_node_masses_z = node_masses_z[bucket_idx];

		const VectorXs& bucket_node_rhs_x = node_rhs_x[bucket_idx];
		const VectorXs& bucket_node_rhs_y = node_rhs_y[bucket_idx];
		const VectorXs& bucket_node_rhs_z = node_rhs_z[bucket_idx];

		VectorXs& bucket_out_node_vec_x = out_node_vec_x[bucket_idx];
		VectorXs& bucket_out_node_vec_y = out_node_vec_y[bucket_idx];
		VectorXs& bucket_out_node_vec_z = out_node_vec_z[bucket_idx];

		for (int i = 0; i < num_nodes; ++i) {
			if (bucket_node_masses_x[i] > 1e-20)
				bucket_out_node_vec_x[i] =
						bucket_node_rhs_x[i] / bucket_node_masses_x[i];
			else
				bucket_out_node_vec_x[i] = bucket_node_rhs_x[i];
			if (bucket_node_masses_y[i] > 1e-20)
				bucket_out_node_vec_y[i] =
						bucket_node_rhs_y[i] / bucket_node_masses_y[i];
			else
				bucket_out_node_vec_y[i] = bucket_node_rhs_y[i];
			if (bucket_node_masses_z[i] > 1e-20)
				bucket_out_node_vec_z[i] =
						bucket_node_rhs_z[i] / bucket_node_masses_z[i];
			else
				bucket_out_node_vec_z[i] = bucket_node_rhs_z[i];
		}
	});
}

bool LinearizedImplicitEuler::stepVelocity(SIMManager& scene, scalar dt) {
	// build node particle pairs
	scene.precompute();

	int ndof_elasto = scene.getNumSoftElastoParticles() * 4;
	const Sorter& buckets = scene.getParticleBuckets();

	allocateNodeVectors(scene, m_node_rhs_x, m_node_rhs_y, m_node_rhs_z);

	// allocateNodeVectors(scene, m_node_hdvm_x, m_node_hdvm_y, m_node_hdvm_z);
	// allocateNodeVectors(scene, m_node_inv_mfhdvm_x, m_node_inv_mfhdvm_y, m_node_inv_mfhdvm_z);
	allocateNodeVectors(scene, m_node_mfhdvm_hdvm_x, m_node_mfhdvm_hdvm_y, m_node_mfhdvm_hdvm_z);
	allocateNodeVectors(scene, m_node_mshdvm_hdvm_x, m_node_mshdvm_hdvm_y, m_node_mshdvm_hdvm_z);
	allocateNodeVectors(scene, m_node_inv_C_x, m_node_inv_C_y, m_node_inv_C_z);
	allocateNodeVectors(scene, m_node_inv_Cs_x, m_node_inv_Cs_y, m_node_inv_Cs_z);
	allocateNodeVectors(scene, m_node_Cs_x, m_node_Cs_y, m_node_Cs_z);

	allocateNodeVectors(scene, m_node_damped_x, m_node_damped_y, m_node_damped_z);
	constructNodeForce(scene, dt, m_node_rhs_x, m_node_rhs_y, m_node_rhs_z);
	// constructHDV(scene, dt);
	constructInvMDV(scene);

	allocateNodeVectors(scene, m_node_v_plus_x, m_node_v_plus_y, m_node_v_plus_z);

	// construct u_s^*
	if (ndof_elasto > 0 && scene.getLiquidInfo().solve_solid) {
		constructMsDVs(scene);

		const std::vector<VectorXs>& node_mass_x = scene.getNodeMassX();
		const std::vector<VectorXs>& node_mass_y = scene.getNodeMassY();
		const std::vector<VectorXs>& node_mass_z = scene.getNodeMassZ();

		performLocalSolve(scene, m_node_rhs_x, m_node_rhs_y, m_node_rhs_z,
						node_mass_x, node_mass_y, node_mass_z, m_node_v_plus_x,
						m_node_v_plus_y, m_node_v_plus_z);

		m_angular_v_plus_buffer.resize(scene.getNumSoftElastoParticles());
		performLocalSolveTwist(scene, m_angular_moment_buffer, scene.getM(), m_angular_v_plus_buffer);
	}

	// record as u_s^* and u_f^*
	acceptVelocity(scene);

	return true;
}

void LinearizedImplicitEuler::performAngularGlobalMultiply(
		const SIMManager& scene, const scalar& dt, const VectorXs& m,
		const VectorXs& v, VectorXs& out) {
	const int num_elasto = scene.getNumSoftElastoParticles();
	struct Triplets_override {
		int m_row, m_col;
		scalar m_value;
	};

	// Ax
	threadutils::for_each(0, num_elasto, [&](int i) {
		const int idata_start = m_angular_triA_sup[i].first;
		const int idata_end = m_angular_triA_sup[i].second;

		scalar val = 0.0;
		for (int j = idata_start; j < idata_end; ++j) {
			const Triplets_override& tri =
					*((const Triplets_override*)&m_angular_triA[j]);

			val += tri.m_value * v[tri.m_col];
		}
		out[i] = val * dt * dt + m[i * 4 + 3] * v[i];
	});
}

void LinearizedImplicitEuler::performGlobalMultiply(const SIMManager& scene,
													const scalar& dt,
													const VectorXs& m,
													const VectorXs& vec,
													VectorXs& out) {
	const int num_elasto = scene.getNumSoftElastoParticles();

	if (num_elasto == 0) return;

	struct Triplets_override {
		int m_row, m_col;
		scalar m_value;
	};

	const int ndof = num_elasto * 4;

	if (m_multiply_buffer.size() != num_elasto * 4)
		m_multiply_buffer.resize(ndof);
	// Ax
	threadutils::for_each(0, ndof, [&](int i) {
		const int idata_start = m_triA_sup[i].first;
		const int idata_end = m_triA_sup[i].second;

		scalar val = 0.0;
		for (int j = idata_start; j < idata_end; ++j) {
			const Triplets_override& tri = *((const Triplets_override*)&m_triA[j]);

			val += tri.m_value * vec[tri.m_col];
		}
		m_multiply_buffer[i] = val;
	});

	out = m_multiply_buffer * (dt * dt) +
				VectorXs(m.segment(0, ndof).array() * vec.array());
}

void LinearizedImplicitEuler::performGlobalMultiply(
		const SIMManager& scene, const scalar& dt,
		const std::vector<VectorXs>& node_m_x,
		const std::vector<VectorXs>& node_m_y,
		const std::vector<VectorXs>& node_m_z,
		const std::vector<VectorXs>& node_v_x,
		const std::vector<VectorXs>& node_v_y,
		const std::vector<VectorXs>& node_v_z,
		std::vector<VectorXs>& out_node_vec_x,
		std::vector<VectorXs>& out_node_vec_y,
		std::vector<VectorXs>& out_node_vec_z) {
	const int num_elasto = scene.getNumSoftElastoParticles();

	if (num_elasto == 0) return;
	if (m_multiply_buffer.size() != num_elasto * 4)
		m_multiply_buffer.resize(num_elasto * 4);
	if (m_pre_mult_buffer.size() != num_elasto * 4)
		m_pre_mult_buffer.resize(num_elasto * 4);

	// Wx
	mapNodeToSoftParticles(scene, node_v_x, node_v_y, node_v_z,
												 m_pre_mult_buffer);

	struct Triplets_override {
		int m_row, m_col;
		scalar m_value;
	};

	// AWx
	threadutils::for_each(0, num_elasto * 4, [&](int i) {
		const int idata_start = m_triA_sup[i].first;
		const int idata_end = m_triA_sup[i].second;

		scalar val = 0.0;
		for (int j = idata_start; j < idata_end; ++j) {
			const Triplets_override& tri = *((const Triplets_override*)&m_triA[j]);

			val += tri.m_value * m_pre_mult_buffer[tri.m_col];
		}
		m_multiply_buffer[i] = val;
	});
	//    m_multiply_buffer = m_A * m_pre_mult_buffer;

	// W^TAWx
	mapSoftParticlesToNode(scene, out_node_vec_x, out_node_vec_y, out_node_vec_z,
												 m_multiply_buffer);

	const Sorter& buckets = scene.getParticleBuckets();

	buckets.for_each_bucket([&](int bucket_idx) {
		VectorXs& bucket_node_vec_x = out_node_vec_x[bucket_idx];
		VectorXs& bucket_node_vec_y = out_node_vec_y[bucket_idx];
		VectorXs& bucket_node_vec_z = out_node_vec_z[bucket_idx];

		// (M+h^2(W^TAW+H)x
		bucket_node_vec_x =
				bucket_node_vec_x * (dt * dt) +
				VectorXs(node_m_x[bucket_idx].array() * node_v_x[bucket_idx].array());
		bucket_node_vec_y =
				bucket_node_vec_y * (dt * dt) +
				VectorXs(node_m_y[bucket_idx].array() * node_v_y[bucket_idx].array());
		bucket_node_vec_z =
				bucket_node_vec_z * (dt * dt) +
				VectorXs(node_m_z[bucket_idx].array() * node_v_z[bucket_idx].array());
		//		std::cout << bucket_node_vec << std::endl;
	});
}

void LinearizedImplicitEuler::performGlobalMultiply(
		const SIMManager& scene, const scalar& dt,
		const std::vector<VectorXs>& node_m_x,
		const std::vector<VectorXs>& node_m_y,
		const std::vector<VectorXs>& node_m_z,
		const std::vector<VectorXs>& node_v_x,
		const std::vector<VectorXs>& node_v_y,
		const std::vector<VectorXs>& node_v_z,
		std::vector<VectorXs>& out_node_vec_x,
		std::vector<VectorXs>& out_node_vec_y,
		std::vector<VectorXs>& out_node_vec_z, const VectorXs& m,
		const VectorXs& angular_vec, VectorXs& out) {
	const int num_elasto = scene.getNumSoftElastoParticles();

	if (num_elasto == 0) return;
	if (m_multiply_buffer.size() != num_elasto * 4)
		m_multiply_buffer.resize(num_elasto * 4);
	if (m_pre_mult_buffer.size() != num_elasto * 4)
		m_pre_mult_buffer.resize(num_elasto * 4);

	// Wx
	mapNodeToSoftParticles(scene, node_v_x, node_v_y, node_v_z, m_pre_mult_buffer);

	// Redistribute angular DOFs
	threadutils::for_each(0, num_elasto, [&](int i) {
		m_pre_mult_buffer[i * 4 + 3] = angular_vec(i);
	});

	struct Triplets_override {
		int m_row, m_col;
		scalar m_value;
	};

	// AWx
	threadutils::for_each(0, num_elasto * 4, [&](int i) {
		const int idata_start = m_triA_sup[i].first;
		const int idata_end = m_triA_sup[i].second;

		scalar val = 0.0;
		for (int j = idata_start; j < idata_end; ++j) {
			const Triplets_override& tri = *((const Triplets_override*)&m_triA[j]);

			val += tri.m_value * m_pre_mult_buffer[tri.m_col];
		}
		m_multiply_buffer[i] = val;
	});

	// grab angular DOFs back
	threadutils::for_each(0, num_elasto, [&](int i) {
		out(i) = m_multiply_buffer[i * 4 + 3] * (dt * dt) +
						 m(i * 4 + 3) * angular_vec(i);
		m_multiply_buffer[i * 4 + 3] = 0.0;
	});

	// W^TAWx
	mapSoftParticlesToNode(scene, out_node_vec_x, out_node_vec_y, out_node_vec_z, m_multiply_buffer);

	const Sorter& buckets = scene.getParticleBuckets();

	buckets.for_each_bucket([&](int bucket_idx) {
		VectorXs& bucket_node_vec_x = out_node_vec_x[bucket_idx];
		VectorXs& bucket_node_vec_y = out_node_vec_y[bucket_idx];
		VectorXs& bucket_node_vec_z = out_node_vec_z[bucket_idx];

		// (M+h^2(W^TAW+H)x
		bucket_node_vec_x = bucket_node_vec_x * (dt * dt) +
				VectorXs(node_m_x[bucket_idx].array() * node_v_x[bucket_idx].array());
		bucket_node_vec_y = bucket_node_vec_y * (dt * dt) +
				VectorXs(node_m_y[bucket_idx].array() * node_v_y[bucket_idx].array());
		bucket_node_vec_z = bucket_node_vec_z * (dt * dt) +
				VectorXs(node_m_z[bucket_idx].array() * node_v_z[bucket_idx].array());
		//		std::cout << bucket_node_vec << std::endl;
	});
}

void LinearizedImplicitEuler::constructNodeForce(
		SIMManager& scene, const scalar& dt, std::vector<VectorXs>& node_rhs_x,
		std::vector<VectorXs>& node_rhs_y, std::vector<VectorXs>& node_rhs_z) {
	const int num_elasto = scene.getNumSoftElastoParticles();
	const VectorXs& m = scene.getM();
	const VectorXs& v = scene.getV();

	int ndof = num_elasto * 4;

	VectorXs rhs = VectorXs::Zero(ndof);

	m_angular_moment_buffer.resize(num_elasto);

	if (scene.getLiquidInfo().solve_solid) {
		scene.accumulateGradU(rhs);
		rhs *= -dt;

		assert(!std::isnan(rhs.sum()));

		threadutils::for_each(0, num_elasto, [&](int pidx) {
			m_angular_moment_buffer[pidx] = rhs[pidx * 4 + 3] + m[pidx * 4 + 3] * v[pidx * 4 + 3];
		});

		mapSoftParticlesToNode(scene, node_rhs_x, node_rhs_y, node_rhs_z, rhs);
	}


	if (scene.getLiquidInfo().solve_solid) {
		MatrixXs rhs_gauss(scene.getNumGausses() * 3, 3);
		rhs_gauss.setZero();
		scene.accumulateGaussGradU(rhs_gauss);  // force for type 3.
		rhs_gauss *= -dt;

		assert(!std::isnan(rhs_gauss.sum()));

		mapGaussToNode(scene, node_rhs_x, node_rhs_y, node_rhs_z, rhs_gauss);
	}
	const Sorter& buckets = scene.getParticleBuckets();

	buckets.for_each_bucket([&](int bucket_idx) {
		if (scene.getLiquidInfo().solve_solid) {
			const VectorXs& node_masses_x = scene.getNodeMassX()[bucket_idx];
			const VectorXs& node_masses_y = scene.getNodeMassY()[bucket_idx];
			const VectorXs& node_masses_z = scene.getNodeMassZ()[bucket_idx];

			const VectorXs& node_vel_x = scene.getNodeVelocityX()[bucket_idx];
			const VectorXs& node_vel_y = scene.getNodeVelocityY()[bucket_idx];
			const VectorXs& node_vel_z = scene.getNodeVelocityZ()[bucket_idx];

			node_rhs_x[bucket_idx] += VectorXs(node_masses_x.array() * node_vel_x.array());
			node_rhs_y[bucket_idx] += VectorXs(node_masses_y.array() * node_vel_y.array());
			node_rhs_z[bucket_idx] += VectorXs(node_masses_z.array() * node_vel_z.array());

			assert(!std::isnan(node_rhs_x[bucket_idx].sum()));
			assert(!std::isnan(node_rhs_y[bucket_idx].sum()));
			assert(!std::isnan(node_rhs_z[bucket_idx].sum()));
		}

	});
}


void LinearizedImplicitEuler::constructHessianPreProcess(SIMManager& scene, const scalar& dt) {
	scene.accumulateddUdxdx(m_triA, dt, 0);

	m_triA.erase(std::remove_if(m_triA.begin(), m_triA.end(),
				[](const auto& info) { return info.value() == 0.0; }),
			m_triA.end());
}

void LinearizedImplicitEuler::constructHessianPostProcess(SIMManager& scene, const scalar& dt) {
	const int num_soft_elasto = scene.getNumSoftElastoParticles();

	boost::sort::block_indirect_sort(
			m_triA.begin(), m_triA.end(),
			[](const Triplets& x, const Triplets& y) { return x.row() < y.row(); });

	if ((int)m_triA_sup.size() != num_soft_elasto * 4)
		m_triA_sup.resize(num_soft_elasto * 4);

	memset(&m_triA_sup[0], 0, num_soft_elasto * 4 * sizeof(std::pair<int, int>));

	const int num_tris = m_triA.size();

	threadutils::for_each(0, num_tris, [&](int pidx) {
		int G_ID = pidx;
		int G_ID_PREV = G_ID - 1;
		int G_ID_NEXT = G_ID + 1;

		unsigned int cell = m_triA[G_ID].row();
		unsigned int cell_prev = G_ID_PREV < 0 ? -1U : m_triA[G_ID_PREV].row();
		unsigned int cell_next =
				G_ID_NEXT >= num_tris ? -1U : m_triA[G_ID_NEXT].row();

		if (cell != cell_prev) {
			m_triA_sup[cell].first = G_ID;
		}

		if (cell != cell_next) {
			m_triA_sup[cell].second = G_ID_NEXT;
		}
	});

	if (scene.getLiquidInfo().use_group_precondition) {
		prepareGroupPrecondition(scene, m_node_Cs_x, m_node_Cs_y, m_node_Cs_z, dt);
	}
}

void LinearizedImplicitEuler::constructAngularHessianPreProcess(SIMManager& scene, const scalar& dt) {
	scene.accumulateAngularddUdxdx(m_angular_triA, dt, 0);
	m_angular_triA.erase(
		std::remove_if(m_angular_triA.begin(), m_angular_triA.end(),
						[](const auto& info) { return info.value() == 0.0; }),
		m_angular_triA.end());
}

void LinearizedImplicitEuler::constructAngularHessianPostProcess(
		SIMManager& scene, const scalar&) {
	const int num_soft_elasto = scene.getNumSoftElastoParticles();

	boost::sort::block_indirect_sort(
			m_angular_triA.begin(), m_angular_triA.end(),
			[](const Triplets& x, const Triplets& y) { return x.row() < y.row(); });

	if ((int)m_angular_triA_sup.size() != num_soft_elasto)
		m_angular_triA_sup.resize(num_soft_elasto);

	memset(&m_angular_triA_sup[0], 0, num_soft_elasto * sizeof(std::pair<int, int>));

	const int num_tris = m_angular_triA.size();

	threadutils::for_each(0, num_tris, [&](int pidx) {
		int G_ID = pidx;
		int G_ID_PREV = G_ID - 1;
		int G_ID_NEXT = G_ID + 1;

		unsigned int cell = m_angular_triA[G_ID].row();
		unsigned int cell_prev =
				G_ID_PREV < 0 ? -1U : m_angular_triA[G_ID_PREV].row();
		unsigned int cell_next =
				G_ID_NEXT >= num_tris ? -1U : m_angular_triA[G_ID_NEXT].row();

		if (cell != cell_prev) {
			m_angular_triA_sup[cell].first = G_ID;
		}

		if (cell != cell_next) {
			m_angular_triA_sup[cell].second = G_ID_NEXT;
		}
	});
}

void LinearizedImplicitEuler::constructMsDVs(SIMManager& scene) {

	const Sorter& buckets = scene.getParticleBuckets();

	buckets.for_each_bucket([&](int bucket_idx) {
		if (!scene.isBucketActivated(bucket_idx)) return;

		const int num_nodes = scene.getNumNodes(bucket_idx);

		for (int i = 0; i < num_nodes; ++i) {
			scalar P = m_node_mfhdvm_hdvm_x[bucket_idx][i];
			scalar Cs = m_node_damped_x[bucket_idx][i];
			m_node_Cs_x[bucket_idx][i] = Cs;
			if (Cs > 1e-20) {
				m_node_inv_Cs_x[bucket_idx][i] = 1.0 / Cs;
			} else {
				m_node_inv_Cs_x[bucket_idx][i] = 1.0;
			}

			P = m_node_mfhdvm_hdvm_y[bucket_idx][i];
			Cs = m_node_damped_y[bucket_idx][i];
			m_node_Cs_y[bucket_idx][i] = Cs;
			if (Cs > 1e-20) {
				m_node_inv_Cs_y[bucket_idx][i] = 1.0 / Cs;
			} else {
				m_node_inv_Cs_y[bucket_idx][i] = 1.0;
			}

			P = m_node_mfhdvm_hdvm_z[bucket_idx][i];
			Cs = m_node_damped_z[bucket_idx][i];
			m_node_Cs_z[bucket_idx][i] = Cs;
			if (Cs > 1e-20) {
				m_node_inv_Cs_z[bucket_idx][i] = 1.0 / Cs;
			} else {
				m_node_inv_Cs_z[bucket_idx][i] = 1.0;
			}
		}
	});
}

void LinearizedImplicitEuler::constructInvMDV(SIMManager& scene) {

	const std::vector<VectorXs>& node_mass_x = scene.getNodeMassX();
	const std::vector<VectorXs>& node_mass_y = scene.getNodeMassY();
	const std::vector<VectorXs>& node_mass_z = scene.getNodeMassZ();
	m_node_damped_x = node_mass_x;
	m_node_damped_y = node_mass_y;
	m_node_damped_z = node_mass_z;

	const Sorter& buckets = scene.getParticleBuckets();

	buckets.for_each_bucket([&](int bucket_idx) {
		if (!scene.isBucketActivated(bucket_idx)) return;

		const int num_nodes = scene.getNumNodes(bucket_idx);

		for (int i = 0; i < num_nodes; ++i) {
			scalar Q = m_node_mshdvm_hdvm_x[bucket_idx][i];
			scalar C = Q * m_node_damped_x[bucket_idx][i];
			if (C > 1e-20) {
				m_node_inv_C_x[bucket_idx][i] = 1.0 / C;
			} else {
				m_node_inv_C_x[bucket_idx][i] = 1.0;
			}

			Q = m_node_mshdvm_hdvm_y[bucket_idx][i];
			C = Q * m_node_damped_y[bucket_idx][i];
			if (C > 1e-20) {
				m_node_inv_C_y[bucket_idx][i] = 1.0 / C;
			} else {
				m_node_inv_C_y[bucket_idx][i] = 1.0;
			}

			Q = m_node_mshdvm_hdvm_z[bucket_idx][i];
			C = Q * m_node_damped_z[bucket_idx][i];
			if (C > 1e-20) {
				m_node_inv_C_z[bucket_idx][i] = 1.0 / C;
			} else {
				m_node_inv_C_z[bucket_idx][i] = 1.0;
			}
		}
	});
}

bool LinearizedImplicitEuler::stepImplicitElastoDiagonalPCR(SIMManager& scene, scalar dt) {
	int ndof_elasto = scene.getNumSoftElastoParticles() * 4;
	const Sorter& buckets = scene.getParticleBuckets();

	if (ndof_elasto == 0) return true;

	scalar res_norm_0 = lengthNodeVectors(m_node_rhs_x, m_node_rhs_y, m_node_rhs_z);
	scalar res_norm_1 = m_angular_moment_buffer.norm();

	if (res_norm_0 > m_pcg_criterion) {
		constructHessianPreProcess(scene, dt);
		constructHessianPostProcess(scene, dt);

		allocateNodeVectors(scene, m_node_r_x, m_node_r_y, m_node_r_z);
		allocateNodeVectors(scene, m_node_z_x, m_node_z_y, m_node_z_z);
		allocateNodeVectors(scene, m_node_p_x, m_node_p_y, m_node_p_z);
		allocateNodeVectors(scene, m_node_q_x, m_node_q_y, m_node_q_z);
		allocateNodeVectors(scene, m_node_w_x, m_node_w_y, m_node_w_z);
		allocateNodeVectors(scene, m_node_t_x, m_node_t_y, m_node_t_z);

		// initial residual = b - Ax
		performGlobalMultiply(scene, dt, m_node_Cs_x, m_node_Cs_y, m_node_Cs_z,
							m_node_v_plus_x, m_node_v_plus_y, m_node_v_plus_z,
							m_node_z_x, m_node_z_y, m_node_z_z);

		buckets.for_each_bucket([&](int bucket_idx) {
			m_node_z_x[bucket_idx] = m_node_rhs_x[bucket_idx] - m_node_z_x[bucket_idx];
			m_node_z_y[bucket_idx] = m_node_rhs_y[bucket_idx] - m_node_z_y[bucket_idx];
			m_node_z_z[bucket_idx] = m_node_rhs_z[bucket_idx] - m_node_z_z[bucket_idx];
		});

		scalar res_norm = lengthNodeVectors(m_node_z_x, m_node_z_y, m_node_z_z) / res_norm_0;

		int iter = 0;

		if (res_norm < m_pcg_criterion) {
			std::cout << "[pcr total iter: " << iter << ", res: " << res_norm << "/"
						<< m_pcg_criterion << ", abs. res: " << (res_norm * res_norm_0)
						<< "/" << (m_pcg_criterion * res_norm_0) << "]" << std::endl;
		} else {
			// Solve Mr=z
			if (scene.getLiquidInfo().use_group_precondition) {
				performGroupedLocalSolve(scene, m_node_z_x, m_node_z_y, m_node_z_z, m_node_r_x, m_node_r_y, m_node_r_z);
			} else {
				performInvLocalSolve(scene, m_node_z_x, m_node_z_y, m_node_z_z,
									m_node_inv_Cs_x, m_node_inv_Cs_y, m_node_inv_Cs_z,
									m_node_r_x, m_node_r_y, m_node_r_z);
			}
			// p = r
			buckets.for_each_bucket([&](int bucket_idx) {
				m_node_p_x[bucket_idx] = m_node_r_x[bucket_idx];
				m_node_p_y[bucket_idx] = m_node_r_y[bucket_idx];
				m_node_p_z[bucket_idx] = m_node_r_z[bucket_idx];
			});

			// t = z
			buckets.for_each_bucket([&](int bucket_idx) {
				m_node_t_x[bucket_idx] = m_node_z_x[bucket_idx];
				m_node_t_y[bucket_idx] = m_node_z_y[bucket_idx];
				m_node_t_z[bucket_idx] = m_node_z_z[bucket_idx];
			});

			// w = Ar
			performGlobalMultiply(scene, dt, m_node_Cs_x, m_node_Cs_y, m_node_Cs_z,
									m_node_r_x, m_node_r_y, m_node_r_z, m_node_w_x,
									m_node_w_y, m_node_w_z);

			// rho = (r, w)
			scalar rho = dotNodeVectors(m_node_r_x, m_node_r_y, m_node_r_z, m_node_w_x, m_node_w_y, m_node_w_z);

			// q = Ap
			performGlobalMultiply(scene, dt, m_node_Cs_x, m_node_Cs_y, m_node_Cs_z,
								m_node_p_x, m_node_p_y, m_node_p_z, m_node_q_x,
								m_node_q_y, m_node_q_z);

			if (scene.getLiquidInfo().use_group_precondition) {
				performGroupedLocalSolve(scene, m_node_q_x, m_node_q_y, m_node_q_z,
										m_node_z_x, m_node_z_y, m_node_z_z);
			} else {
				// Mz=q
				performInvLocalSolve(scene, m_node_q_x, m_node_q_y, m_node_q_z,
									m_node_inv_Cs_x, m_node_inv_Cs_y, m_node_inv_Cs_z,
									m_node_z_x, m_node_z_y, m_node_z_z);
			}

			// alpha = rho / (q, z)
			scalar alpha = rho / dotNodeVectors(m_node_q_x, m_node_q_y, m_node_q_z, m_node_z_x, m_node_z_y, m_node_z_z);

			// x = x + alpha * p
			// r = r - alpha * z
			// t = t - alpha * q
			buckets.for_each_bucket([&](int bucket_idx) {
				m_node_v_plus_x[bucket_idx] += m_node_p_x[bucket_idx] * alpha;
				m_node_r_x[bucket_idx] -= m_node_z_x[bucket_idx] * alpha;
				m_node_t_x[bucket_idx] -= m_node_q_x[bucket_idx] * alpha;
				m_node_v_plus_y[bucket_idx] += m_node_p_y[bucket_idx] * alpha;
				m_node_r_y[bucket_idx] -= m_node_z_y[bucket_idx] * alpha;
				m_node_t_y[bucket_idx] -= m_node_q_y[bucket_idx] * alpha;
				m_node_v_plus_z[bucket_idx] += m_node_p_z[bucket_idx] * alpha;
				m_node_r_z[bucket_idx] -= m_node_z_z[bucket_idx] * alpha;
				m_node_t_z[bucket_idx] -= m_node_q_z[bucket_idx] * alpha;
			});

			res_norm = lengthNodeVectors(m_node_t_x, m_node_t_y, m_node_t_z) / res_norm_0;

			const scalar rho_criterion = (m_pcg_criterion * res_norm_0) * (m_pcg_criterion * res_norm_0);

			scalar rho_old, beta;
			for (; iter < m_maxiters && res_norm > m_pcg_criterion &&
						 rho > rho_criterion;
					 ++iter) {
				rho_old = rho;

				// w = Ar
				performGlobalMultiply(scene, dt, m_node_Cs_x, m_node_Cs_y, m_node_Cs_z,
															m_node_r_x, m_node_r_y, m_node_r_z, m_node_w_x,
															m_node_w_y, m_node_w_z);

				// rho = (r, w)
				rho = dotNodeVectors(m_node_r_x, m_node_r_y, m_node_r_z, m_node_w_x,
														 m_node_w_y, m_node_w_z);

				beta = rho / rho_old;

				// p = beta * p + r
				// q = beta * q + w
				buckets.for_each_bucket([&](int bucket_idx) {
					m_node_p_x[bucket_idx] = m_node_r_x[bucket_idx] + m_node_p_x[bucket_idx] * beta;
					m_node_p_y[bucket_idx] = m_node_r_y[bucket_idx] + m_node_p_y[bucket_idx] * beta;
					m_node_p_z[bucket_idx] = m_node_r_z[bucket_idx] + m_node_p_z[bucket_idx] * beta;
					m_node_q_x[bucket_idx] = m_node_w_x[bucket_idx] + m_node_q_x[bucket_idx] * beta;
					m_node_q_y[bucket_idx] = m_node_w_y[bucket_idx] + m_node_q_y[bucket_idx] * beta;
					m_node_q_z[bucket_idx] = m_node_w_z[bucket_idx] + m_node_q_z[bucket_idx] * beta;
				});

				// Mz = q
				if (scene.getLiquidInfo().use_group_precondition) {
					performGroupedLocalSolve(scene, m_node_q_x, m_node_q_y, m_node_q_z,
											m_node_z_x, m_node_z_y, m_node_z_z);
				} else {
					// Mz=q
					performInvLocalSolve(scene, m_node_q_x, m_node_q_y, m_node_q_z,
										m_node_inv_Cs_x, m_node_inv_Cs_y,
										m_node_inv_Cs_z, m_node_z_x, m_node_z_y,
										m_node_z_z);
				}

				// alpha = rho / (q, z)
				alpha = rho / dotNodeVectors(m_node_q_x, m_node_q_y, m_node_q_z,
											m_node_z_x, m_node_z_y, m_node_z_z);

				// x = x + alpha * p
				// r = r - alpha * z
				// t = t - alpha * q
				buckets.for_each_bucket([&](int bucket_idx) {
					m_node_v_plus_x[bucket_idx] += m_node_p_x[bucket_idx] * alpha;
					m_node_r_x[bucket_idx] -= m_node_z_x[bucket_idx] * alpha;
					m_node_t_x[bucket_idx] -= m_node_q_x[bucket_idx] * alpha;
					m_node_v_plus_y[bucket_idx] += m_node_p_y[bucket_idx] * alpha;
					m_node_r_y[bucket_idx] -= m_node_z_y[bucket_idx] * alpha;
					m_node_t_y[bucket_idx] -= m_node_q_y[bucket_idx] * alpha;
					m_node_v_plus_z[bucket_idx] += m_node_p_z[bucket_idx] * alpha;
					m_node_r_z[bucket_idx] -= m_node_z_z[bucket_idx] * alpha;
					m_node_t_z[bucket_idx] -= m_node_q_z[bucket_idx] * alpha;
				});

				res_norm = lengthNodeVectors(m_node_t_x, m_node_t_y, m_node_t_z) / res_norm_0;
				if (scene.getLiquidInfo().iteration_print_step > 0 &&
						iter % scene.getLiquidInfo().iteration_print_step == 0)
					std::cout << "[pcr total iter: " << iter << ", res: " << res_norm
										<< "/" << m_pcg_criterion
										<< ", abs. res: " << (res_norm * res_norm_0) << "/"
										<< (m_pcg_criterion * res_norm_0)
										<< ", rho: " << (rho / (res_norm_0 * res_norm_0)) << "/"
										<< (rho_criterion / (res_norm_0 * res_norm_0))
										<< ", abs. rho: " << rho << "/" << rho_criterion << "]"
										<< std::endl;
			}

			std::cout << "[pcr total iter: " << iter << ", res: " << res_norm << "/"
								<< m_pcg_criterion << ", abs. res: " << (res_norm * res_norm_0)
								<< "/" << (m_pcg_criterion * res_norm_0)
								<< ", rho: " << (rho / (res_norm_0 * res_norm_0)) << "/"
								<< (rho_criterion / (res_norm_0 * res_norm_0))
								<< ", abs. rho: " << rho << "/" << rho_criterion << "]"
								<< std::endl;
		}
	}

	if (res_norm_1 > m_pcg_criterion) {
		const int num_elasto = scene.getNumSoftElastoParticles();

		constructAngularHessianPreProcess(scene, dt);
		constructAngularHessianPostProcess(scene, dt);

		m_angular_r.resize(num_elasto);
		m_angular_z.resize(num_elasto);
		m_angular_p.resize(num_elasto);
		m_angular_q.resize(num_elasto);
		m_angular_w.resize(num_elasto);
		m_angular_t.resize(num_elasto);

		m_angular_r.setZero();
		m_angular_z.setZero();
		m_angular_p.setZero();
		m_angular_q.setZero();
		m_angular_w.setZero();
		m_angular_t.setZero();

		performAngularGlobalMultiply(scene, dt, scene.getM(), m_angular_v_plus_buffer, m_angular_z);

		m_angular_z = m_angular_moment_buffer - m_angular_z;

		scalar res_norm = m_angular_z.norm() / res_norm_1;

		int iter = 0;

		if (res_norm < m_pcg_criterion) {
			std::cout << "[angular pcr total iter: " << iter << ", res: " << res_norm
								<< "/" << m_pcg_criterion
								<< ", abs. res: " << (res_norm * res_norm_1) << "/"
								<< (m_pcg_criterion * res_norm_1) << "]" << std::endl;
		} else {
			// Solve Mr=z
			performLocalSolveTwist(scene, m_angular_z, scene.getM(), m_angular_r);

			// p = r
			m_angular_p = m_angular_r;

			// t = z
			m_angular_t = m_angular_z;

			// w = Ar
			performAngularGlobalMultiply(scene, dt, scene.getM(), m_angular_r, m_angular_w);

			// rho = (r, w)
			scalar rho = m_angular_r.dot(m_angular_w);

			// q = Ap
			performAngularGlobalMultiply(scene, dt, scene.getM(), m_angular_p, m_angular_q);

			// Mz=q
			performLocalSolveTwist(scene, m_angular_q, scene.getM(), m_angular_z);

			// alpha = rho / (q, z)
			scalar alpha = rho / m_angular_q.dot(m_angular_z);

			// x = x + alpha * p
			// r = r - alpha * z
			// t = t - alpha * q
			m_angular_v_plus_buffer += m_angular_p * alpha;
			m_angular_r -= m_angular_z * alpha;
			m_angular_t -= m_angular_q * alpha;

			res_norm = m_angular_t.norm() / res_norm_1;

			const scalar rho_criterion = (m_pcg_criterion * res_norm_1) * (m_pcg_criterion * res_norm_1);

			scalar rho_old, beta;
			for (; iter < m_maxiters && res_norm > m_pcg_criterion && rho > rho_criterion; ++iter) {
				rho_old = rho;

				// w = Ar
				performAngularGlobalMultiply(scene, dt, scene.getM(), m_angular_r, m_angular_w);

				// rho = (r, w)
				rho = m_angular_r.dot(m_angular_w);

				beta = rho / rho_old;

				// p = beta * p + r
				// q = beta * q + w
				m_angular_p = m_angular_r + m_angular_p * beta;
				m_angular_q = m_angular_w + m_angular_q * beta;

				// Mz = q
				performLocalSolveTwist(scene, m_angular_q, scene.getM(), m_angular_z);

				// alpha = rho / (q, z)
				alpha = rho / m_angular_q.dot(m_angular_z);

				// x = x + alpha * p
				// r = r - alpha * z
				// t = t - alpha * q
				m_angular_v_plus_buffer += m_angular_p * alpha;
				m_angular_r -= m_angular_z * alpha;
				m_angular_t -= m_angular_q * alpha;

				res_norm = m_angular_t.norm() / res_norm_1;

				if (scene.getLiquidInfo().iteration_print_step > 0 &&
						iter % scene.getLiquidInfo().iteration_print_step == 0)
					std::cout << "[angular pcr total iter: " << iter
										<< ", res: " << res_norm << "/" << m_pcg_criterion
										<< ", abs. res: " << (res_norm * res_norm_1) << "/"
										<< (m_pcg_criterion * res_norm_1)
										<< ", rho: " << (rho / (res_norm_1 * res_norm_1)) << "/"
										<< (rho_criterion / (res_norm_1 * res_norm_1))
										<< ", abs. rho: " << rho << "/" << rho_criterion << "]"
										<< std::endl;
			}

			std::cout << "[angular pcr total iter: " << iter << ", res: " << res_norm
								<< "/" << m_pcg_criterion
								<< ", abs. res: " << (res_norm * res_norm_1) << "/"
								<< (m_pcg_criterion * res_norm_1)
								<< ", rho: " << (rho / (res_norm_1 * res_norm_1)) << "/"
								<< (rho_criterion / (res_norm_1 * res_norm_1))
								<< ", abs. rho: " << rho << "/" << rho_criterion << "]"
								<< std::endl;
		}
	}
	return true;
}

void LinearizedImplicitEuler::performGroupedLocalSolve(
		const SIMManager& scene, const std::vector<VectorXs>& node_rhs_x,
		const std::vector<VectorXs>& node_rhs_y,
		const std::vector<VectorXs>& node_rhs_z,
		std::vector<VectorXs>& out_node_vec_x,
		std::vector<VectorXs>& out_node_vec_y,
		std::vector<VectorXs>& out_node_vec_z) {
	const std::vector<VectorXi>& groups = scene.getSolveGroup();

	const int num_groups = (int)groups.size();

	VectorXs rhs_buffer(scene.getNumSoftElastoParticles() * 4);
	rhs_buffer.setZero();

	VectorXs sol_buffer(scene.getNumSoftElastoParticles() * 4);
	sol_buffer.setZero();

	mapNodeToSoftParticles(scene, node_rhs_x, node_rhs_y, node_rhs_z, rhs_buffer);

	threadutils::for_each(0, num_groups, [&](int igroup) {
		const VectorXi& members = groups[igroup];
		const int num_members = members.size();

		VectorXs group_rhs(num_members * 3);
		for (int i = 0; i < num_members; ++i) {
			group_rhs.segment<3>(i * 3) = rhs_buffer.segment<3>(members[i] * 4);
		}

		VectorXs group_sol = m_group_preconditioners[igroup]->solve(group_rhs);

		for (int i = 0; i < num_members; ++i) {
			sol_buffer.segment<3>(members[i] * 4) = group_sol.segment<3>(i * 3);
		}
	});

	mapSoftParticlesToNode(scene, out_node_vec_x, out_node_vec_y, out_node_vec_z, sol_buffer);
}

void LinearizedImplicitEuler::prepareGroupPrecondition(
		const SIMManager& scene, const std::vector<VectorXs>& node_m_x,
		const std::vector<VectorXs>& node_m_y,
		const std::vector<VectorXs>& node_m_z, const scalar& dt) {
	const std::vector<VectorXi>& groups = scene.getSolveGroup();

	const int num_groups = (int)groups.size();

	m_group_preconditioners.resize(num_groups);

	VectorXs mass_buffer(scene.getNumSoftElastoParticles() * 4);
	mass_buffer.setZero();
	// map drag + mass vector back to particles
	mapNodeToSoftParticles(scene, node_m_x, node_m_y, node_m_z, mass_buffer);

	threadutils::for_each(0, num_groups, [&](int igroup) {
		const VectorXi& members = groups[igroup];
		std::unordered_map<int, int> finder;

		const int num_members = members.size();

		for (int i = 0; i < num_members; ++i) {
			finder[members[i]] = i;
		}

		TripletXs tri_sub_A;

		for (auto p : finder) {
			for (int r = 0; r < 3; ++r) {
				int i = p.first * 4 + r;
				const int idata_start = m_triA_sup[i].first;
				const int idata_end = m_triA_sup[i].second;

				for (int j = idata_start; j < idata_end; ++j) {
					const Triplets& tri = m_triA[j];
					const int qidx = tri.col() / 4;
					const int s = tri.col() - qidx * 4;
					if (s >= 3) continue;

					auto q = finder.find(qidx);
					if (q == finder.end()) continue;

					tri_sub_A.push_back(Triplets(p.second * 3 + r, q->second * 3 + s, tri.value() * dt * dt));
				}
			}
		}

		for (int i = 0; i < num_members; ++i) {
			const int pidx = members[i];

			for (int r = 0; r < 3; ++r) {
				tri_sub_A.push_back(Triplets(i * 3 + r, i * 3 + r, mass_buffer[pidx * 4 + r]));
			}
		}

		SparseXs sub_A(num_members * 3, num_members * 3);
		sub_A.setFromTriplets(tri_sub_A.begin(), tri_sub_A.end());

		m_group_preconditioners[igroup] = std::make_shared<Eigen::SimplicialLDLT<SparseXs> >(sub_A);
	});
}


bool LinearizedImplicitEuler::acceptVelocity(SIMManager& scene) {
	const Sorter& buckets = scene.getParticleBuckets();

	if (scene.getLiquidInfo().solve_solid &&
			scene.getNumSoftElastoParticles() > 0) {
		buckets.for_each_bucket([&](int bucket_idx) {
			scene.getNodeVelocityX()[bucket_idx] = m_node_v_plus_x[bucket_idx];
			scene.getNodeVelocityY()[bucket_idx] = m_node_v_plus_y[bucket_idx];
			scene.getNodeVelocityZ()[bucket_idx] = m_node_v_plus_z[bucket_idx];
		});

		const int num_elasto = scene.getNumSoftElastoParticles();
		threadutils::for_each(0, num_elasto, [&](int pidx) {
			scene.getV()[pidx * 4 + 3] = m_angular_v_plus_buffer[pidx];
		});
	}

	return true;
}


bool LinearizedImplicitEuler::stepImplicitElasto(SIMManager& scene, scalar dt) {

	if (scene.getLiquidInfo().use_pcr) {
		return stepImplicitElastoDiagonalPCR(scene, dt);
	} else {
		std::cerr << "WARNING: PCR not implemented yet!" << std::endl;
		return false;
	}

	// if (scene.getLiquidInfo().use_amgpcg_solid) {
	// 	std::cout << "use_amgpcg~~~~~~~~~~~~~~~" << std::endl;
	// 	return stepImplicitElastoAMGPCG(scene, dt);
	// } else if (scene.getLiquidInfo().use_pcr) {
	// 	std::cout << "use_pcr~~~~~~~~~~~~~~~" << std::endl;
	// 	return stepImplicitElastoDiagonalPCR(scene, dt);
	// } else {
	// 	if (scene.getLiquidInfo().use_cosolve_angular) {
	// 		std::cout << "use_cosolve~~~~~~~~~~~~~~~" << std::endl;
	// 		return stepImplicitElastoDiagonalPCGCoSolve(scene, dt);
	// 	}
	// 	else {
	// 		std::cout << "use_nothing~~~~~~~~~~~~~~~" << std::endl;
	// 		return stepImplicitElastoDiagonalPCG(scene, dt);
	// 	}
	// }
}

void LinearizedImplicitEuler::pushElastoVelocity() {
	m_elasto_vel_stack.push(m_node_v_plus_x);
	m_elasto_vel_stack.push(m_node_v_plus_y);
	m_elasto_vel_stack.push(m_node_v_plus_z);

	m_elasto_vel_stack.push(m_node_rhs_x);
	m_elasto_vel_stack.push(m_node_rhs_y);
	m_elasto_vel_stack.push(m_node_rhs_z);
}

void LinearizedImplicitEuler::popElastoVelocity() {
	m_node_rhs_z = m_elasto_vel_stack.top();
	m_elasto_vel_stack.pop();
	m_node_rhs_y = m_elasto_vel_stack.top();
	m_elasto_vel_stack.pop();
	m_node_rhs_x = m_elasto_vel_stack.top();
	m_elasto_vel_stack.pop();

	m_node_v_plus_z = m_elasto_vel_stack.top();
	m_elasto_vel_stack.pop();
	m_node_v_plus_y = m_elasto_vel_stack.top();
	m_elasto_vel_stack.pop();
	m_node_v_plus_x = m_elasto_vel_stack.top();
	m_elasto_vel_stack.pop();
}

std::string LinearizedImplicitEuler::getName() const {
	return "Linearized Implicit Euler";
}

void LinearizedImplicitEuler::zeroFixedDoFs(const SIMManager& scene, VectorXs& vec) {
	int nprts = scene.getNumParticles();
	threadutils::for_each(0, nprts, [&](int i) {
		if (scene.isFixed(i) & 1) vec.segment<3>(4 * i).setZero();
		if (scene.isFixed(i) & 2) vec(4 * i + 3) = 0.0;
	});
}
