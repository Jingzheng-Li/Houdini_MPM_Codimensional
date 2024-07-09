
#include "SceneStepper.h"

#include <numeric>

SceneStepper::~SceneStepper() {}

bool SceneStepper::advectScene(SIMManager& scene, scalar dt) {
	VectorXs& x = scene.getX();
	const VectorXs& v = scene.getV();

	assert(!std::isnan(v.sum()));
	assert(!std::isnan(x.sum()));

	const int num_parts = scene.getNumParticles();
	const int num_elasto_parts = scene.getNumElastoParticles();
	threadutils::for_each(0, num_parts, [&](int pidx) {
		bool is_fluid = pidx >= num_elasto_parts;

		x.segment<4>(pidx * 4) += v.segment<4>(pidx * 4) * dt;

	});

	//  std::cout << "v: \n" << v << std::endl;
	//  std::cout << "x: \n" << x << std::endl;

	return true;
}

void SceneStepper::setUseApic(bool apic) { m_apic = apic; }

bool SceneStepper::useApic() const { return m_apic; }

void SceneStepper::mapNodeToSoftParticles(
		const SIMManager& scene, const std::vector<VectorXs>& node_vec_x,
		const std::vector<VectorXs>& node_vec_y,
		const std::vector<VectorXs>& node_vec_z, VectorXs& part_vec) const {
	part_vec.setZero();

	const int num_soft_elasto = scene.getNumSoftElastoParticles();

	threadutils::for_each(0, num_soft_elasto, [&](int pidx) {
		auto& indices_x = scene.getParticleNodesX(pidx);
		auto& indices_y = scene.getParticleNodesY(pidx);
		auto& indices_z = scene.getParticleNodesZ(pidx);

		auto& weights = scene.getParticleWeights(pidx);

		scalar sum_x(0.), sum_y(0.), sum_z(0.);
		for (int i = 0; i < 27; ++i) {
			if (!scene.isBucketActivated(indices_x(i, 0)) || weights(i, 0) == 0.0)
				continue;
			sum_x += node_vec_x[indices_x(i, 0)](indices_x(i, 1)) * weights(i, 0);
		}
		part_vec(pidx * 4 + 0) = sum_x;

		for (int i = 0; i < 27; ++i) {
			if (!scene.isBucketActivated(indices_y(i, 0)) || weights(i, 1) == 0.0)
				continue;
			sum_y += node_vec_y[indices_y(i, 0)](indices_y(i, 1)) * weights(i, 1);
		}
		part_vec(pidx * 4 + 1) = sum_y;

		for (int i = 0; i < 27; ++i) {
			if (!scene.isBucketActivated(indices_z(i, 0)) || weights(i, 2) == 0.0)
				continue;
			sum_z += node_vec_z[indices_z(i, 0)](indices_z(i, 1)) * weights(i, 2);
		}
		part_vec(pidx * 4 + 2) = sum_z;
	});
}

void SceneStepper::mapSoftParticlesToNode(const SIMManager& scene,
											std::vector<VectorXs>& node_vec_x,
											std::vector<VectorXs>& node_vec_y,
											std::vector<VectorXs>& node_vec_z,
											const VectorXs& part_vec) const {
	const Sorter& buckets = scene.getParticleBuckets();

	if ((int)node_vec_x.size() != buckets.size())
		node_vec_x.resize(buckets.size());
	if ((int)node_vec_y.size() != buckets.size())
		node_vec_y.resize(buckets.size());
	if ((int)node_vec_z.size() != buckets.size())
		node_vec_z.resize(buckets.size());

	const std::vector<int>& particle_to_surfels = scene.getParticleToSurfels();
	const int num_elasto = scene.getNumElastoParticles();
	const std::vector<Matrix27x4s>& particle_weights = scene.getParticleWeights();

	buckets.for_each_bucket([&](int bucket_idx) {
		if (!scene.isBucketActivated(bucket_idx)) return;

		const int num_nodes = scene.getNumNodes(bucket_idx);

		VectorXs& bucket_node_vec_x = node_vec_x[bucket_idx];
		VectorXs& bucket_node_vec_y = node_vec_y[bucket_idx];
		VectorXs& bucket_node_vec_z = node_vec_z[bucket_idx];

		if (bucket_node_vec_x.size() != num_nodes)
			bucket_node_vec_x.resize(num_nodes);
		if (bucket_node_vec_y.size() != num_nodes)
			bucket_node_vec_y.resize(num_nodes);
		if (bucket_node_vec_z.size() != num_nodes)
			bucket_node_vec_z.resize(num_nodes);

		for (int i = 0; i < num_nodes; ++i) {
			auto& particle_indices = scene.getNodeParticlePairsX(bucket_idx, i);

			scalar ret(0.);
			for (auto& pp : particle_indices) {
				if (pp.first >= num_elasto || particle_to_surfels[pp.first] >= 0)
					continue;

				const auto& weights = particle_weights[pp.first];

				ret += part_vec(pp.first * 4 + 0) * weights(pp.second, 0);
			}

			bucket_node_vec_x(i) = ret;
		}

		for (int i = 0; i < num_nodes; ++i) {
			auto& particle_indices = scene.getNodeParticlePairsY(bucket_idx, i);

			scalar ret(0.);
			for (auto& pp : particle_indices) {
				if (pp.first >= num_elasto || particle_to_surfels[pp.first] >= 0)
					continue;

				const auto& weights = particle_weights[pp.first];

				ret += part_vec(pp.first * 4 + 1) * weights(pp.second, 1);
			}
			bucket_node_vec_y(i) = ret;
		}

		for (int i = 0; i < num_nodes; ++i) {
			auto& particle_indices = scene.getNodeParticlePairsZ(bucket_idx, i);

			scalar ret(0.);
			for (auto& pp : particle_indices) {
				if (pp.first >= num_elasto || particle_to_surfels[pp.first] >= 0)
					continue;

				const auto& weights = particle_weights[pp.first];

				ret += part_vec(pp.first * 4 + 2) * weights(pp.second, 2);
			}
			bucket_node_vec_z(i) = ret;
		}
	});
}


void SceneStepper::mapGaussToNode(const SIMManager& scene,
																	std::vector<VectorXs>& node_vec_x,
																	std::vector<VectorXs>& node_vec_y,
																	std::vector<VectorXs>& node_vec_z,
																	const MatrixXs& gauss_vec) const {
	const Sorter& g_buckets = scene.getGaussBuckets();
	const scalar iD = scene.getInverseDCoeff();

	//  std::cout << "gauss_vec: \n" << gauss_vec << std::endl;
	const VectorXs& gauss_x = scene.getGaussX();

	g_buckets.for_each_bucket_particles_colored(
			[&](int pidx, int bucket_idx) {
				if (!scene.isBucketActivated(bucket_idx)) return;

				auto& indices_x = scene.getGaussNodesX(pidx);
				auto& indices_y = scene.getGaussNodesY(pidx);
				auto& indices_z = scene.getGaussNodesZ(pidx);

				auto& weights = scene.getGaussWeights(pidx);
				const Vector3s& pos = gauss_x.segment<3>(pidx * 4);

				for (int i = 0; i < indices_x.rows(); ++i) {
					const int node_bucket_idx = indices_x(i, 0);
					const int node_idx = indices_x(i, 1);

					if (!scene.isBucketActivated(node_bucket_idx)) continue;

					const Vector3s& np = scene.getNodePosX(node_bucket_idx, node_idx);
					node_vec_x[node_bucket_idx](node_idx) +=
							iD * weights(i, 0) *
							gauss_vec.block<1, 3>(pidx * 3 + 0, 0).dot(np - pos);
				}

				for (int i = 0; i < indices_y.rows(); ++i) {
					const int node_bucket_idx = indices_y(i, 0);
					const int node_idx = indices_y(i, 1);

					if (!scene.isBucketActivated(node_bucket_idx)) continue;

					const Vector3s& np = scene.getNodePosY(node_bucket_idx, node_idx);
					node_vec_y[node_bucket_idx](node_idx) +=
							iD * weights(i, 1) *
							gauss_vec.block<1, 3>(pidx * 3 + 1, 0).dot(np - pos);
				}

				for (int i = 0; i < indices_z.rows(); ++i) {
					const int node_bucket_idx = indices_z(i, 0);
					const int node_idx = indices_z(i, 1);

					if (!scene.isBucketActivated(node_bucket_idx)) continue;

					const Vector3s& np = scene.getNodePosZ(node_bucket_idx, node_idx);
					node_vec_z[node_bucket_idx](node_idx) +=
							iD * weights(i, 2) *
							gauss_vec.block<1, 3>(pidx * 3 + 2, 0).dot(np - pos);
				}
			},
			scene.getNumBucketColors());
}


void SceneStepper::allocateNodeVectors(
		const SIMManager& scene, std::vector<VectorXs>& node_vec_x,
		std::vector<VectorXs>& node_vec_y,
		std::vector<VectorXs>& node_vec_z) const {
	const int num_buckets = scene.getNumBuckets();

	if ((int)node_vec_x.size() != num_buckets) node_vec_x.resize(num_buckets);
	if ((int)node_vec_y.size() != num_buckets) node_vec_y.resize(num_buckets);
	if ((int)node_vec_z.size() != num_buckets) node_vec_z.resize(num_buckets);

	const Sorter& buckets = scene.getParticleBuckets();

	buckets.for_each_bucket([&](int bucket_idx) {
		const int num_nodes = scene.getNumNodes(bucket_idx);

		node_vec_x[bucket_idx].resize(num_nodes);
		node_vec_x[bucket_idx].setZero();
		node_vec_y[bucket_idx].resize(num_nodes);
		node_vec_y[bucket_idx].setZero();
		node_vec_z[bucket_idx].resize(num_nodes);
		node_vec_z[bucket_idx].setZero();
	});
}

void SceneStepper::allocateNodeVectors(
		const SIMManager& scene, std::vector<VectorXi>& node_vec_x,
		std::vector<VectorXi>& node_vec_y,
		std::vector<VectorXi>& node_vec_z) const {
	const int num_buckets = scene.getNumBuckets();

	if ((int)node_vec_x.size() != num_buckets) node_vec_x.resize(num_buckets);
	if ((int)node_vec_y.size() != num_buckets) node_vec_y.resize(num_buckets);
	if ((int)node_vec_z.size() != num_buckets) node_vec_z.resize(num_buckets);

	const Sorter& buckets = scene.getParticleBuckets();

	buckets.for_each_bucket([&](int bucket_idx) {
		const int num_nodes = scene.getNumNodes(bucket_idx);

		node_vec_x[bucket_idx].resize(num_nodes);
		node_vec_x[bucket_idx].setZero();
		node_vec_y[bucket_idx].resize(num_nodes);
		node_vec_y[bucket_idx].setZero();
		node_vec_z[bucket_idx].resize(num_nodes);
		node_vec_z[bucket_idx].setZero();
	});
}

scalar SceneStepper::dotNodeVectors(
		const std::vector<VectorXs>& node_vec_ax,
		const std::vector<VectorXs>& node_vec_ay,
		const std::vector<VectorXs>& node_vec_az,
		const std::vector<VectorXs>& node_vec_bx,
		const std::vector<VectorXs>& node_vec_by,
		const std::vector<VectorXs>& node_vec_bz) const {
	VectorXs bucket_dot(node_vec_ax.size());

	assert(node_vec_ax.size() == node_vec_bx.size());
	assert(node_vec_ay.size() == node_vec_by.size());
	assert(node_vec_az.size() == node_vec_bz.size());

	const int num_buckets = node_vec_ax.size();

	threadutils::for_each(0, num_buckets, [&](int bucket_idx) {
		bucket_dot[bucket_idx] =
				node_vec_ax[bucket_idx].dot(node_vec_bx[bucket_idx]) +
				node_vec_ay[bucket_idx].dot(node_vec_by[bucket_idx]) +
				node_vec_az[bucket_idx].dot(node_vec_bz[bucket_idx]);
	});

	return bucket_dot.sum();
}

scalar SceneStepper::dotNodeVectors(const std::vector<VectorXs>& node_vec_ax,
									const std::vector<VectorXs>& node_vec_ay,
									const std::vector<VectorXs>& node_vec_az,
									const std::vector<VectorXs>& node_vec_bx,
									const std::vector<VectorXs>& node_vec_by,
									const std::vector<VectorXs>& node_vec_bz,
									const VectorXs& twist_vec_a,
									const VectorXs& twist_vec_b) const {
	return dotNodeVectors(node_vec_ax, node_vec_ay, node_vec_az, node_vec_bx, node_vec_by, node_vec_bz) +
				twist_vec_a.dot(twist_vec_b);
}

scalar SceneStepper::dotNodeVectors(
		const std::vector<VectorXs>& node_vec_a,
		const std::vector<VectorXs>& node_vec_b) const {
	VectorXs bucket_dot(node_vec_a.size());

	assert(node_vec_a.size() == node_vec_b.size());

	const int num_buckets = node_vec_a.size();

	threadutils::for_each(0, num_buckets, [&](int bucket_idx) {
		bucket_dot[bucket_idx] = node_vec_a[bucket_idx].dot(node_vec_b[bucket_idx]);
	});

	return bucket_dot.sum();
}

scalar SceneStepper::lengthNodeVectors(const std::vector<VectorXs>& node_vec_ax,
										const std::vector<VectorXs>& node_vec_ay,
										const std::vector<VectorXs>& node_vec_az,
										const VectorXs& twist_vec) const {
	VectorXs bucket_length(node_vec_ax.size());

	const int num_buckets = node_vec_ax.size();

	threadutils::for_each(0, num_buckets, [&](int bucket_idx) {
		bucket_length[bucket_idx] = node_vec_ax[bucket_idx].squaredNorm() +
																node_vec_ay[bucket_idx].squaredNorm() +
																node_vec_az[bucket_idx].squaredNorm();
	});

	return sqrt(bucket_length.sum() + twist_vec.squaredNorm());
}

scalar SceneStepper::lengthNodeVectors(
		const std::vector<VectorXs>& node_vec_ax,
		const std::vector<VectorXs>& node_vec_ay,
		const std::vector<VectorXs>& node_vec_az) const {
	VectorXs bucket_length(node_vec_ax.size());

	const int num_buckets = node_vec_ax.size();

	threadutils::for_each(0, num_buckets, [&](int bucket_idx) {
		bucket_length[bucket_idx] = node_vec_ax[bucket_idx].squaredNorm() +
																node_vec_ay[bucket_idx].squaredNorm() +
																node_vec_az[bucket_idx].squaredNorm();
	});

	return sqrt(bucket_length.sum());
}

scalar SceneStepper::lengthNodeVectors(
		const std::vector<VectorXs>& node_vec) const {
	VectorXs bucket_length(node_vec.size());

	const int num_buckets = node_vec.size();

	threadutils::for_each(0, num_buckets, [&](int bucket_idx) {
		bucket_length[bucket_idx] = node_vec[bucket_idx].squaredNorm();
	});

	return sqrt(bucket_length.sum());
}