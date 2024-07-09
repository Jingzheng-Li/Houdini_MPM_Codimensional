
#ifndef SCENE_STEPPER
#define SCENE_STEPPER

#include <functional>
#include <stack>

#include "Utils/MathDefs.h"
#include "SIMManager.h"

class SceneStepper {
 public:
	virtual ~SceneStepper();

	virtual bool stepVelocity(SIMManager& scene, scalar dt) = 0;

	virtual bool acceptVelocity(SIMManager& scene) = 0;

	virtual bool stepImplicitElasto(SIMManager& scene, scalar dt) = 0;

	virtual void pushElastoVelocity() = 0;

	virtual void popElastoVelocity() = 0;

	virtual bool advectScene(SIMManager& scene, scalar dt);

	virtual std::string getName() const = 0;

	virtual void setUseApic(bool apic);

	virtual bool useApic() const;

	// tools function
	void mapNodeToSoftParticles(const SIMManager& scene,
								const std::vector<VectorXs>& node_vec_x,
								const std::vector<VectorXs>& node_vec_y,
								const std::vector<VectorXs>& node_vec_z,
								VectorXs& part_vec) const;

	void mapSoftParticlesToNode(const SIMManager& scene,
															std::vector<VectorXs>& node_vec_x,
															std::vector<VectorXs>& node_vec_y,
															std::vector<VectorXs>& node_vec_z,
															const VectorXs& part_vec) const;

	void allocateNodeVectors(const SIMManager& scene,
													 std::vector<VectorXs>& node_vec_x,
													 std::vector<VectorXs>& node_vec_y,
													 std::vector<VectorXs>& node_vec_z) const;

	void allocateNodeVectors(const SIMManager& scene,
													 std::vector<VectorXi>& node_vec_x,
													 std::vector<VectorXi>& node_vec_y,
													 std::vector<VectorXi>& node_vec_z) const;

	scalar dotNodeVectors(const std::vector<VectorXs>& node_vec_ax,
												const std::vector<VectorXs>& node_vec_ay,
												const std::vector<VectorXs>& node_vec_az,
												const std::vector<VectorXs>& node_vec_bx,
												const std::vector<VectorXs>& node_vec_by,
												const std::vector<VectorXs>& node_vec_bz) const;

	scalar dotNodeVectors(const std::vector<VectorXs>& node_vec_ax,
												const std::vector<VectorXs>& node_vec_ay,
												const std::vector<VectorXs>& node_vec_az,
												const std::vector<VectorXs>& node_vec_bx,
												const std::vector<VectorXs>& node_vec_by,
												const std::vector<VectorXs>& node_vec_bz,
												const VectorXs& twist_vec_a,
												const VectorXs& twist_vec_b) const;

	scalar dotNodeVectors(const std::vector<VectorXs>& node_vec_a,
												const std::vector<VectorXs>& node_vec_b) const;

	scalar lengthNodeVectors(const std::vector<VectorXs>& node_vec_ax,
													 const std::vector<VectorXs>& node_vec_ay,
													 const std::vector<VectorXs>& node_vec_az,
													 const VectorXs& twist_vec) const;

	scalar lengthNodeVectors(const std::vector<VectorXs>& node_vec_ax,
													 const std::vector<VectorXs>& node_vec_ay,
													 const std::vector<VectorXs>& node_vec_az) const;

	scalar lengthNodeVectors(const std::vector<VectorXs>& node_vec) const;

	void mapGaussToNode(const SIMManager& scene, std::vector<VectorXs>& node_vec_x,
											std::vector<VectorXs>& node_vec_y,
											std::vector<VectorXs>& node_vec_z,
											const MatrixXs& gauss_vec) const;

 protected:

	bool m_apic;
};

#endif
