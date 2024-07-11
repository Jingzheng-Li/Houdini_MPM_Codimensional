
#ifndef LINEARIZED_IMPLICIT_EULER
#define LINEARIZED_IMPLICIT_EULER

#include <Eigen/Core>
#include <iostream>

#include "Utils/MathUtilities.h"
#include "SceneStepper.h"
#include "Utils/StringUtilities.h"
class LinearizedImplicitEuler : public SceneStepper {
 public:
	LinearizedImplicitEuler(const scalar& criterion, int maxiters);

	virtual ~LinearizedImplicitEuler();

	virtual bool stepVelocity(SIMManager& scene, scalar dt);

	virtual bool acceptVelocity(SIMManager& scene);

	virtual bool stepImplicitElasto(SIMManager& scene, scalar dt);

	virtual bool stepImplicitElastoDiagonalPCR(SIMManager& scene, scalar dt);

	virtual void pushElastoVelocity();

	virtual void popElastoVelocity();

	virtual std::string getName() const;

 private:
	void zeroFixedDoFs(const SIMManager& scene, VectorXs& vec);

	void performLocalSolve(const SIMManager& scene,
						const std::vector<VectorXs>& node_rhs_x,
						const std::vector<VectorXs>& node_rhs_y,
						const std::vector<VectorXs>& node_rhs_z,
						const std::vector<VectorXs>& node_mass_x,
						const std::vector<VectorXs>& node_mass_y,
						const std::vector<VectorXs>& node_mass_z,
						std::vector<VectorXs>& out_node_vec_x,
						std::vector<VectorXs>& out_node_vec_y,
						std::vector<VectorXs>& out_node_vec_z);

	// void prepareGroupPrecondition(const SIMManager& scene,
	// 							const std::vector<VectorXs>& node_m_x,
	// 							const std::vector<VectorXs>& node_m_y,
	// 							const std::vector<VectorXs>& node_m_z,
	// 							const scalar& dt);

	void performLocalSolveTwist(const SIMManager& scene, const VectorXs& rhs,
															const VectorXs& m, VectorXs& out);

	void performLocalSolve(const SIMManager& scene, const VectorXs& rhs,
												 const VectorXs& m, VectorXs& out);

	void performInvLocalSolve(const SIMManager& scene,
							const std::vector<VectorXs>& node_rhs_x,
							const std::vector<VectorXs>& node_rhs_y,
							const std::vector<VectorXs>& node_rhs_z,
							const std::vector<VectorXs>& node_inv_mass_x,
							const std::vector<VectorXs>& node_inv_mass_y,
							const std::vector<VectorXs>& node_inv_mass_z,
							std::vector<VectorXs>& out_node_vec_x,
							std::vector<VectorXs>& out_node_vec_y,
							std::vector<VectorXs>& out_node_vec_z);

	void performGlobalMultiply(const SIMManager& scene, const scalar& dt,
							const std::vector<VectorXs>& node_m_x,
							const std::vector<VectorXs>& node_m_y,
							const std::vector<VectorXs>& node_m_z,
							const std::vector<VectorXs>& node_v_x,
							const std::vector<VectorXs>& node_v_y,
							const std::vector<VectorXs>& node_v_z,
							std::vector<VectorXs>& out_node_vec_x,
							std::vector<VectorXs>& out_node_vec_y,
							std::vector<VectorXs>& out_node_vec_z);

	void performGlobalMultiply(const SIMManager& scene, const scalar& dt,
								const std::vector<VectorXs>& node_m_x,
								const std::vector<VectorXs>& node_m_y,
								const std::vector<VectorXs>& node_m_z,
								const std::vector<VectorXs>& node_v_x,
								const std::vector<VectorXs>& node_v_y,
								const std::vector<VectorXs>& node_v_z,
								std::vector<VectorXs>& out_node_vec_x,
								std::vector<VectorXs>& out_node_vec_y,
								std::vector<VectorXs>& out_node_vec_z,
								const VectorXs& m, const VectorXs& angular_vec,
								VectorXs& out);

	void performGlobalMultiply(const SIMManager& scene, const scalar& dt,
														 const VectorXs& m, const VectorXs& vec,
														 VectorXs& out);

	void performAngularGlobalMultiply(const SIMManager& scene, const scalar& dt,
									const VectorXs& m, const VectorXs& v, VectorXs& out);

	void constructNodeForceCoarse(SIMManager& scene, const scalar& dt);

	void performInvLocalSolveCoarse(
			SIMManager& scene, const Array3s& node_rhs_x, const Array3s& node_rhs_y,
			const Array3s& node_rhs_z, const Array3s& node_inv_mass_x,
			const Array3s& node_inv_mass_y, const Array3s& node_inv_mass_z,
			Array3s& out_node_vec_x, Array3s& out_node_vec_y,
			Array3s& out_node_vec_z);

	void constructHDVCoarse(SIMManager& scene, const scalar& dt,
							Array3s& node_hdv_x, Array3s& node_hdv_y,
							Array3s& node_hdv_z, Array3s& node_hdvs_x,
							Array3s& node_hdvs_y, Array3s& node_hdvs_z,
							const Array3s& node_v_x, const Array3s& node_v_y,
							const Array3s& node_v_z);

	void constructInvMDVCoarse(SIMManager& scene, Array3s& node_inv_mdv_x,
								Array3s& node_inv_mdv_y, Array3s& node_inv_mdv_z,
								const Array3s& node_hdv_x,
								const Array3s& node_hdv_y,
								const Array3s& node_hdv_z);

	void constructMsDVsCoarse(SIMManager& scene, Array3s& node_msdv2_x,
							Array3s& node_msdv2_y, Array3s& node_msdv2_z,
							Array3s& node_inv_msdv2_x,
							Array3s& node_inv_msdv2_y,
							Array3s& node_inv_msdv2_z,
							const Array3s& node_hdvs_x,
							const Array3s& node_hdvs_y,
							const Array3s& node_hdvs_z);

	void constructPsiFSCoarse(SIMManager& scene, Array3s& node_psi_fs_x,
								Array3s& node_psi_fs_y, Array3s& node_psi_fs_z,
								const Array3s& node_inv_mdvs_x,
								const Array3s& node_inv_mdvs_y,
								const Array3s& node_inv_mdvs_z,
								const Array3s& node_hdvs_x,
								const Array3s& node_hdvs_y,
								const Array3s& node_hdvs_z);

	void constructNodeForce(SIMManager& scene, const scalar& dt,
							std::vector<VectorXs>& node_rhs_x,
							std::vector<VectorXs>& node_rhs_y,
							std::vector<VectorXs>& node_rhs_z);

	void constructInvMDV(SIMManager& scene);

	void constructMsDVs(SIMManager& scene);

	void constructHessianPreProcess(SIMManager& scene, const scalar& dt);

	void constructHessianPostProcess(SIMManager& scene, const scalar& dt);

	void constructAngularHessianPreProcess(SIMManager& scene, const scalar& dt);

	void constructAngularHessianPostProcess(SIMManager& scene, const scalar& dt);

	//    SparseXs m_A;
	std::vector<std::pair<int, int> > m_triA_sup;
	std::vector<std::pair<int, int> > m_angular_triA_sup;
	TripletXs m_triA;
	TripletXs m_angular_triA;
	VectorXs m_multiply_buffer;
	VectorXs m_pre_mult_buffer;

	VectorXs m_angular_moment_buffer;
	VectorXs m_angular_v_plus_buffer;

	const scalar m_pcg_criterion;
	const int m_maxiters;

	std::vector<VectorXs> m_node_rhs_x;
	std::vector<VectorXs> m_node_rhs_y;
	std::vector<VectorXs> m_node_rhs_z;
	std::vector<VectorXs> m_node_v_plus_x;
	std::vector<VectorXs> m_node_v_plus_y;
	std::vector<VectorXs> m_node_v_plus_z;
	std::vector<VectorXs> m_node_v_0_x;
	std::vector<VectorXs> m_node_v_0_y;
	std::vector<VectorXs> m_node_v_0_z;
	std::vector<VectorXs> m_node_v_tmp_x;
	std::vector<VectorXs> m_node_v_tmp_y;
	std::vector<VectorXs> m_node_v_tmp_z;
	std::vector<VectorXs> m_node_r_x;  // r0
	std::vector<VectorXs> m_node_r_y;
	std::vector<VectorXs> m_node_r_z;
	std::vector<VectorXs> m_node_z_x;  // s
	std::vector<VectorXs> m_node_z_y;
	std::vector<VectorXs> m_node_z_z;
	std::vector<VectorXs> m_node_p_x;  // p
	std::vector<VectorXs> m_node_p_y;
	std::vector<VectorXs> m_node_p_z;
	std::vector<VectorXs> m_node_q_x;  // h
	std::vector<VectorXs> m_node_q_y;
	std::vector<VectorXs> m_node_q_z;
	std::vector<VectorXs> m_node_w_x;  // v
	std::vector<VectorXs> m_node_w_y;
	std::vector<VectorXs> m_node_w_z;
	std::vector<VectorXs> m_node_t_x;  // t
	std::vector<VectorXs> m_node_t_y;
	std::vector<VectorXs> m_node_t_z;

	VectorXs m_angular_r;
	VectorXs m_angular_z;
	VectorXs m_angular_p;
	VectorXs m_angular_q;
	VectorXs m_angular_w;
	VectorXs m_angular_t;

	std::vector<VectorXs> m_node_damped_x;  // damped M_s
	std::vector<VectorXs> m_node_damped_y;
	std::vector<VectorXs> m_node_damped_z;

	std::vector<VectorXs> m_node_mshdvm_hdvm_x;  // (M_s+hDVm)^{-1}hDVm
	std::vector<VectorXs> m_node_mshdvm_hdvm_y;
	std::vector<VectorXs> m_node_mshdvm_hdvm_z;

	std::vector<VectorXs> m_node_inv_C_x;  // (M_f+[hdv]*M_s)^{-1}
	std::vector<VectorXs> m_node_inv_C_y;
	std::vector<VectorXs> m_node_inv_C_z;

	std::vector<VectorXs> m_node_Cs_x;  // M_s+[hdvs]*M_f
	std::vector<VectorXs> m_node_Cs_y;
	std::vector<VectorXs> m_node_Cs_z;

	std::vector<VectorXs> m_node_inv_Cs_x;  // (M_s+[hdvs]*M_f)^{-1}
	std::vector<VectorXs> m_node_inv_Cs_y;
	std::vector<VectorXs> m_node_inv_Cs_z;

	std::stack<std::vector<VectorXs> > m_elasto_vel_stack;

	std::vector<std::shared_ptr<Eigen::SimplicialLDLT<SparseXs> > > m_group_preconditioners;

};

#endif
