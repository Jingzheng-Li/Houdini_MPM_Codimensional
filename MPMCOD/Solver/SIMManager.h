

#ifndef SIM_MANAGER_H
#define SIM_MANAGER_H

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <fstream>

#include "Utils/ElasticParameters.h"
#include "Utils/DistanceFields.h"
#include "SolidForce/Force.h"
#include "Utils/Script.h"
#include "Utils/Sorter.h"

class StrandForce;
class AttachForce;

struct SIMInfo {
	scalar viscosity;
	scalar lambda;
	scalar elasto_flip_asym_coeff;
	scalar elasto_flip_coeff;
	scalar elasto_advect_coeff;
	scalar levelset_young_modulus;
	int bending_scheme;
	int iteration_print_step;
	bool solve_solid;
	bool use_pcr;

	friend std::ostream& operator<<(std::ostream&, const SIMInfo&);
};

std::ostream& operator<<(std::ostream&, const SIMInfo&);

struct RayTriInfo {
	int start_geo_id;
	Vector3s end;
	Vector3s norm;
	int intersect_geo_id;
	Vector2s uv;
	scalar dist;
	scalar volume_frac;
	scalar c0;
	scalar c1;
	scalar weight;
};

class SIMManager : public std::enable_shared_from_this<SIMManager> {
	const static int m_kernel_order = 2;
	const static int m_num_armor = 0;

 public:
	SIMManager();
	SIMManager(const SIMManager& otherscene) = delete;
	~SIMManager();

	int getNumParticles() const;
	int getNumEdges() const;
	int getNumFaces() const;
	int getNumSurfels() const;
	int getNumGausses() const;
	int getNumBuckets() const;

	int getNumElasticParameters() const;

	const std::vector<int> getParticleGroup() const;

	const std::vector<unsigned char>& getFixed() const;

	const VectorXs& getX() const;

	VectorXs& getX();

	const VectorXs& getV() const;

	VectorXs& getV();

	const VectorXs& getM() const;

	VectorXs& getM();

	const VectorXs& getVol() const;

	VectorXs& getVol();

	SIMInfo& getSIMInfo();

	const SIMInfo& getSIMInfo() const;

	const std::vector<int>& getParticleEdges(int pidx) const;

	const std::vector<std::pair<int, scalar> >& getParticleFaces(int pidx) const;

	const VectorXs& getRadius() const;

	VectorXs& getRadius();

	const VectorXs& getGaussX() const;

	VectorXs& getGaussX();

	const VectorXs& getGaussV() const;

	VectorXs& getGaussV();

	const MatrixXs& getGaussNormal() const;

	MatrixXs& getGaussNormal();

	const VectorXs& getGaussDV() const;

	VectorXs& getGaussDV();

	const VectorXs& getGaussM() const;

	VectorXs& getGaussM();

	const VectorXs& getGaussVol() const;

	VectorXs& getGaussVol();

	const MatrixXs& getGaussFe() const;

	MatrixXs& getGaussFe();

	const MatrixXs& getGaussd() const;

	MatrixXs& getGaussd();

	const MatrixXs& getGaussDinv() const;

	MatrixXs& getGaussDinv();

	const MatrixXs& getGaussD() const;

	MatrixXs& getGaussD();

	int getDefaultNumNodes() const;

	int getNumNodes(int bucket_idx) const;

	inline bool isFluid(int pidx) const;

	const std::vector<std::shared_ptr<DistanceField> >& getGroupDistanceField() const;

	const VectorXs& getNodePos(int bucket_idx) const;

	VectorXs& getNodePos(int bucket_idx);

	const std::vector<VectorXs>& getNodePos() const;

	std::vector<VectorXs>& getNodePos();

	const std::vector<VectorXs>& getNodeSolidPhi() const;

	std::vector<VectorXs>& getNodeSolidPhi();

	const std::vector<VectorXs>& getNodeOrientationX() const;

	const std::vector<VectorXs>& getNodeOrientationY() const;

	const std::vector<VectorXs>& getNodeOrientationZ() const;

	const std::vector<VectorXs>& getNodeCellSolidPhi() const;

	std::vector<VectorXs>& getNodeCellSolidPhi();

	const std::vector<VectorXs>& getNodeVelocityX() const;

	std::vector<VectorXs>& getNodeVelocityX();

	const std::vector<VectorXs>& getNodeVelocityY() const;

	std::vector<VectorXs>& getNodeVelocityY();

	const std::vector<VectorXs>& getNodeVelocityZ() const;

	std::vector<VectorXs>& getNodeVelocityZ();

	const std::vector<VectorXs>& getNodeMassX() const;

	std::vector<VectorXs>& getNodeMassX();

	const std::vector<VectorXs>& getNodeMassY() const;

	std::vector<VectorXs>& getNodeMassY();

	const std::vector<VectorXs>& getNodeMassZ() const;

	std::vector<VectorXs>& getNodeMassZ();

	const std::vector<VectorXs>& getNodeVolX() const;

	std::vector<VectorXs>& getNodeVolX();

	const std::vector<VectorXs>& getNodeVolY() const;

	std::vector<VectorXs>& getNodeVolY();

	const std::vector<VectorXs>& getNodeVolZ() const;

	std::vector<VectorXs>& getNodeVolZ();

	const std::vector<VectorXs>& getNodeSolidWeightX() const;

	const std::vector<VectorXs>& getNodeSolidWeightY() const;

	const std::vector<VectorXs>& getNodeSolidWeightZ() const;

	const std::vector<VectorXs>& getNodeSolidVelX() const;

	const std::vector<VectorXs>& getNodeSolidVelY() const;

	const std::vector<VectorXs>& getNodeSolidVelZ() const;

	const std::vector<VectorXi>& getNodeIndexEdgeX() const;

	const std::vector<VectorXi>& getNodeIndexEdgeY() const;

	const std::vector<VectorXi>& getNodeIndexEdgeZ() const;

	const Matrix27x2i& getParticleNodesX(int pidx) const;

	Matrix27x2i& getParticleNodesX(int pidx);

	const Matrix27x2i& getGaussNodesX(int pidx) const;

	Matrix27x2i& getGaussNodesX(int pidx);

	const Matrix27x2i& getParticleNodesY(int pidx) const;

	Matrix27x2i& getParticleNodesY(int pidx);

	const Matrix27x2i& getGaussNodesY(int pidx) const;

	Matrix27x2i& getGaussNodesY(int pidx);

	const Matrix27x2i& getParticleNodesZ(int pidx) const;

	Matrix27x2i& getParticleNodesZ(int pidx);

	const Matrix27x2i& getParticleNodesSolidPhi(int pidx) const;

	Matrix27x2i& getParticleNodesSolidPhi(int pidx);

	const Matrix27x2i& getGaussNodesZ(int pidx) const;

	Matrix27x2i& getGaussNodesZ(int pidx);

	const std::vector<Matrix27x4s>& getParticleWeights() const;

	const Matrix27x4s& getParticleWeights(int pidx) const;

	Matrix27x4s& getParticleWeights(int pidx);

	const Matrix27x3s& getGaussWeights(int pidx) const;

	Matrix27x3s& getGaussWeights(int pidx);

	void swapParticles(int i, int j);

	void mapParticleNodesAPIC();  // particles to nodes mapping

	void mapNodeParticlesAPIC();  // nodes to particles mapping

	void resizeParticleSystem(int num_particles);

	void conservativeResizeParticles(int num_particles);

	void conservativeResizeEdges(int num_edges);

	void conservativeResizeFaces(int num_faces);

	void updateGaussSystem(scalar dt);

	void updateGaussManifoldSystem();

	void updateGaussAccel();

	void initGaussSystem();

	VectorXs getPosition(int particle);

	int getDof(int particle) const;

	void setPosition(int particle, const Vector3s& pos);

	void setSIMInfo(const SIMInfo& info);

	void setVelocity(int particle, const Vector3s& vel);

	void setVolume(int particle, const scalar& volume);

	void setGroup(int particle, int group);

	void setMass(int particle, const scalar& mass, const scalar& second_moments);

	void setRadius(int particle, const scalar& radiusA, const scalar& radiusB);

	void setFixed(int particle, unsigned char fixed);

	void setTwist(int particle, bool twist);

	void setEdge(int idx, const std::pair<int, int>& edge);

	void setFace(int idx, const Vector3i& face);

	void setParticleToParameters(int idx, int params);

	scalar getParticleRestLength(int idx) const;

	scalar getParticleRestArea(int idx) const;

	scalar getCellSize() const;

	scalar getInverseDCoeff() const;

	scalar getGaussDensity(int pidx) const;

	scalar getGaussRadius(int pidx, int dir) const;

	scalar getMu(int pidx) const;

	scalar getLa(int pidx) const;

	scalar getViscousModulus(int pidx) const;

	scalar getYoungModulus(int pidx) const;

	scalar getShearModulus(int pidx) const;

	scalar getAttachMultiplier(int pidx) const;

	scalar getCollisionMultiplier(int pidx) const;

	Vector3i getNodeHandle(int node_idx) const;

	int getNodeIndex(const Vector3i& handle) const;

	Vector3s getNodePosSolidPhi(int bucket_idx, int node_idx) const;

	Vector3s getNodePosX(int bucket_idx, int node_idx) const;

	Vector3s getNodePosY(int bucket_idx, int node_idx) const;

	Vector3s getNodePosZ(int bucket_idx, int node_idx) const;

	Vector3s getNodePosP(int bucket_idx, int node_idx) const;

	Vector3s getNodePosEX(int bucket_idx, int node_idx) const;

	Vector3s getNodePosEY(int bucket_idx, int node_idx) const;

	Vector3s getNodePosEZ(int bucket_idx, int node_idx) const;

	void loadAttachForces();

	Vector3s getTwistDir(int particle) const;

	Vector3s getRestTwistDir(int particle) const;

	void setEdgeRestLength(int idx, const scalar& l0);

	void setEdgeToParameter(int idx, int params);

	void setFaceRestArea(int idx, const scalar& a0);

	void setFaceToParameter(int idx, int params);

	const VectorXs& getFaceRestArea() const;

	const VectorXs& getEdgeRestLength() const;

	unsigned char isFixed(int particle) const;

	bool isGaussFixed(int pidx) const;

	bool isTwist(int particle) const;

	const std::vector<bool>& getTwist() const;

	void clearEdges();

	const std::vector<int>& getSurfels() const;

	const MatrixXi& getEdges() const;

	const MatrixXi& getFaces() const;

	void insertElasticParameters(
			const std::shared_ptr<ElasticParameters>& newparams);

	std::shared_ptr<ElasticParameters>& getElasticParameters(const int index);

	const Vector2iT getEdge(int edg) const;

	void insertForce(const std::shared_ptr<Force>& newforce);

	void setTipVerts(int particle, bool tipVerts);

	void accumulateGradU(VectorXs& F, const VectorXs& dx = VectorXs(), const VectorXs& dv = VectorXs());

	void accumulateGaussGradU(MatrixXs& F, const VectorXs& dx = VectorXs(), const VectorXs& dv = VectorXs());

	void precompute();

	void postcompute(VectorXs& v, const scalar& dt);

	void stepScript(const scalar& dt, const scalar& current_time);

	void applyScript(const scalar& dt);

	void updateStartState();

	void accumulateddUdxdx(TripletXs& A, const scalar& dt, int base_idx,
							const VectorXs& dx = VectorXs(), const VectorXs& dv = VectorXs());

	void accumulateAngularddUdxdx(TripletXs& A, const scalar& dt, int base_idx,
								const VectorXs& dx = VectorXs(), const VectorXs& dv = VectorXs());

	void computedEdFe();

	scalar computeKineticEnergy() const;
	scalar computePotentialEnergy() const;
	scalar computeTotalEnergy() const;

	void checkConsistency();

	bool isTip(int particle) const;

	int getNumBucketColors() const;

	int getNumElastoParticles() const;

	int getNumSoftElastoParticles() const;

	Sorter& getParticleBuckets();

	const Sorter& getParticleBuckets() const;

	Sorter& getGaussBuckets();

	const Sorter& getGaussBuckets() const;

	void updateParticleBoundingBox();
	void rebucketizeParticles();
	void resampleNodes();
	void updateParticleWeights(scalar dt, int start, int end);
	void updateGaussWeights(scalar dt);
	void computeWeights(scalar dt);
	void updateSolidWeights();

	void updateRestPos();

	const VectorXs& getRestPos() const;

	VectorXs& getRestPos();

	void initGroupPos();

	void updateDeformationGradient(scalar dt);
	void updatePlasticity(scalar dt);

	void updateTotalMass();

	void setBucketInfo(const scalar& bucket_size, int num_cells, int kernel_order);

	void buildNodeParticlePairs();

	void insertScript(const std::shared_ptr<Script>& script);

	const std::vector<std::shared_ptr<AttachForce> >& getAttachForces() const;

	const std::vector<std::pair<int, int> >& getNodeParticlePairsX(int bucket_idx, int pidx) const;

	const std::vector<std::pair<int, int> >& getNodeGaussPairsX(int bucket_idx, int pidx) const;

	const std::vector<std::pair<int, int> >& getNodeParticlePairsY(int bucket_idx, int pidx) const;

	const std::vector<std::pair<int, int> >& getNodeGaussPairsY(int bucket_idx, int pidx) const;

	const std::vector<std::pair<int, int> >& getNodeParticlePairsZ(int bucket_idx, int pidx) const;

	const std::vector<std::pair<int, int> >& getNodeGaussPairsZ(int bucket_idx, int pidx) const;

	scalar getFrictionAlpha(int pidx) const;

	scalar getFrictionBeta(int pidx) const;

	Eigen::Quaternion<scalar>& getGroupRotation(int group_idx);

	Vector3s& getGroupTranslation(int group_idx);

	Eigen::Quaternion<scalar>& getPrevGroupRotation(int group_idx);
	Vector3s& getPrevGroupTranslation(int group_idx);

	void resizeGroups(int num_group);

	std::shared_ptr<DistanceField>& getGroupDistanceField(int igroup);

	const std::shared_ptr<DistanceField>& getGroupDistanceField(int igroup) const;

	std::vector<std::shared_ptr<DistanceField> >& getDistanceFields();

	const std::vector<std::shared_ptr<DistanceField> >& getDistanceFields() const;

	const VectorXuc& getOutsideInfo() const;

	scalar computePhiVel(const Vector3s& pos, Vector3s& vel,
		const std::function<bool(const std::shared_ptr<DistanceField>&)> selector = nullptr) const;

	scalar computePhi(const Vector3s& pos,
		const std::function<bool(const std::shared_ptr<DistanceField>&)> selector = nullptr) const;

	void dump_geometry(std::string filename);

	int getKernelOrder() const;

	void updateSolidPhi();

	void updateMultipliers(const scalar& dt);

	void solidProjection(const scalar& dt);

	void preAllocateNodes();

	template <typename Callable>
	void findNodes(const Sorter& buckets, const VectorXs& x,
					std::vector<Matrix27x2i>& particle_nodes,
					const Vector3s& offset, Callable func);

	void generateNodes();

	void connectSolidPhiNodes();

	void connectEdgeNodes();

	void postAllocateNodes();

	scalar getMaxVelocity() const;

	const Vector3s& getBucketMinCorner() const;

	scalar getBucketLength() const;

	int getNumColors() const;

	const std::vector<int>& getParticleToSurfels() const;

	const std::vector<std::vector<RayTriInfo> >& getIntersections() const;

	const std::vector<Vector3s>& getFaceWeights() const;

	scalar interpolateValue(const Vector3s& pos, const std::vector<VectorXs>& phi,
							const Vector3s& phi_ori, const scalar& default_val);

	inline Vector3s nodePosFromBucket(int bucket_idx, int raw_node_idx, const Vector3s& offset) const;

	void markInsideOut();

	bool isSoft(int pidx) const;

	void updateVelocityDifference();

	void saveParticleVelocity();

	void computeDDA();

	void updateStrandParamViscosity(const scalar& dt);

	bool isBucketActivated(int bucket_index) const;

	const std::vector<unsigned char>& getBucketActivated() const;

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW


 private:
	int step_count;
	VectorXs m_x;       // particle pos
	VectorXs m_rest_x;  // particle rest pos

	VectorXs m_v;  // particle velocity
	VectorXs m_saved_v;
	VectorXs m_dv;
	VectorXs m_m;        // particle mass
	VectorXs m_radius;  // particle radius
	VectorXs m_vol;     // particle volume
	VectorXs m_rest_vol;
	VectorXuc m_inside;

	VectorXs m_particle_rest_length;
	VectorXs m_edge_rest_length;
	VectorXs m_particle_rest_area;
	VectorXs m_face_rest_area;
	MatrixXs m_B;  // particle B matrix

	VectorXs m_x_gauss;
	VectorXs m_v_gauss;
	VectorXs m_dv_gauss;
	VectorXs m_m_gauss;
	VectorXs m_vol_gauss;
	VectorXs m_rest_vol_gauss;
	VectorXs m_radius_gauss;

	std::vector<std::vector<RayTriInfo> > m_ray_tri_gauss;

	MatrixXs m_Fe_gauss;     // elastic deformation gradient
	MatrixXs m_d_gauss;      // material axis
	MatrixXs m_d_old_gauss;  // store last deformation information
	MatrixXs m_D_gauss;      // original material axis
	MatrixXs m_D_inv_gauss;  // inverse of original material axis
	MatrixXs m_dFe_gauss;    // dphidFe
	MatrixXs m_norm_gauss;  // normalized material axis with rigid transformation

	std::vector<unsigned char> m_fixed;
	std::vector<bool> m_twist;
	MatrixXi m_edges;
	MatrixXi m_faces;  // store the face id
	std::vector<Vector3s> m_face_weights;

	std::vector<Vector3i> m_face_inv_mapping;  // location of the face itself in particle_to_face array.
	std::vector<Vector2i> m_edge_inv_mapping;  // location of the edge itself in particle_to_face array.

	std::vector<int> m_surfels;
	std::vector<Vector3s> m_surfel_norms;
	std::vector<std::vector<std::pair<int, scalar>>> m_particle_to_face;
	std::vector<std::vector<int>> m_particle_to_edge;
	std::vector<int> m_particle_to_surfel;
	
	std::vector<int> m_gauss_to_parameters;
	std::vector<int> m_face_to_parameters;
	std::vector<int> m_edge_to_parameters;

	std::vector<Matrix27x2i> m_particle_nodes_x;
	std::vector<Matrix27x2i> m_particle_nodes_y;
	std::vector<Matrix27x2i> m_particle_nodes_z;
	std::vector<Matrix27x2i> m_particle_nodes_solid_phi;
	std::vector<Matrix27x2i> m_particle_nodes_p;

	std::vector<Matrix27x2i> m_gauss_nodes_x;
	std::vector<Matrix27x2i> m_gauss_nodes_y;
	std::vector<Matrix27x2i> m_gauss_nodes_z;
	std::vector<Matrix27x2i> m_gauss_nodes_p;

	std::vector<Vector27s> m_particle_weights_p;
	std::vector<Matrix27x4s> m_particle_weights;

	std::vector<Matrix27x3s> m_gauss_weights;

	// bucket id -> nodes -> pairs of (particle id, id in particle neighborhoods)
	std::vector<std::vector<std::vector<std::pair<int, int>>>> m_node_particles_x;
	std::vector<std::vector<std::vector<std::pair<int, int>>>> m_node_particles_y;
	std::vector<std::vector<std::vector<std::pair<int, int>>>> m_node_particles_z;
	std::vector<std::vector<std::vector<std::pair<int, int>>>> m_node_particles_p;

	std::vector<int> m_particle_group;

	Vector3s m_bbx_min;
	Vector3s m_bbx_max;

	scalar m_bucket_size;
	int m_num_colors;
	int m_num_nodes;
	int m_num_bucket_colors;
	Vector3s m_bucket_mincorner;
	Vector3s m_grid_mincorner;

	Sorter m_particle_buckets;
	Sorter m_gauss_buckets;
	Sorter m_particle_cells;

	std::vector<unsigned char> m_bucket_activated;

	std::vector<VectorXs> m_node_pos;

	std::vector<VectorXi> m_node_index_solid_phi_x;  // bucket id -> 4x2 neighbors to the solid phi nodes
	std::vector<VectorXi> m_node_index_solid_phi_y;  // bucket id -> 4x2 neighbors to the solid phi nodes
	std::vector<VectorXi> m_node_index_solid_phi_z;  // bucket id -> 4x2 neighbors to the solid phi nodes

	std::vector<VectorXi> m_node_index_edge_x;  // bucket id -> 4x2 neighbors to the edge nodes (ey, ez)
	std::vector<VectorXi> m_node_index_edge_y;  // bucket id -> 4x2 neighbors to the edge nodes (ex, ez)
	std::vector<VectorXi> m_node_index_edge_z;  // bucket id -> 4x2 neighbors to the edge nodes (ex, ey)

	std::vector<VectorXs> m_node_solid_phi;
	std::vector<VectorXs> m_node_combined_phi;
	std::vector<VectorXs> m_node_cell_solid_phi;

	std::vector<VectorXs> m_node_solid_vel_x;
	std::vector<VectorXs> m_node_solid_vel_y;
	std::vector<VectorXs> m_node_solid_vel_z;

	std::vector<VectorXs> m_node_solid_weight_x;  // bucket id -> node solid weight
	std::vector<VectorXs> m_node_solid_weight_y;
	std::vector<VectorXs> m_node_solid_weight_z;

	std::vector<VectorXs> m_node_vel_x;  // bucket id -> nodes velocity
	std::vector<VectorXs> m_node_vel_y;
	std::vector<VectorXs> m_node_vel_z;

	std::vector<VectorXs> m_node_mass_x;  // bucket id -> nodes mass
	std::vector<VectorXs> m_node_mass_y;
	std::vector<VectorXs> m_node_mass_z;

	std::vector<VectorXs> m_node_vol_x;  // bucket id -> nodes volume
	std::vector<VectorXs> m_node_vol_y;
	std::vector<VectorXs> m_node_vol_z;

	std::vector<std::shared_ptr<StrandForce> > m_strands;
	std::vector<bool> m_is_strand_tip;

	std::vector<MatrixXi> m_gauss_bucket_neighbors;

	SIMInfo m_siminfo;

	std::vector<std::shared_ptr<Force> > m_forces;

	std::vector<std::shared_ptr<AttachForce> > m_attach_forces;

	std::vector<std::shared_ptr<ElasticParameters> > m_strandParameters;

	std::vector<Vector3s> m_group_pos;
	std::vector<Eigen::Quaternion<scalar> > m_group_rot;
	std::vector<Vector3s> m_group_prev_pos;
	std::vector<Eigen::Quaternion<scalar> > m_group_prev_rot;

	std::vector<std::shared_ptr<Script> > m_scripts;

	std::vector<scalar> m_shooting_vol_accum;

	std::vector<std::shared_ptr<DistanceField> > m_group_distance_field;

	std::vector<std::shared_ptr<DistanceField> > m_distance_fields;

};

#endif
