

#ifndef __THIN_SHELL_FORCE_H__
#define __THIN_SHELL_FORCE_H__

#include <Eigen/Core>

#include "SolidForce/Force.h"
#include "Utils/MathDefs.h"
#include "Solver/SIMManager.h"
#include "Forces/ShellBendingForce.h"
#include "Forces/ShellMembraneForce.h"

class ThinShellForce : public Force {
 protected:
	std::shared_ptr<SIMManager> m_scene;
	MatrixXi m_F;

	std::vector<std::vector<int>> m_node_neighbors;
	std::vector<std::vector<int>> m_per_node_triangles;

	// node local index (0, 1 or 2) for each node's incident triangle
	std::vector<std::vector<int>> m_per_node_triangles_node_local_index;

	// #F by #3 adjacent matrix, the element i,j is the id of the triangle
	// adjacent to the j edge of triangle i
	MatrixXi m_per_triangle_triangles;

	// #F by #3 adjacent matrix, the element i,j is the id of edge of the triangle
	// m_per_triangle_triangles(i,j) that is adjacent with triangle i. NOTE: the
	// first edge of a triangle is [0,1] the second [1,2] and the third [2,3].
	MatrixXi m_per_triangle_triangles_edge_local_index;

	// #F*3 by 2 list of all of directed edges
	MatrixXi m_E_directed;

	// #uE by 2 list of unique undirected edges
	MatrixXi m_E_unique;

	// #F*3 list of indices into uE, mapping each directed edge to unique
	// undirected edge
	MatrixXi m_map_edge_directed_to_unique;

	// #uE list of lists of indices into E of coexisting edges
	std::vector<std::vector<int>> m_map_edge_unique_to_directed;

	// #E by 2 list of edge flaps, EF(e,0)=f means e=(i-->j) is the edge of F(f,:)
	// opposite the vth corner, where EI(e,0)=v. Similarly EF(e,1) e=(j->i)
	MatrixXi m_per_unique_edge_triangles;

	// #E by 2 list of edge flap corners
	MatrixXi m_per_unique_edge_triangles_local_corners;

	std::vector<std::vector<int>> m_per_node_edges;

	MatrixXi m_per_triangles_unique_edges;

	VectorXs m_triangle_rest_areas;

	std::vector<std::shared_ptr<Force>> m_forces;

 public:
	ThinShellForce(const std::shared_ptr<SIMManager>& scene,
								 const std::vector<Vector3i>& faces, const int& parameterIndex,
								 int globalIndex);

	virtual ~ThinShellForce();

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

	virtual int numHessX();

	virtual void preCompute();

	virtual void updateStartState();

	virtual Force* createNewCopy();

	virtual int flag() const;

	virtual bool parallelized() const;
};

#endif
