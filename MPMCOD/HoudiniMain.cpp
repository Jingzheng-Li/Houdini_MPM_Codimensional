


#include "HoudiniMain.h"
#include <unordered_set>


// 1 指针好像有点问题 几个智能指针
// 2. group怎么使用的
// TODO: get vdb

namespace MPMCOD {
    std::shared_ptr<SIMManager> m_manager;
    std::shared_ptr<SIMCore> m_core;
	std::shared_ptr<SceneStepper> m_stepper;
	std::shared_ptr<SIMInfo> m_siminfo;
}

const SIM_DopDescription* GAS_MPM_CODIMENSIONAL::getDopDescription() {
	static PRM_Template templateList[] = {
		PRM_Template()
	};

	static SIM_DopDescription dopDescription(
		true,
		"GAS_MPM_CODIMENSIONAL", // internal name of the dop
		"Gas MPM CODIMENSIONAL", // label of the dop
		"GASMPMCODIMENSIONAL", // template list for generating the dop
		classname(),
		templateList);

	setGasDescription(dopDescription);

	return &dopDescription;
}

GAS_MPM_CODIMENSIONAL::GAS_MPM_CODIMENSIONAL(const SIM_DataFactory* factory) : BaseClass(factory) {}

GAS_MPM_CODIMENSIONAL::~GAS_MPM_CODIMENSIONAL() {
	MPMCOD::m_manager.reset();
	MPMCOD::m_core.reset();
	MPMCOD::m_stepper.reset();
	MPMCOD::m_siminfo.reset();
}

bool GAS_MPM_CODIMENSIONAL::solveGasSubclass(SIM_Engine& engine,
											SIM_Object* object,
											SIM_Time time,
											SIM_Time timestep) {

	//////////////////////////////////
	// read geometry from Houdini
	//////////////////////////////////
	const SIM_Geometry *geo = object->getGeometry();
	CHECK_ERROR_SOLVER(geo != NULL, "Failed to get readBuffer geometry object")
	GU_DetailHandleAutoReadLock readlock(geo->getGeometry());
	CHECK_ERROR_SOLVER(readlock.isValid(), "Failed to get readBuffer geometry detail");
	const GU_Detail *gdp = readlock.getGdp();
	CHECK_ERROR_SOLVER(!gdp->isEmpty(), "readBuffer Geometry is empty");

	if (!MPMCOD::m_manager) {

		omp_set_num_threads(1); // set openmp number

		MPMCOD::m_manager = std::make_shared<SIMManager>();		

		transferPointAttribTOEigen(geo, gdp);
		loadSIMInfos();

		MPMCOD::m_manager->resizeGroups(1); // TODO: change groups meaning
		MPMCOD::m_manager->updateRestPos();
		MPMCOD::m_manager->initGroupPos();

		transferFaceAttribTOEigen(geo, gdp);

		MPMCOD::m_manager->initGaussSystem();
		MPMCOD::m_manager->updateParticleBoundingBox();
		MPMCOD::m_manager->rebucketizeParticles();
		MPMCOD::m_manager->resampleNodes();
		MPMCOD::m_manager->computeWeights(0.0);
		MPMCOD::m_manager->updatePlasticity(0.0);
		MPMCOD::m_manager->computedEdFe();
		
		MPMCOD::m_manager->updateSolidPhi(); // TODO: change inside
		MPMCOD::m_manager->updateSolidWeights(); // TODO: may be useless(this is for fluidsolid)
		MPMCOD::m_manager->mapParticleNodesAPIC();
		MPMCOD::m_manager->saveParticleVelocity();

		MPMCOD::m_manager->loadAttachForces();
		MPMCOD::m_manager->insertForce(std::make_shared<JunctionForce>(MPMCOD::m_manager));
		MPMCOD::m_manager->insertForce(std::make_shared<LevelSetForce>(MPMCOD::m_manager, MPMCOD::m_manager->getCellSize() * 0.25));

		loadGravity(object, gdp);

	}

	

	///////////////////////////////////////////
	// MPM simulation
	///////////////////////////////////////////
	CHECK_ERROR_SOLVER(MPMCOD::m_manager, "not initialize m_manager");
	CHECK_ERROR_SOLVER(MPMCOD::m_stepper, "not initialize m_stepper");
	std::cout << "info~~~~~~~~~" << MPMCOD::m_siminfo->elasto_flip_coeff << std::endl;
	if (!MPMCOD::m_core)
		MPMCOD::m_core = std::make_shared<SIMCore>(MPMCOD::m_manager, MPMCOD::m_stepper);
	scalar dt = 0.001;
	MPMCOD::m_core->stepSystem(dt);










	///////////////////////////////////////////
	// write geometry back to Houdini
	///////////////////////////////////////////
	SIM_GeometryCopy *newgeo = SIM_DATA_CREATE(*object, "Geometry", SIM_GeometryCopy, SIM_DATA_RETURN_EXISTING | SIM_DATA_ADOPT_EXISTING_ON_DELETE);
	CHECK_ERROR_SOLVER(newgeo != NULL, "Failed to create writeBuffer GeometryCopy object");
	GU_DetailHandleAutoWriteLock writelock(newgeo->getOwnGeometry());
	CHECK_ERROR_SOLVER(writelock.isValid(), "Failed to get writeBuffer geometry detail");
	GU_Detail *newgdp = writelock.getGdp();
	CHECK_ERROR_SOLVER(!newgdp->isEmpty(), "writeBuffer Geometry is empty");

	transferPointAttribTOHoudini(newgeo, newgdp);

	return true;
}


void GAS_MPM_CODIMENSIONAL::loadSIMInfos() {

	// load bucket parameters
	scalar bucket_size = 4.0;
	int num_cells = 4;
	int kernel_order = 2;
	MPMCOD::m_manager->setBucketInfo(bucket_size, num_cells, kernel_order);

	scalar criterion = 1e-6;
	int maxiters = 100;
	MPMCOD::m_stepper = std::make_shared<LinearizedImplicitEuler>(criterion, maxiters);

	// load material parameters
	scalar dt = 0.001;
	scalar yarnradius = 0.0165;
	scalar YoungsModulus = 6.6e5;
	scalar poissonRatio = 0.35;
	scalar shearModulus = YoungsModulus / ((1.0 + poissonRatio) * 2.0);
	scalar density = 1.3; // g/cm^3
	scalar viscosity = 1e3;
	scalar stretchingMultiplier = 1.0;
	scalar collisionMultiplier = 1.0;
	scalar attachMultiplier = 0.1;
	scalar baseRotation = 0.0;
	bool accumulateWithViscous = false;
	bool accumulateViscousOnlyForBendingModes = true;
	bool postProjectFixed = false;
	bool useApproxJacobian = false;
	bool useTournierJacobian = true;
	scalar straightHairs = 1.;
	Vec3 haircolor = Vec3(0, 0, 0);
	scalar friction_angle = 0;
	friction_angle = std::max(0., std::min(90.0, friction_angle)) / 180.0 * M_PI;
	const scalar friction_alpha = 1.6329931619 * sin(friction_angle) / (3.0 - sin(friction_angle));
	const scalar friction_beta = tan(friction_angle);
	VecX rad_vec(2);
	rad_vec(0) = yarnradius;
	rad_vec(1) = yarnradius;
	MPMCOD::m_manager->insertElasticParameters(
	std::make_shared<ElasticParameters>(
		rad_vec, YoungsModulus, shearModulus, stretchingMultiplier,
		collisionMultiplier, attachMultiplier, density, viscosity, baseRotation,
		dt, friction_alpha, friction_beta,
		accumulateWithViscous, accumulateViscousOnlyForBendingModes,
		postProjectFixed, useApproxJacobian, useTournierJacobian, straightHairs,
		haircolor));


	// load sim parameters
	MPMCOD::m_siminfo = std::make_shared<SIMInfo>();
	
	MPMCOD::m_siminfo->viscosity = 8.9e-3;
	MPMCOD::m_siminfo->lambda = 1.0;
	MPMCOD::m_siminfo->elasto_flip_asym_coeff = 1.0;
	MPMCOD::m_siminfo->elasto_flip_coeff = 0.95;
	MPMCOD::m_siminfo->elasto_advect_coeff = 1.0;
	MPMCOD::m_siminfo->bending_scheme = 2;
	MPMCOD::m_siminfo->levelset_young_modulus = 6.6e6;
	MPMCOD::m_siminfo->iteration_print_step = 0;

}

/////////////////////////////////////////
// tranfer Data from Houdini to Eigen
/////////////////////////////////////////
void GAS_MPM_CODIMENSIONAL::transferPointAttribTOEigen(const SIM_Geometry *geo, const GU_Detail *gdp) {

	int numParticles = gdp->getNumPoints();
	CHECK_ERROR(numParticles > 0, "No particles found in geometry");
	Eigen::MatrixXd positions(numParticles, 3);
	Eigen::MatrixXd velocities(numParticles, 3);
	Eigen::VectorXd masses(numParticles);
	Eigen::VectorXd radius(numParticles);

	GA_ROHandleV3D velHandle(gdp, GA_ATTRIB_POINT, "v");
	GA_ROHandleD massHandle(gdp, GA_ATTRIB_POINT, "mass");
	GA_ROHandleD radiusHandle(gdp, GA_ATTRIB_POINT, "pscale");
	CHECK_ERROR(velHandle.isValid(), "Failed to get velocity attributes");
	CHECK_ERROR(massHandle.isValid(), "Failed to get mass attributes");
	CHECK_ERROR(radiusHandle.isValid(), "Failed to get radius attributes");

	GA_Offset ptoff;
	int idx = 0;
	GA_FOR_ALL_PTOFF(gdp, ptoff) {
		if (idx >= numParticles) break;
		UT_Vector3D pos3 = gdp->getPos3D(ptoff);
		UT_Vector3D vel3 = velHandle.get(ptoff);
		positions.row(idx) << pos3.x() * 100.0, pos3.y() * 100.0, pos3.z() * 100.0;
		velocities.row(idx) << vel3.x() * 100.0, vel3.y() * 100.0, vel3.z() * 100.0;
		masses(idx) = massHandle.get(ptoff) * 1000.0;
		radius(idx) = radiusHandle.get(ptoff) * 100.0;
		idx++; // Increment the index for the next row
	}
	CHECK_ERROR(idx == gdp->getNumPoints(), "Failed to get all points");

	MPMCOD::m_manager->resizeParticleSystem(numParticles);
	for (int i = 0; i < numParticles; ++i) {
		MPMCOD::m_manager->setPosition(i, positions.row(i));
		MPMCOD::m_manager->setVelocity(i, velocities.row(i));
		MPMCOD::m_manager->setMass(i, masses(i), 0.0);
		MPMCOD::m_manager->setRadius(i, radius(i), radius(i));
	}

}

void GAS_MPM_CODIMENSIONAL::transferFaceAttribTOEigen(const SIM_Geometry *geo, const GU_Detail *gdp) {

	int numfaces = gdp->getNumPrimitives();
	std::vector<Vector3i> faces(numfaces);

	int faceIndex = 0;
	GA_Offset primoff;
	GA_FOR_ALL_PRIMOFF(gdp, primoff) {
		const GA_Primitive* prim = gdp->getPrimitive(primoff);
		if (prim->getVertexCount() == 3) {
			Vector3i face;
			for (int i = 0; i < 3; ++i) {
				GA_Offset vtxoff = prim->getVertexOffset(i);
				GA_Offset ptoff = gdp->vertexPoint(vtxoff);
				face(i) = static_cast<int>(gdp->pointIndex(ptoff));
			}
			faces[faceIndex++] = face;
		}
	}
	CHECK_ERROR(faceIndex == numfaces, "Failed to get all faces");

	int paramsIndex = 0;
	std::unordered_set<int> unique_particles;
	MPMCOD::m_manager->conservativeResizeFaces(numfaces);

	for (int i = 0; i < numfaces; ++i) {
		MPMCOD::m_manager->setFace(i, faces[i]);
		Vector3s dx0 = MPMCOD::m_manager->getPosition(faces[i](1)) - MPMCOD::m_manager->getPosition(faces[i](0));
		Vector3s dx1 = MPMCOD::m_manager->getPosition(faces[i](2)) - MPMCOD::m_manager->getPosition(faces[i](0));
		MPMCOD::m_manager->setFaceRestArea(i, (dx0.cross(dx1)).norm() * 0.5);
		MPMCOD::m_manager->setFaceToParameter(i, paramsIndex);

		unique_particles.insert(faces[i](0));
		unique_particles.insert(faces[i](1));
		unique_particles.insert(faces[i](2));
	}
	CHECK_ERROR(unique_particles.size() == gdp->getNumPoints(), "Failed to get all unique particles");

	MPMCOD::m_manager->insertForce(std::make_shared<ThinShellForce>(MPMCOD::m_manager, faces, paramsIndex, 0));
	const std::shared_ptr<ElasticParameters>& params = MPMCOD::m_manager->getElasticParameters(paramsIndex);
	for (int pidx : unique_particles) {
		scalar radius_A = params->getRadiusA(0);
		scalar radius_B = params->getRadiusB(0);
		if (MPMCOD::m_manager->getRadius()(pidx * 2 + 0) == 0.0 || MPMCOD::m_manager->getRadius()(pidx * 2 + 1) == 0.0) {
			MPMCOD::m_manager->setRadius(pidx, radius_A, radius_B);
		}
		scalar vol = MPMCOD::m_manager->getParticleRestArea(pidx) * (radius_A + radius_B);
		MPMCOD::m_manager->setVolume(pidx, vol);
		std::cout << "volrightnow~~~~~~~~~~" << vol << std::endl;


		const scalar original_mass = MPMCOD::m_manager->getM()(pidx * 4);
		scalar mass = params->m_density * vol;
		const scalar original_inertia = MPMCOD::m_manager->getM()(pidx * 4 + 3);
		MPMCOD::m_manager->setMass(pidx, original_mass + mass, original_inertia);
		std::cout << "currentmass~~~~~~~~~" << original_mass + mass << std::endl;

	}
}

void GAS_MPM_CODIMENSIONAL::transferDTAttribTOEigen(const SIM_Geometry *geo, const GU_Detail *gdp) {}


void GAS_MPM_CODIMENSIONAL::loadGravity(SIM_Object* object, const GU_Detail *gdp) {
	SIM_ConstDataArray gravities;
	object->filterConstSubData(gravities, 0, SIM_DataFilterByType("SIM_ForceGravity"), SIM_FORCES_DATANAME, SIM_DataFilterNone());
	const SIM_ForceGravity* force = SIM_DATA_CASTCONST(gravities(0), SIM_ForceGravity);
	CHECK_ERROR(force != nullptr, "Failed to get gravity force");
	UT_Vector3 gravityforce, gravitytorque;
	force->getForce(*object, UT_Vector3(), UT_Vector3(), UT_Vector3(), 1.0f, gravityforce, gravitytorque);
	Vector3s vec_gravityforce;
	vec_gravityforce << gravityforce.x(), gravityforce.y(), gravityforce.z();
	vec_gravityforce *= 100.0f;
	MPMCOD::m_manager->insertForce(std::make_shared<SimpleGravityForce>(vec_gravityforce));
}

/////////////////////////////////////////
// tranfer Data from Eigen to Houdini
/////////////////////////////////////////
void GAS_MPM_CODIMENSIONAL::transferPointAttribTOHoudini(SIM_GeometryCopy *geo, GU_Detail *gdp) {
    int numParticles = gdp->getNumPoints();
    CHECK_ERROR(MPMCOD::m_manager->getNumParticles() == numParticles, "Number of particles in geometry and MPM system do not match");

    const VectorXs& position = MPMCOD::m_manager->getX();
    const VectorXs& velocity = MPMCOD::m_manager->getV();

    GA_RWHandleV3 velHandle(gdp, GA_ATTRIB_POINT, "v");
    CHECK_ERROR(velHandle.isValid(), "Failed to get velocity attribute handle");

    GA_Offset ptoff;
    int idx = 0;
    GA_FOR_ALL_PTOFF(gdp, ptoff) {
        if (idx >= numParticles) break;
        UT_Vector3 pos(position(4 * idx) / 100.0, position(4 * idx + 1) / 100.0, position(4 * idx + 2) / 100.0);
        UT_Vector3 vel(velocity(4 * idx) / 100.0, velocity(4 * idx + 1) / 100.0, velocity(4 * idx + 2) / 100.0);
        gdp->setPos3(ptoff, pos);
        velHandle.set(ptoff, vel);
        idx++;
    }
}


void GAS_MPM_CODIMENSIONAL::transferFaceAttribTOHoudini(SIM_GeometryCopy *geo, GU_Detail *gdp) { }

void GAS_MPM_CODIMENSIONAL::transferDTAttribTOHoudini(SIM_GeometryCopy *geo, GU_Detail *gdp) { }
