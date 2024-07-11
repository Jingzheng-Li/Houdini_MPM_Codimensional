
#include "HoudiniMain.h"
#include <unordered_set>


namespace FIRSTFRAME {
	static bool hou_initialized = false;

	static const scalar dt = 0.001;
	static const scalar criterion = 1e-6;
	static const int maxiters = 100;
	static const scalar bucket_size = 2.0; // unit: cm
	static const int num_cells = 4;
	static const int kernel_order = 2;

	static std::shared_ptr<ParticleSimulation> execsim;
};

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

GAS_MPM_CODIMENSIONAL::GAS_MPM_CODIMENSIONAL(const SIM_DataFactory* factory) : BaseClass(factory) { }

GAS_MPM_CODIMENSIONAL::~GAS_MPM_CODIMENSIONAL() { 
	FIRSTFRAME::hou_initialized = false;
	FIRSTFRAME::execsim.reset();
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

	if (!FIRSTFRAME::hou_initialized) {

		std::cout << "hou_initialized~~~~~~~~~" << std::endl;

		Eigen::initParallel();
		Eigen::setNbThreads(std::thread::hardware_concurrency());
		omp_set_num_threads(16);
		srand(0x0108170F);

		auto manager = std::make_shared<SIMManager>(); 
		std::shared_ptr<SceneStepper> scenestepper = nullptr;
		scenestepper = std::make_shared<LinearizedImplicitEuler>(FIRSTFRAME::criterion, FIRSTFRAME::maxiters);
		CHECK_ERROR_SOLVER(scenestepper != nullptr, "Failed to create scene stepper");

		loadSIMInfos(manager);

		transferPointAttribTOEigen(geo, gdp, manager);

		loadVDBCollisions(object, manager); // load VDB objects

		int maxgroup = 0;
		manager->resizeGroups(maxgroup + 1);
		manager->updateRestPos();
		manager->initGroupPos();

		transferFaceAttribTOEigen(geo, gdp, manager); // load faces

		manager->initGaussSystem();
		manager->updateParticleBoundingBox();
		manager->rebucketizeParticles();
		manager->resampleNodes();
		manager->computeWeights(0.0);
		manager->updatePlasticity(0.0);
		manager->computedEdFe();
		
		// manager->updateSolidPhi(); // TODO: change inside
		// manager->updateSolidWeights(); // TODO: may be useless(this is for fluidsolid)
		manager->mapParticleNodesAPIC();
		manager->saveParticleVelocity();

		manager->loadAttachForces();
		manager->insertForce(std::make_shared<JunctionForce>(manager));
		manager->insertForce(std::make_shared<LevelSetForce>(manager, manager->getCellSize() * manager->getSIMInfo().levelset_thickness));

		loadGravity(object, manager);

		FIRSTFRAME::execsim = std::make_shared<ParticleSimulation>(manager, scenestepper);

		FIRSTFRAME::hou_initialized = true;

	}

	

	///////////////////////////////////////////
	// MPM simulation
	///////////////////////////////////////////
	CHECK_ERROR_SOLVER(FIRSTFRAME::execsim, "not initialize ParticleSimulation");
	for (int substep = 0; substep < 10; ++substep) {
		FIRSTFRAME::execsim->stepSystem(FIRSTFRAME::dt);
	}








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


void GAS_MPM_CODIMENSIONAL::loadSIMInfos(const std::shared_ptr<SIMManager>& simmanager) {

	// load SIMInfo
	SIMInfo siminfo;
	siminfo.viscosity = 8.9e-3;
	siminfo.elasto_flip_asym_coeff = 1.0;
	siminfo.elasto_flip_coeff = 0.95;
	siminfo.elasto_advect_coeff = 1.0;
	siminfo.bending_scheme = 2;
	siminfo.levelset_young_modulus = 6.6e6;
	siminfo.iteration_print_step = 0;
	siminfo.use_twist = true;
	siminfo.levelset_thickness = 0.25;
	simmanager->setSIMInfo(siminfo);


	// load bucket parameters
	simmanager->setBucketInfo(
		FIRSTFRAME::bucket_size, 
		FIRSTFRAME::num_cells, 
		FIRSTFRAME::kernel_order);


	// load material parameters
	scalar radius = 0.0165;
	scalar biradius = radius;
	scalar YoungsModulus = 6.6e5;
	scalar poissonRatio = 0.35;
	scalar shearModulus = YoungsModulus / ((1.0 + poissonRatio) * 2.0);
	scalar density = 1.32;
	scalar viscosity = 1e3;
	scalar stretchingMultiplier = 1.0;
	scalar collisionMultiplier = 1.0;
	scalar attachMultiplier = 0.1;
	scalar baseRotation = 0.0;
	bool accumulateWithViscous = true;
	bool accumulateViscousOnlyForBendingModes = false;
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
	rad_vec(0) = radius;
	rad_vec(1) = biradius;
	simmanager->insertElasticParameters(
	std::make_shared<ElasticParameters>(
		rad_vec, YoungsModulus, shearModulus, stretchingMultiplier,
		collisionMultiplier, attachMultiplier, density, viscosity, baseRotation,
		FIRSTFRAME::dt, friction_alpha, friction_beta,
		accumulateWithViscous, accumulateViscousOnlyForBendingModes,
		postProjectFixed, useApproxJacobian, useTournierJacobian, straightHairs,
		haircolor));

}

/////////////////////////////////////////
// tranfer Data from Houdini to Eigen
/////////////////////////////////////////
void GAS_MPM_CODIMENSIONAL::transferPointAttribTOEigen(const SIM_Geometry *geo, const GU_Detail *gdp, const std::shared_ptr<SIMManager>& simmanager) {

	int numParticles = gdp->getNumPoints();
	CHECK_ERROR(numParticles > 0, "No particles found in geometry");
	simmanager->resizeParticleSystem(numParticles);

	GA_ROHandleV3D velHandle(gdp, GA_ATTRIB_POINT, "v");
	GA_ROHandleD massHandle(gdp, GA_ATTRIB_POINT, "mass");
	GA_ROHandleD pscaleHandle(gdp, GA_ATTRIB_POINT, "pscale");
	CHECK_ERROR(velHandle.isValid(), "Failed to get velocity attributes");
	CHECK_ERROR(massHandle.isValid(), "Failed to get mass attributes");
	CHECK_ERROR(pscaleHandle.isValid(), "Failed to get pscale attributes");

	GA_Offset ptoff;
	int idx = 0;
	Vector3s p_pos = Vector3s::Zero();
	Vector3s p_vel = Vector3s::Zero();
	int p_fixed = 0;
	scalar p_scale = 0.0;
	scalar p_mass = 0.0;
	GA_FOR_ALL_PTOFF(gdp, ptoff) {
		if (idx >= numParticles) break;
		UT_Vector3D pos3 = gdp->getPos3D(ptoff);
		UT_Vector3D vel3 = velHandle.get(ptoff);
		
		p_pos << pos3.x() * 100.0, pos3.y() * 100.0, pos3.z() * 100.0;
		simmanager->setPosition(idx, p_pos);

		p_vel << vel3.x() * 100.0, vel3.y() * 100.0, vel3.z() * 100.0;
		simmanager->setVelocity(idx, p_vel);

		p_fixed = 0;
		simmanager->setFixed(idx, (unsigned char)(p_fixed & 0xFFU));
		simmanager->setTwist(idx, false);

		p_scale = pscaleHandle.get(ptoff) * 100.0;
		simmanager->setRadius(idx, p_scale, p_scale);

		p_mass = massHandle.get(ptoff) * 1000.0;
		simmanager->setMass(idx, p_mass, 0.0);

		simmanager->setGroup(idx, 0);

		idx++;
	}
	CHECK_ERROR(idx == gdp->getNumPoints(), "Failed to get all points");

}

void GAS_MPM_CODIMENSIONAL::transferFaceAttribTOEigen(const SIM_Geometry *geo, const GU_Detail *gdp, const std::shared_ptr<SIMManager>& simmanager) {

	int numfaces = gdp->getNumPrimitives();
	std::vector<Vector3i> faces(numfaces);

	int faceIndex = 0;
	GA_Offset primoff;
	Vector3i face = Vector3i::Zero();
	GA_FOR_ALL_PRIMOFF(gdp, primoff) {
		const GA_Primitive* prim = gdp->getPrimitive(primoff);
		if (prim->getVertexCount() == 3) {
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
	simmanager->conservativeResizeFaces(numfaces);

	for (int i = 0; i < numfaces; ++i) {
		simmanager->setFace(i, faces[i]);
		Vector3s dx0 = simmanager->getPosition(faces[i](1)) - simmanager->getPosition(faces[i](0));
		Vector3s dx1 = simmanager->getPosition(faces[i](2)) - simmanager->getPosition(faces[i](0));
		simmanager->setFaceRestArea(i, (dx0.cross(dx1)).norm() * 0.5);
		simmanager->setFaceToParameter(i, paramsIndex);

		unique_particles.insert(faces[i](0));
		unique_particles.insert(faces[i](1));
		unique_particles.insert(faces[i](2));
	}
	CHECK_ERROR(unique_particles.size() == gdp->getNumPoints(), "Failed to get all unique particles");

	simmanager->insertForce(std::make_shared<ThinShellForce>(simmanager, faces, paramsIndex, 0));
	const std::shared_ptr<ElasticParameters>& params = simmanager->getElasticParameters(paramsIndex);
	for (int pidx : unique_particles) {
		scalar radius_A = params->getRadiusA(0);
		scalar radius_B = params->getRadiusB(0);
		if (simmanager->getRadius()(pidx * 2 + 0) == 0.0 || simmanager->getRadius()(pidx * 2 + 1) == 0.0) {
			simmanager->setRadius(pidx, radius_A, radius_B);
		}
		scalar vol = simmanager->getParticleRestArea(pidx) * (radius_A + radius_B);
		simmanager->setVolume(pidx, vol);
		std::cout << "volrightnow~~~~~~~~~~" << vol << std::endl;

		const scalar original_mass = simmanager->getM()(pidx * 4);
		scalar mass = params->m_density * vol;
		const scalar original_inertia = simmanager->getM()(pidx * 4 + 3);
		simmanager->setMass(pidx, original_mass + mass, original_inertia);
		std::cout << "currentmass~~~~~~~~~" << original_mass + mass << std::endl;
	}
}


void GAS_MPM_CODIMENSIONAL::loadGravity(SIM_Object* object, const std::shared_ptr<SIMManager>& simmanager) {
	SIM_ConstDataArray gravities;
	object->filterConstSubData(gravities, 0, SIM_DataFilterByType("SIM_ForceGravity"), SIM_FORCES_DATANAME, SIM_DataFilterNone());
	const SIM_ForceGravity* force = SIM_DATA_CASTCONST(gravities(0), SIM_ForceGravity);
	CHECK_ERROR(force != nullptr, "Failed to get gravity force");
	UT_Vector3 gravityforce, gravitytorque;
	force->getForce(*object, UT_Vector3(), UT_Vector3(), UT_Vector3(), 1.0f, gravityforce, gravitytorque);
	Vector3s vec_gravityforce;
	vec_gravityforce << gravityforce.x(), gravityforce.y(), gravityforce.z();
	vec_gravityforce *= 100.0f;
	simmanager->insertForce(std::make_shared<SimpleGravityForce>(vec_gravityforce));
}


/////////////////////////////////////////
// tranfer Data from Eigen to Houdini
/////////////////////////////////////////
void GAS_MPM_CODIMENSIONAL::transferPointAttribTOHoudini(SIM_GeometryCopy *geo, GU_Detail *gdp) {

	const std::shared_ptr<SIMManager>& localmanager = FIRSTFRAME::execsim->m_core->getScene();
    int numParticles = gdp->getNumPoints();
    CHECK_ERROR(localmanager->getNumParticles() == numParticles, "Number of particles in geometry and MPM system do not match");

    const VectorXs& position = localmanager->getX();
    const VectorXs& velocity = localmanager->getV();

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


void GAS_MPM_CODIMENSIONAL::loadVDBCollisions(SIM_Object* object, const std::shared_ptr<SIMManager>& simmanager) {

	SIM_Data *colliderData = object->getNamedSubData("Colliders");
    if (!colliderData) {
        std::cerr << "No Colliders data or incorrect data type" << std::endl;
        return;
    }

}







ParticleSimulation::ParticleSimulation(
    const std::shared_ptr<SIMManager>& scene,
    const std::shared_ptr<SceneStepper>& scene_stepper)
	: m_core(std::make_shared<SIMCore>(scene, scene_stepper)) {}

ParticleSimulation::~ParticleSimulation() {}

void ParticleSimulation::stepSystem(const scalar& dt) {
	m_core->stepSystem(dt);
}