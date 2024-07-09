
#include "HoudiniMain.h"

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
    m_scene.reset();
    
}

bool GAS_MPM_CODIMENSIONAL::solveGasSubclass(SIM_Engine& engine,
                                            SIM_Object* object,
                                            SIM_Time time,
                                            SIM_Time timestep) {

    if (!m_scene) {
        m_scene = std::make_shared<SIMManager>();
    }

    const SIM_Geometry *geo = object->getGeometry();
    CHECK_ERROR_SOLVER(geo != NULL, "Failed to get readBuffer geometry object")
    GU_DetailHandleAutoReadLock readlock(geo->getGeometry());
    CHECK_ERROR_SOLVER(readlock.isValid(), "Failed to get readBuffer geometry detail");
    const GU_Detail *gdp = readlock.getGdp();
    CHECK_ERROR_SOLVER(!gdp->isEmpty(), "readBuffer Geometry is empty");

    transferPTAttribTOEigen(geo, gdp);
    transferPRIMAttribTOEigen(geo, gdp);
    transferDTAttribTOEigen(geo, gdp);

    SIM_GeometryCopy *newgeo = SIM_DATA_CREATE(*object, "Geometry", SIM_GeometryCopy, SIM_DATA_RETURN_EXISTING | SIM_DATA_ADOPT_EXISTING_ON_DELETE);
    CHECK_ERROR_SOLVER(newgeo != NULL, "Failed to create writeBuffer GeometryCopy object");
    GU_DetailHandleAutoWriteLock writelock(newgeo->getOwnGeometry());
    CHECK_ERROR_SOLVER(writelock.isValid(), "Failed to get writeBuffer geometry detail");
    GU_Detail *newgdp = writelock.getGdp();
    CHECK_ERROR_SOLVER(!newgdp->isEmpty(), "writeBuffer Geometry is empty");

    transferPTAttribTOHoudini(geo, newgdp);
    transferPRIMAttribTOHoudini(geo, newgdp);
    transferDTAttribTOHoudini(geo, newgdp);

    return true;
}

/////////////////////////////////////////
// tranfer Data from Houdini to Eigen
/////////////////////////////////////////
void GAS_MPM_CODIMENSIONAL::transferPTAttribTOEigen(const SIM_Geometry *geo, const GU_Detail *gdp) {

    // particle velocity [theta omega fixed] radius biradius vol fvol group m fm
    // TODO: how to fix group??
    // TODO: unit should be cm!

    int numParticles = gdp->getNumPoints();
    CHECK_ERROR(numParticles > 0, "No particles found in geometry");
    Eigen::MatrixXd positions(3, numParticles);
    Eigen::MatrixXd velocities(3, numParticles);
    Eigen::VectorXd masses(numParticles);
    Eigen::VectorXd radius(numParticles);

    GA_ROHandleV3D velHandle(gdp, GA_ATTRIB_POINT, "v");
    GA_ROHandleD massHandle(gdp, GA_ATTRIB_POINT, "mass");
    GA_ROHandleD radiusHandle(gdp, GA_ATTRIB_POINT, "pscale");
    CHECK_ERROR(velHandle.isValid(), "Failed to get velocity attributes");
    CHECK_ERROR(massHandle.isValid(), "Failed to get mass attributes");
    CHECK_ERROR(radiusHandle.isValid(), "Failed to get radius attributes");

    GA_Offset ptoff;
    int idx = 0; // Index to track the row in positions and velocities
    GA_FOR_ALL_PTOFF(gdp, ptoff) {
        if (idx >= numParticles) break; // Ensure we don't go out of bounds
        UT_Vector3D pos3 = gdp->getPos3D(ptoff);
        UT_Vector3D vel3 = velHandle.get(ptoff);
        positions.row(idx) << pos3.x() * 100.0, pos3.y() * 100.0, pos3.z() * 100.0;
        velocities.row(idx) << vel3.x() * 100.0, vel3.y() * 100.0, vel3.z() * 100.0;
        masses(idx) = massHandle.get(ptoff) * 1000.0;
        radius(idx) = radiusHandle.get(ptoff) * 100.0;
        idx++; // Increment the index for the next row
    }

    CHECK_ERROR(idx == gdp->getNumPoints(), "Failed to get all points");

    m_scene->resizeParticleSystem(numParticles);
    for (int i = 0; i < numParticles; ++i) {
        m_scene->setPosition(i, positions.row(i).transpose());
        m_scene->setVelocity(i, velocities.row(i).transpose());
        m_scene->setMass(i, masses(i), 0.0);
        m_scene->setRadius(i, radius(i), radius(i));
    }


    m_criterion = 1e-6;
    m_maxiters = 100;
    std::shared_ptr<SceneStepper> scenestepper = nullptr;
    scenestepper = std::make_shared<LinearizedImplicitEuler>(m_criterion, m_maxiters);
    CHECK_ERROR(scenestepper != nullptr, "Failed to create LinearizedImplicitEuler stepper");

    info.use_pcr = true;
    info.solve_solid = true;
    info.viscosity = 8.9e-3;
    info.lambda = 2.0;
    info.elasto_flip_asym_coeff = 1.0;
    info.elasto_flip_coeff = 0.95;
    info.elasto_advect_coeff = 1.0;
    info.bending_scheme = 2;
    info.levelset_young_modulus = 6.6e6;
    info.use_group_precondition = false;
    info.iteration_print_step = 0;


}

void GAS_MPM_CODIMENSIONAL::transferPRIMAttribTOEigen(const SIM_Geometry *geo, const GU_Detail *gdp) {}

void GAS_MPM_CODIMENSIONAL::transferDTAttribTOEigen(const SIM_Geometry *geo, const GU_Detail *gdp) {}







/////////////////////////////////////////
// tranfer Data from Eigen to Houdini
/////////////////////////////////////////
void GAS_MPM_CODIMENSIONAL::transferPTAttribTOHoudini(const SIM_Geometry *geo, const GU_Detail *gdp) {
    

}

void GAS_MPM_CODIMENSIONAL::transferPRIMAttribTOHoudini(const SIM_Geometry *geo, const GU_Detail *gdp) {}

void GAS_MPM_CODIMENSIONAL::transferDTAttribTOHoudini(const SIM_Geometry *geo, const GU_Detail *gdp) {}
