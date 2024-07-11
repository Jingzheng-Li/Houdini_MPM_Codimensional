#pragma once

#include <iostream>

#include <GAS/GAS_SubSolver.h>
#include <GAS/GAS_Utils.h>

#include <SIM/SIM_GeometryCopy.h>
#include <SIM/SIM_Geometry.h>
#include <SIM/SIM_OptionsUser.h>
#include <SIM/SIM_Object.h>
#include <SIM/SIM_ScalarField.h>
#include <SIM/SIM_DataFilter.h>
#include <SIM/SIM_Engine.h>
#include <SIM/SIM_VectorField.h>
#include <SIM/SIM_Time.h>
#include <SIM/SIM_ForceGravity.h>
#include <SIM/SIM_Solver.h>
#include <SIM/SIM_DopDescription.h>
#include <GEO/GEO_Primitive.h>
#include <GEO/GEO_PrimVDB.h>
#include <UT/UT_Interrupt.h>
#include <UT/UT_VoxelArray.h>
#include <GA/GA_Handle.h>
#include <GU/GU_Detail.h>
#include <DOP/DOP_Node.h>
#include <DOP/DOP_Engine.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>


#include "Solver/SIMManager.h"
#include "Solver/SceneStepper.h"
#include "Solver/SIMCore.h"
#include "Solver/LinearizedImplicitEuler.h"
#include "SolidForce/SimpleGravityForce.h"
#include "SolidForce/AttachForce.h"
#include "SolidForce/JunctionForce.h"
#include "SolidForce/LevelSetForce.h"

#include "ModelThinShell/ThinShellForce.h"


#define CHECK_ERROR(correctcond, msg) \
    if (!(correctcond)) { \
        std::cerr << msg << std::endl; \
        return; \
    }

#define CHECK_ERROR_SOLVER(correctcond, msg) \
    if (!(correctcond)) { \
        std::cerr << msg << std::endl; \
        return false; \
    }

class GAS_MPM_CODIMENSIONAL : public GAS_SubSolver {

public:

protected:

    explicit GAS_MPM_CODIMENSIONAL(const SIM_DataFactory* factory);
    virtual ~GAS_MPM_CODIMENSIONAL() override;

    bool solveGasSubclass(SIM_Engine& engine,
                        SIM_Object* object,
                        SIM_Time time,
                        SIM_Time timestep) override;

protected:

    void transferPointAttribTOEigen(const SIM_Geometry *geo, const GU_Detail *gdp, const std::shared_ptr<SIMManager>& simmanager);
    void transferFaceAttribTOEigen(const SIM_Geometry *geo, const GU_Detail *gdp, const std::shared_ptr<SIMManager>& simmanager);
    void transferHairAttribTOEigen(const SIM_Geometry *geo, const GU_Detail *gdp);

    void transferPointAttribTOHoudini(SIM_GeometryCopy *geo, GU_Detail *gdp);


    void loadSIMInfos(const std::shared_ptr<SIMManager>& simmanager);
    void loadGravity(SIM_Object* object, const std::shared_ptr<SIMManager>& simmanager);
    void loadVDBCollisions(SIM_Object* object, const std::shared_ptr<SIMManager>& simmanager);

private:

    static const SIM_DopDescription* getDopDescription();

    DECLARE_STANDARD_GETCASTTOTYPE();
    DECLARE_DATAFACTORY(GAS_MPM_CODIMENSIONAL,
                        GAS_SubSolver,
                        "gas mpm codimensional",
                        getDopDescription());

};


class ParticleSimulation {
public:
    ParticleSimulation(const std::shared_ptr<SIMManager>& manager, const std::shared_ptr<SceneStepper>& scenestepper);
	virtual ~ParticleSimulation();

	void stepSystem(const scalar& dt);

public:
	std::shared_ptr<SIMCore> m_core;
};



