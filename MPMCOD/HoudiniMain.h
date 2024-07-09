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

    void transferPointAttribTOEigen(const SIM_Geometry *geo, const GU_Detail *gdp);
    void transferFaceAttribTOEigen(const SIM_Geometry *geo, const GU_Detail *gdp);
    void transferHairAttribTOEigen(const SIM_Geometry *geo, const GU_Detail *gdp);
    void transferDTAttribTOEigen(const SIM_Geometry *geo, const GU_Detail *gdp);

    void transferPointAttribTOHoudini(SIM_GeometryCopy *geo, GU_Detail *gdp);
    void transferFaceAttribTOHoudini(SIM_GeometryCopy *geo, GU_Detail *gdp);
    void transferDTAttribTOHoudini(SIM_GeometryCopy *geo, GU_Detail *gdp);

    void loadSIMInfos();
    void loadGravity(SIM_Object* object, const GU_Detail *gdp);

private:

    // 总感觉这个指针这里有点问题
    std::shared_ptr<SIMManager> m_manager;
    std::shared_ptr<SIMCore> m_core;


    scalar m_criterion;
    int m_maxiters;

    SIMInfo info;


private:

    static const SIM_DopDescription* getDopDescription();

    DECLARE_STANDARD_GETCASTTOTYPE();
    DECLARE_DATAFACTORY(GAS_MPM_CODIMENSIONAL,
                        GAS_SubSolver,
                        "gas mpm codimensional",
                        getDopDescription());

};
