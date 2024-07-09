
#ifndef FORCE_H
#define FORCE_H

#include <Eigen/Core>

#include "Utils/MathDefs.h"

class SIMManager;

class Force {
 public:
  virtual ~Force();

  virtual void addEnergyToTotal(const VectorXs& x, const VectorXs& v,
                                const VectorXs& m,  scalar& E) = 0;

  virtual void addGradEToTotal(const VectorXs& x, const VectorXs& v,
                               const VectorXs& m, VectorXs& gradE) = 0;

  virtual void addHessXToTotal(const VectorXs& x, const VectorXs& v,
                               const VectorXs& m, TripletXs& hessE,
                               int hessE_index, const scalar& dt) = 0;

  virtual void addAngularHessXToTotal(const VectorXs& x, const VectorXs& v,
                                      const VectorXs& m, TripletXs& hessE,
                                      int hessE_index, const scalar& dt);

  virtual int numHessX() = 0;

  virtual int numAngularHessX();

  virtual void preCompute() = 0;

  virtual void updateMultipliers(const VectorXs& x, const VectorXs& vplus,
                                 const VectorXs& m, const scalar& dt) = 0;

  virtual void updateStartState() = 0;

  virtual Force* createNewCopy() = 0;

  virtual void postCompute(VectorXs& v, const scalar& dt);

  virtual int flag() const = 0;

  virtual bool parallelized() const;
  
};

#endif
