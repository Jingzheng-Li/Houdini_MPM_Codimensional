
#include "TwistingForce.h"

#include "ViscousOrNotViscous.h"

template <typename ViscousT>
void TwistingForce<ViscousT>::computeLocal(LocalMultiplierType& localL,
                                           const StrandForce& strand,
                                           const IndexType vtx,
                                           const scalar& dt) {
  if (strand.m_strandParams->m_useTournierJacobian) {
    const scalar kt = ViscousT::kt(strand, vtx);
    const scalar undefTwist = ViscousT::thetaBar(strand, vtx);
    const scalar ilen = strand.m_invVoronoiLengths[vtx];
    const scalar twist = strand.m_strandState->m_twists[vtx];
    const scalar psi_coeff = 1.0;

    localL = -kt * ilen * psi_coeff * (twist - undefTwist +
              dt * strand.m_strandState->m_gradTwists[vtx].dot(
                       strand.m_v_plus.segment<11>(4 * (vtx - 1))));
  } else {
    localL = 0.0;
  }
}

template <typename ViscousT>
scalar TwistingForce<ViscousT>::localEnergy(const StrandForce& strand,
                                            const IndexType vtx) {
  const scalar kt = ViscousT::kt(strand, vtx);
  const scalar undefTwist = ViscousT::thetaBar(strand, vtx);
  const scalar ilen = strand.m_invVoronoiLengths[vtx];
  const scalar twist = strand.m_strandState->m_twists[vtx];
  const scalar psi_coeff = 1.0;

  return 0.5 * kt * square(twist - undefTwist) * ilen * psi_coeff;
}

template <typename ViscousT>
void TwistingForce<ViscousT>::computeLocal(Eigen::Matrix<scalar, 11, 1>& localF,
                                           const StrandForce& strand,
                                           const IndexType vtx) {
  const scalar kt = ViscousT::kt(strand, vtx);
  const scalar undefTwist = ViscousT::thetaBar(strand, vtx);
  const scalar ilen = strand.m_invVoronoiLengths[vtx];
  const scalar twist = strand.m_strandState->m_twists[vtx];
  const scalar psi_coeff = 1.0;

  localF = -kt * ilen * (twist - undefTwist) *
           strand.m_strandState->m_gradTwists[vtx] * psi_coeff;
}

template <typename ViscousT>
void TwistingForce<ViscousT>::computeLocal(
    Eigen::Matrix<scalar, 11, 11>& localJ, const StrandForce& strand,
    const IndexType vtx) {
  const scalar kt = ViscousT::kt(strand, vtx);
  const scalar ilen = strand.m_invVoronoiLengths[vtx];
  const Mat11& gradTwistSquared =
      strand.m_strandState->m_gradTwistsSquared[vtx];
  const scalar psi_coeff = 1.0;

  localJ = -kt * ilen * gradTwistSquared * psi_coeff;
  if (strand.m_strandParams->m_useApproxJacobian) {
    return;
  } else if (strand.m_strandParams->m_useTournierJacobian) {
    const Mat11& hessTwist = strand.m_strandState->m_hessTwists[vtx];
    const scalar localL = ViscousT::twistingMultiplier(strand, vtx);

    localJ += localL * hessTwist;
  } else {
    const scalar undeformedTwist = ViscousT::thetaBar(strand, vtx);
    const scalar twist = strand.m_strandState->m_twists[vtx];
    const Mat11& hessTwist = strand.m_strandState->m_hessTwists[vtx];
    localJ += -kt * ilen * (twist - undeformedTwist) * hessTwist;
  }
}

template <typename ViscousT>
void TwistingForce<ViscousT>::addInPosition(VecX& globalForce,
                                            const IndexType vtx,
                                            const LocalForceType& localForce) {
  globalForce.segment<11>(4 * (vtx - 1)) += localForce;
}

template <typename ViscousT>
void TwistingForce<ViscousT>::addInPosition(VecX& globalMultiplier,
                                            const IndexType vtx,
                                            const LocalMultiplierType& localL) {
  globalMultiplier(vtx) += localL;
}

template <typename ViscousT>
void TwistingForce<ViscousT>::accumulateCurrentE(scalar& energy,
                                                 StrandForce& strand) {
  for (IndexType vtx = s_first; vtx < strand.getNumVertices() - s_last; ++vtx) {
    energy += localEnergy(strand, vtx);
  }
}

template <typename ViscousT>
void TwistingForce<ViscousT>::accumulateCurrentF(VecX& force,
                                                 StrandForce& strand) {
  for (IndexType vtx = s_first; vtx < strand.getNumVertices() - s_last; ++vtx) {
    LocalForceType localF;
    computeLocal(localF, strand, vtx);
    addInPosition(force, vtx, localF);
  }
}

template class TwistingForce<NonViscous>;
template class TwistingForce<Viscous>;
