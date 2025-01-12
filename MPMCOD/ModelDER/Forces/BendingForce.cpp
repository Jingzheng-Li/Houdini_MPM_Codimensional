
#include "BendingForce.h"

#include "../Dependencies/ElasticStrandUtils.h"
#include "ViscousOrNotViscous.h"

template <typename ViscousT>
void BendingForce<ViscousT>::computeLocal(LocalMultiplierType& localL,
											const StrandForce& strand,
											const IndexType vtx,
											const scalar& dt) {

	if (strand.m_strandParams->m_useTournierJacobian) {
		// B.ilen.(phi+hJv)
		const Mat2& B = ViscousT::bendingMatrix(strand, vtx);
		const Vec4& kappaBar = ViscousT::kappaBar(strand, vtx);
		const scalar ilen = strand.m_invVoronoiLengths[vtx];
		const Vec4& kappa = strand.m_strandState->m_kappas[vtx];
		const GradKType& gradKappa = strand.m_strandState->m_gradKappas[vtx];
		const scalar psi_coeff = 1.0;

		Vec4 Jv = gradKappa.transpose() * strand.m_v_plus.segment<11>(4 * (vtx - 1));

		localL.segment<2>(0) =
				-ilen * psi_coeff * B *
				(kappa.segment<2>(0) - kappaBar.segment<2>(0) + dt * Jv.segment<2>(0));
		localL.segment<2>(2) =
				-ilen * psi_coeff * B *
				(kappa.segment<2>(2) - kappaBar.segment<2>(2) + dt * Jv.segment<2>(2));
	} else {
		localL.setZero();
	}
}

template <typename ViscousT>
scalar BendingForce<ViscousT>::localEnergy(const StrandForce& strand,
																					 const IndexType vtx) {
	const Mat2& B = ViscousT::bendingMatrix(strand, vtx);
	const Vec4& kappaBar = ViscousT::kappaBar(strand, vtx);
	const scalar ilen = strand.m_invVoronoiLengths[vtx];
	const Vec4& kappa = strand.m_strandState->m_kappas[vtx];
	const scalar psi_coeff = 1.0;

	return 0.25 * ilen *
				 ((kappa.segment<2>(0) - kappaBar.segment<2>(0))
							.dot(Vec2(B * (kappa.segment<2>(0) - kappaBar.segment<2>(0)))) +
					(kappa.segment<2>(2) - kappaBar.segment<2>(2))
							.dot(Vec2(B * (kappa.segment<2>(2) - kappaBar.segment<2>(2)))));
}

template <typename ViscousT>
void BendingForce<ViscousT>::computeLocal(Eigen::Matrix<scalar, 11, 1>& localF,
																					const StrandForce& strand,
																					const IndexType vtx) {
	const Mat2& B = ViscousT::bendingMatrix(strand, vtx);
	const Vec4& kappaBar = ViscousT::kappaBar(strand, vtx);
	const scalar ilen = strand.m_invVoronoiLengths[vtx];
	const Vec4& kappa = strand.m_strandState->m_kappas[vtx];
	const GradKType& gradKappa = strand.m_strandState->m_gradKappas[vtx];
	const scalar psi_coeff = 1.0;

	localF = -ilen * psi_coeff * 0.5 *
					 (gradKappa.block<11, 2>(0, 0) * B *
								(kappa.segment<2>(0) - kappaBar.segment<2>(0)) +
						gradKappa.block<11, 2>(0, 2) * B *
								(kappa.segment<2>(2) - kappaBar.segment<2>(2)));
}

template <typename ViscousT>
void BendingForce<ViscousT>::computeLocal(Eigen::Matrix<scalar, 11, 11>& localJ,
																					const StrandForce& strand,
																					const IndexType vtx) {

	const scalar psi_coeff = 1.0;
	const scalar ilen = strand.m_invVoronoiLengths[vtx];
	localJ = strand.m_strandState->m_bendingProducts[vtx] * 0.5;

	if (strand.m_strandParams->m_useApproxJacobian) {
		localJ *= -ilen * ViscousT::bendingCoefficient(strand, vtx) * psi_coeff;
	} else if (strand.m_strandParams->m_useTournierJacobian) {
		localJ *= -ilen * ViscousT::bendingCoefficient(strand, vtx) * psi_coeff;

		const LocalJacobianType& hessKappa0 =
				strand.m_strandState->m_hessKappas[vtx * 4 + 0];
		const LocalJacobianType& hessKappa1 =
				strand.m_strandState->m_hessKappas[vtx * 4 + 1];
		const LocalJacobianType& hessKappa2 =
				strand.m_strandState->m_hessKappas[vtx * 4 + 2];
		const LocalJacobianType& hessKappa3 =
				strand.m_strandState->m_hessKappas[vtx * 4 + 3];

		const Vec4 temp = ViscousT::bendingMultiplier(strand, vtx);
		localJ += (temp(0) * hessKappa0 + temp(1) * hessKappa1 +
							 temp(2) * hessKappa2 + temp(3) * hessKappa3) *
							0.5;
	} else {
		const Mat2& bendingMatrixBase =
				strand.m_strandParams->bendingMatrixBase(vtx);
		const Vec4& kappaBar = ViscousT::kappaBar(strand, vtx);
		const Vec4& kappa = strand.m_strandState->m_kappas[vtx];
		const LocalJacobianType& hessKappa0 =
				strand.m_strandState->m_hessKappas[vtx * 4 + 0];
		const LocalJacobianType& hessKappa1 =
				strand.m_strandState->m_hessKappas[vtx * 4 + 1];
		const LocalJacobianType& hessKappa2 =
				strand.m_strandState->m_hessKappas[vtx * 4 + 2];
		const LocalJacobianType& hessKappa3 =
				strand.m_strandState->m_hessKappas[vtx * 4 + 3];
		const Vec2& temp_e =
				bendingMatrixBase * (kappa.segment<2>(0) - kappaBar.segment<2>(0));
		const Vec2& temp_f =
				bendingMatrixBase * (kappa.segment<2>(2) - kappaBar.segment<2>(2));

		localJ += (temp_e(0) * hessKappa0 + temp_e(1) * hessKappa1 +
							 temp_f(0) * hessKappa2 + temp_f(1) * hessKappa3) *
							0.5;

		localJ *= -ilen * ViscousT::bendingCoefficient(strand, vtx) * psi_coeff;
	}
}

template <typename ViscousT>
void BendingForce<ViscousT>::addInPosition(VecX& globalForce,
																					 const IndexType vtx,
																					 const LocalForceType& localForce) {
	globalForce.segment<11>(4 * (vtx - 1)) += localForce;
}

template <typename ViscousT>
void BendingForce<ViscousT>::addInPosition(VecX& globalMultiplier,
																					 const IndexType vtx,
																					 const LocalMultiplierType& localL) {
	globalMultiplier.segment<4>(4 * vtx) += localL;
}

template <typename ViscousT>
void BendingForce<ViscousT>::accumulateCurrentE(scalar& energy,
																								StrandForce& strand) {
	for (IndexType vtx = s_first; vtx < strand.getNumVertices() - s_last; ++vtx) {
		energy += localEnergy(strand, vtx);
	}
}

template <typename ViscousT>
void BendingForce<ViscousT>::accumulateCurrentF(VecX& force,
																								StrandForce& strand) {
	for (IndexType vtx = s_first; vtx < strand.getNumVertices() - s_last; ++vtx) {
		LocalForceType localF;
		computeLocal(localF, strand, vtx);
		addInPosition(force, vtx, localF);
	}
}

template class BendingForce<NonViscous>;
template class BendingForce<Viscous>;
