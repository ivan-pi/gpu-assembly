
#include <cusolverdx.hpp>

namespace operators {
	
using namespace cusolverdx;

constexpr int M = 15;
constexpr int NRHS = 9;

using GESV = decltype(Size<M,M,NRHS>()
					  + Function<function::gesv_partial_pivot>()
					  + Type<type::real>()
					  + Precision<double>()
					  + Arrangement<col_major>()
					  + LeadingDimension<M>()
					  + SM<900>()
					  + Block());

template<class GESV = GESV, int BatchesPerBlock = 1>
__global__ void assemble_2nd_order(
		typename GESV::a_data_type* A,
		typename GESV::b_data_type* B,
		typename GESV::status_type* info) {

	// copy A and B from global memory to shared memory
	// (optional for Block operators)

	GESV().execute(A_smem, B_smem, info);

	// copy the output A and B from shared memory to global memory
}

} // namespace operators

