
#include "operators.hpp"

extern "C" {
	__global__ void solve(double *A, double *B, int *info) {
		operators::assemble_2nd_order(A,B,*info);
	}
}