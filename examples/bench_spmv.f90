! bench_spmv -- the fixed-row-length CSR product of rbf_csr and the
! ELLPACK product of rbf_ellpack on the same matrix, read from the
! stream data/tools/knn_stream.py writes:
!
!     python data/tools/knn_stream.py 1000000 21 morton knn.bin
!     OMP_NUM_THREADS=4 ./bench_spmv knn.bin
!
! Prints the label, threads, n, k, then the effective bandwidth of
! each product in GB/s, best of nrep passes, counting the bytes of a
! and ja: the reads one product cannot avoid. x is gathered nnzrow
! times and y written once, from and to cache when the ordering is
! good, so the true traffic is higher. The measurements in
! docs/solver.md come from this program built with
! -O2 -march=x86-64-v3 -fopenmp.
program bench_spmv
use, intrinsic :: iso_c_binding, only: c_int, c_double
use omp_lib, only: omp_get_wtime, omp_get_max_threads
use rbf_csr, only: csr_mv
use rbf_ellpack, only: ellpack_mv
implicit none

integer, parameter :: wp = c_double, nrep = 20
real(wp), allocatable :: a_csr(:, :), a_ell(:, :), x(:), y(:), yref(:)
integer(c_int), allocatable :: ja_csr(:, :), ja_ell(:, :)
integer(c_int) :: n, k
integer :: i, rep, u
real(wp) :: t0, t1, tcsr, tell, bytes
character(len=256) :: fname, label

call get_command_argument(1, fname)
if (len_trim(fname) == 0) error stop "usage: bench_spmv stream [label]"
call get_command_argument(2, label)
if (len_trim(label) == 0) label = fname

open(newunit=u, file=fname, access="stream", form="unformatted", status="old")
read(u) n, k
allocate(ja_csr(k, n), a_csr(k, n), ja_ell(n, k), a_ell(n, k), x(n), y(n), yref(n))
read(u) ja_csr
read(u) a_csr
close(u)
ja_ell = transpose(ja_csr)
a_ell = transpose(a_csr)
do i = 1, n
    x(i) = sin(0.001_wp*i)
end do

! the two must agree before their times mean anything
call csr_mv(n, k, 1.0_wp, a_csr, ja_csr, k, x, 0.0_wp, yref)
call ellpack_mv(n, k, 1.0_wp, a_ell, ja_ell, n, x, 0.0_wp, y)
if (maxval(abs(y - yref)) > 1.0e-12_wp*maxval(abs(yref))) error stop "the products disagree"

tcsr = huge(1.0_wp)
do rep = 1, nrep
    t0 = omp_get_wtime()
    call csr_mv(n, k, 1.0_wp, a_csr, ja_csr, k, x, 0.0_wp, y)
    t1 = omp_get_wtime()
    tcsr = min(tcsr, t1 - t0)
end do
tell = huge(1.0_wp)
do rep = 1, nrep
    t0 = omp_get_wtime()
    call ellpack_mv(n, k, 1.0_wp, a_ell, ja_ell, n, x, 0.0_wp, y)
    t1 = omp_get_wtime()
    tell = min(tell, t1 - t0)
end do

bytes = real(n, wp)*k*(8 + 4)
print '(a,1x,i0,1x,i0,1x,i0,a,f6.1,a,f6.1,a)', trim(label), omp_get_max_threads(), n, k, &
    "  csr_mv ", bytes/tcsr/1e9, " GB/s  ellpack_mv ", bytes/tell/1e9, " GB/s"
end program
