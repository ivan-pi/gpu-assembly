! Timing of the host RBF-FD assembly, rbf_fd_assemble of src/rbf_fd.f90,
! on a case of the file formats in docs/file_formats.md:
!
!   rbf_fd_bench <case> <Lx> <Ly> [p] [q] [nrhs] [reps]
!
! <case> is the path prefix of a points/graph pair on the periodic box
! [0, Lx) x [0, Ly); p and q the polynomial degree and PHS exponent
! (default 3 and 3); nrhs how many of the operators [Laplacian, d/dx,
! d/dy, d2/dx2, d2/dxdy, d2/dy2, interpolation at an offset] to assemble
! (default 1, the Laplacian alone); reps how many assemblies to time,
! of which the fastest counts (default 5).
!
! Each solver assembles the operators into fixed-row-length CSR values
! through the gather/scatter callbacks, on all OpenMP threads; then one
! thread times the phases of the per-stencil work in isolation.
program rbf_fd_bench
use, intrinsic :: iso_fortran_env, only: int64
!$ use omp_lib, only: omp_get_max_threads
use rbf_precision, only: wp
use rbf_periodic_box, only: periodic_box
use rbf_io, only: read_points, read_graph_csr
use rbf_fd
implicit none

integer, parameter :: nops = 7
integer, parameter :: ops(nops) = [OP_LAPLACE, OP_DX, OP_DY, OP_DXX, OP_DXY, OP_DYY, OP_VALUE]
character(len=*), parameter :: names(nops) = [character(len=8) :: &
    "lap", "d/dx", "d/dy", "d2/dx2", "d2/dxdy", "d2/dy2", "interp"]

integer :: npts, nnz, nmax, p, q, nrhs, reps, info, rep, solver, nthreads
integer, allocatable :: ia(:), ja(:)
real(wp), allocatable :: x(:), y(:), va(:,:), va_lu(:,:), xs(:), ys(:)
real(wp) :: Lx, Ly, xc(nops), yc(nops), t, best(4), tg, tf, ts
type(periodic_box) :: box
type(rbf_fd_workspace) :: ws
character(256) :: arg

if (command_argument_count() < 3) then
    print '(a)', "usage: rbf_fd_bench <case> <Lx> <Ly> [p] [q] [nrhs] [reps]"
    error stop
end if
call get_command_argument(1, arg)
call read_points(trim(arg)//".points", npts, x, y)
call read_graph_csr(trim(arg)//".graph", nnz, ia, ja)
if (nnz /= npts) error stop "points and graph disagree on the node count"
nnz = ia(npts)
nmax = maxval(ia(1:npts) - ia(0:npts - 1))
call get_command_argument(2, arg)
read (arg, *) Lx
call get_command_argument(3, arg)
read (arg, *) Ly
box = periodic_box(Lx, Ly)
p = 3
q = 3
nrhs = 1
reps = 5
if (command_argument_count() >= 4) then
    call get_command_argument(4, arg)
    read (arg, *) p
end if
if (command_argument_count() >= 5) then
    call get_command_argument(5, arg)
    read (arg, *) q
end if
if (command_argument_count() >= 6) then
    call get_command_argument(6, arg)
    read (arg, *) nrhs
end if
if (command_argument_count() >= 7) then
    call get_command_argument(7, arg)
    read (arg, *) reps
end if
if (nrhs < 1 .or. nrhs > nops) error stop "nrhs must be between 1 and 7"

! The derivatives at the node, interpolation at an offset from it
xc = 0
yc = 0
xc(nops) = 0.3_wp
yc(nops) = -0.2_wp

nthreads = 1
!$ nthreads = omp_get_max_threads()

print '(a,i0,a,i0,a,i0,a,i0)', "case: ", npts, " stencils, ", nnz, " nonzeros, ", &
    minval(ia(1:npts) - ia(0:npts - 1)), " to ", nmax, " nodes each"
print '(a,i0,a,i0,a,i0,a,i0,a,i0)', "p = ", p, ", q = ", q, ", nt <= ", nmax + npoly(p), &
    ", nrhs = ", nrhs, ", threads = ", nthreads
print '(a,*(a,1x))', "operators: ", (trim(names(rep)), rep=1, nrhs)
print *

allocate(va(nnz, nrhs), va_lu(nnz, nrhs))

! --- the full assembly, both solvers, all threads ---

print '(a)', "assembly, all threads, best of reps:"
print '(a)', "  solver         seconds   stencils/s        nnz/s   max rel diff to LU"
do solver = SOLVER_LU, SOLVER_RECLU_LDLT
    ws = rbf_fd_workspace(q, p, nmax, nrhs, solver)
    best(solver) = huge(t)
    do rep = 1, reps
        t = wtime()
        call rbf_fd_assemble(ws, npts, nrhs, ops(1:nrhs), xc(1:nrhs), yc(1:nrhs), gather, scatter, info)
        t = wtime() - t
        if (info /= 0) then
            print '(a,i0)', "singular stencil ", info
            error stop
        end if
        best(solver) = min(best(solver), t)
    end do
    if (solver == SOLVER_LU) va_lu = va
    print '(a,a,f10.4,2es13.3,es14.2)', "  ", solver_name(solver), &
        best(solver), npts/best(solver), nnz/best(solver), maxval(abs(va - va_lu))/maxval(abs(va_lu))
end do
print *

! --- the phases of one stencil, one thread ---

print '(a)', "per stencil, one thread, microseconds:"
print '(a)', "  solver         gather      fill     solve     total"
allocate(xs(nmax), ys(nmax))
do solver = SOLVER_LU, SOLVER_RECLU_LDLT
    ws = rbf_fd_workspace(q, p, nmax, nrhs, solver)
    tg = huge(t)
    tf = huge(t)
    ts = huge(t)
    do rep = 1, reps
        t = wtime()
        call sweep(0)
        tg = min(tg, wtime() - t)
        t = wtime()
        call sweep(1)
        tf = min(tf, wtime() - t)
        t = wtime()
        call sweep(2)
        ts = min(ts, wtime() - t)
    end do
    print '(a,a,4f10.3)', "  ", solver_name(solver), 1.0e6_wp*[tg, tf - tg, ts - tf, ts]/npts
end do

contains

    ! The stencil loop up to a phase: 0 gather, 1 fill, 2 solve
    subroutine sweep(phase)
        integer, intent(in) :: phase
        integer :: s, n
        do s = 1, npts
            call gather(s, nmax, n, xs, ys)
            if (phase >= 1) call rbf_fd_fill(ws, n, xs, ys, nrhs, ops(1:nrhs), xc(1:nrhs), yc(1:nrhs))
            if (phase >= 2) then
                call rbf_fd_solve(ws, n, nrhs, info)
                if (info /= 0) error stop "singular stencil"
            end if
        end do
    end subroutine

    ! The callbacks, recursive because the assembly's threads call them
    ! concurrently. The nodes of stencil s as minimum-image
    ! displacements from node s:
    recursive subroutine gather(s, nmax, n, xs, ys)
        integer, intent(in) :: s, nmax
        integer, intent(out) :: n
        real(wp), intent(out) :: xs(nmax), ys(nmax)
        integer :: l, col
        real(wp) :: d(2)
        n = ia(s) - ia(s - 1)
        do l = 1, n
            col = ja(ia(s - 1) + l - 1) + 1
            d = box%minimum_image([x(col) - x(s), y(col) - y(s)])
            xs(l) = d(1)
            ys(l) = d(2)
        end do
    end subroutine

    ! Row s of the CSR values, one column per operator
    recursive subroutine scatter(s, n, nrhs, w, ldw)
        integer, intent(in) :: s, n, nrhs, ldw
        real(wp), intent(in) :: w(ldw, *)
        va(ia(s - 1) + 1:ia(s), 1:nrhs) = w(1:n, 1:nrhs)
    end subroutine

    function wtime() result(t)
        real(wp) :: t
        integer(int64) :: count, rate
        call system_clock(count, rate)
        t = real(count, wp)/real(rate, wp)
    end function

end program
