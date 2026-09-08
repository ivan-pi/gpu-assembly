! Tests of rbf_fd, the host RBF-FD weights.
!
!   test_rbf_fd <case> <Lx> <Ly>
!
! <case> is the path prefix of a points/graph pair of k-nearest-neighbour
! stencils on the periodic box [0, Lx) x [0, Ly), such as
! data/poisson_32_21; CMake passes it. The first two groups of checks
! need no input: the basis derivatives against central differences,
! and the exactness of one stencil's weights on polynomials. The third
! assembles the derivative and interpolation operators of the whole
! case through the callbacks and applies them to a smooth periodic
! field.
module test_rbf_fd_cases
use, intrinsic :: iso_c_binding, only: c_loc, c_intptr_t
use rbf_precision, only: wp, pi
use rbf_periodic_box, only: periodic_box
use rbf_io, only: read_points, read_graph_csr
use rbf_fd
implicit none
private

public :: nfail
public :: test_basis, test_one_stencil, test_assembly

integer :: nfail = 0

! The case of test_assembly, which its callbacks see: nodes, their
! k-nearest-neighbour stencils (1-based, k per node), the box, and
! the CSR values the two assemblies produce
integer :: npts, k
real(wp), allocatable :: x(:), y(:)
integer, allocatable :: ja(:)
type(periodic_box) :: box
real(wp), allocatable :: va(:,:), va2(:,:)

contains

    subroutine check(ok, what, err)
        logical, intent(in) :: ok
        character(*), intent(in) :: what
        real(wp), intent(in), optional :: err
        if (ok) return
        nfail = nfail + 1
        if (present(err)) then
            print '(a,a,a,es10.2)', "FAIL: ", what, ", error ", err
        else
            print '(a,a)', "FAIL: ", what
        end if
    end subroutine

    ! A stencil of 21 nodes on three rings, the origin among them
    subroutine ring_stencil(xs, ys)
        real(wp), intent(out) :: xs(21), ys(21)
        integer :: l
        xs(1) = 0
        ys(1) = 0
        do l = 1, 7
            xs(1 + l) = cos(2*pi*l/7)
            ys(1 + l) = sin(2*pi*l/7)
        end do
        do l = 1, 13
            xs(8 + l) = 2*cos(2*pi*l/13 + 0.3_wp)
            ys(8 + l) = 2*sin(2*pi*l/13 + 0.3_wp)
        end do
    end subroutine

    ! The derivative operators of the basis against central differences
    ! of its values, for every PHS exponent the header instantiates
    subroutine test_basis()
        integer, parameter :: n = 21, p = 3
        real(wp), parameter :: h = 1.0e-3_wp, tol = 1.0e-5_wp
        real(wp) :: xs(n), ys(n), xc, yc, scale, err
        real(wp), allocatable :: b(:), fd(:), fpp(:), fpm(:), fmp(:), fmm(:), f0(:)
        type(rbf_fd_workspace) :: ws
        integer :: q, nt
        character(32) :: label

        call ring_stencil(xs, ys)
        nt = n + npoly(p)
        allocate(b(nt), fd(nt), fpp(nt), fpm(nt), fmp(nt), fmm(nt), f0(nt))

        do q = 3, 7, 2
            ws = rbf_fd_workspace(q, p, n, 1)

            ! Off the nodes, where every basis function is smooth
            xc = 0.3_wp
            yc = -0.2_wp
            call rbf_fd_basis(ws, n, xs, ys, OP_VALUE, xc, yc, f0)
            scale = maxval(abs(f0))

            call rbf_fd_basis(ws, n, xs, ys, OP_VALUE, xc + h, yc, fpp)
            call rbf_fd_basis(ws, n, xs, ys, OP_VALUE, xc - h, yc, fmm)
            call rbf_fd_basis(ws, n, xs, ys, OP_DX, xc, yc, b)
            fd = (fpp - fmm)/(2*h)
            err = maxval(abs(b - fd))/scale
            write (label, '(a,i0)') "d/dx of the basis, q = ", q
            call check(err < tol, trim(label), err)

            call rbf_fd_basis(ws, n, xs, ys, OP_DXX, xc, yc, b)
            fd = (fpp - 2*f0 + fmm)/h**2
            err = maxval(abs(b - fd))/scale
            write (label, '(a,i0)') "d2/dx2 of the basis, q = ", q
            call check(err < tol, trim(label), err)

            call rbf_fd_basis(ws, n, xs, ys, OP_VALUE, xc, yc + h, fpp)
            call rbf_fd_basis(ws, n, xs, ys, OP_VALUE, xc, yc - h, fmm)
            call rbf_fd_basis(ws, n, xs, ys, OP_DY, xc, yc, b)
            fd = (fpp - fmm)/(2*h)
            err = maxval(abs(b - fd))/scale
            write (label, '(a,i0)') "d/dy of the basis, q = ", q
            call check(err < tol, trim(label), err)

            call rbf_fd_basis(ws, n, xs, ys, OP_DYY, xc, yc, b)
            fd = (fpp - 2*f0 + fmm)/h**2
            err = maxval(abs(b - fd))/scale
            write (label, '(a,i0)') "d2/dy2 of the basis, q = ", q
            call check(err < tol, trim(label), err)

            call rbf_fd_basis(ws, n, xs, ys, OP_VALUE, xc + h, yc + h, fpp)
            call rbf_fd_basis(ws, n, xs, ys, OP_VALUE, xc + h, yc - h, fpm)
            call rbf_fd_basis(ws, n, xs, ys, OP_VALUE, xc - h, yc + h, fmp)
            call rbf_fd_basis(ws, n, xs, ys, OP_VALUE, xc - h, yc - h, fmm)
            call rbf_fd_basis(ws, n, xs, ys, OP_DXY, xc, yc, b)
            fd = (fpp - fpm - fmp + fmm)/(4*h**2)
            err = maxval(abs(b - fd))/scale
            write (label, '(a,i0)') "d2/dxdy of the basis, q = ", q
            call check(err < tol, trim(label), err)

            ! The Laplacian is the sum of the two second derivatives
            call rbf_fd_basis(ws, n, xs, ys, OP_LAPLACE, xc, yc, b)
            call rbf_fd_basis(ws, n, xs, ys, OP_DXX, xc, yc, fpp)
            call rbf_fd_basis(ws, n, xs, ys, OP_DYY, xc, yc, fmm)
            err = maxval(abs(b - fpp - fmm))/scale
            write (label, '(a,i0)') "Laplacian of the basis, q = ", q
            call check(err < 1.0e-12_wp, trim(label), err)

            ! At its own node a PHS and all its derivatives vanish
            call rbf_fd_basis(ws, n, xs, ys, OP_DXX, xs(4), ys(4), b)
            call check(b(4) == 0, "d2/dx2 of phi at its own node is zero")
            call rbf_fd_basis(ws, n, xs, ys, OP_DX, xs(4), ys(4), b)
            call check(b(4) == 0, "d/dx of phi at its own node is zero")
        end do

    end subroutine

    ! One stencil: the weights reproduce every monomial of degree <= p
    ! exactly, interpolation at a node is the unit vector, and both
    ! factorizations give the same weights
    subroutine test_one_stencil()
        integer, parameter :: n = 21, p = 3, nrhs = 8
        real(wp), parameter :: tol = 1.0e-9_wp
        real(wp) :: xs(n), ys(n), w(n, nrhs), w2(n, nrhs), xc(nrhs), yc(nrhs)
        real(wp), allocatable :: lm(:), m(:,:)
        integer :: op(nrhs), info, j, l, i, np, solver
        type(rbf_fd_workspace), target :: ws, ws2
        real(wp) :: err
        character(64) :: label
        integer(c_intptr_t) :: addr

        call ring_stencil(xs, ys)

        ! Derivatives at the node, interpolation at two points, one of
        ! them node 5 itself
        op = [OP_DX, OP_DY, OP_DXX, OP_DXY, OP_DYY, OP_LAPLACE, OP_VALUE, OP_VALUE]
        xc = 0
        yc = 0
        xc(7) = 0.4_wp
        yc(7) = -0.7_wp
        xc(8) = xs(5)
        yc(8) = ys(5)

        ws = rbf_fd_workspace(q=3, p=p, nmax=n, nrhs_max=nrhs, solver=SOLVER_LU)
        call rbf_fd_weights(ws, n, xs, ys, nrhs, op, xc, yc, info)
        call check(info == 0, "LU solve of the ring stencil succeeds")
        ! The storage contract of the reclu kernels, met by every solver
        addr = transfer(c_loc(ws%abuf(ws%a0)), addr)
        call check(mod(addr, 32_c_intptr_t) == 0, "the matrix starts on a 32-byte boundary")
        call check(mod(ws%lda, 8) == 0 .and. ws%lda >= n + npoly(p), "the leading dimension is padded")
        w = ws%B(1:n, 1:nrhs)

        ! m(i, l) = m_i(x_l): the polynomial rows of the basis, asked
        ! with a "stencil" of no nodes
        np = npoly(p)
        allocate(m(np, n), lm(np))
        do l = 1, n
            call rbf_fd_basis(ws, 0, xs, ys, OP_VALUE, xs(l), ys(l), m(:, l))
        end do

        ! sum_l w(l, j) m_i(x_l) against L m_i(xc_j)
        do j = 1, nrhs
            call rbf_fd_basis(ws, 0, xs, ys, op(j), xc(j), yc(j), lm)
            do i = 1, np
                err = abs(dot_product(w(:, j), m(i, :)) - lm(i))
                write (label, '(a,i0,a,i0)') "reproduction of monomial ", i, " by operator ", j
                call check(err < tol, trim(label), err)
            end do
        end do

        ! Interpolation at node 5: the unit vector e_5
        err = maxval(abs(w(:, 8) - merge(1.0_wp, 0.0_wp, [(l == 5, l=1, n)])))
        call check(err < tol, "interpolation at a node is the unit vector", err)

        ! The other factorizations agree with LU
        do solver = SOLVER_LDLT, SOLVER_RECLU_LDLT
            ws2 = rbf_fd_workspace(q=3, p=p, nmax=n, nrhs_max=nrhs, solver=solver)
            call rbf_fd_weights(ws2, n, xs, ys, nrhs, op, xc, yc, info)
            call check(info == 0, trim(solver_name(solver))//" solve of the ring stencil succeeds")
            addr = transfer(c_loc(ws2%abuf(ws2%a0)), addr)
            call check(mod(addr, 32_c_intptr_t) == 0, "the matrix starts on a 32-byte boundary")
            w2 = ws2%B(1:n, 1:nrhs)
            err = maxval(abs(w2 - w))/maxval(abs(w))
            call check(err < tol, trim(solver_name(solver))//" and LU weights agree", err)
        end do

    end subroutine

    ! The whole case: derivatives at the nodes and interpolation at an
    ! offset point, assembled into CSR values through the callbacks and
    ! applied to sin(kx x) cos(ky y)
    subroutine test_assembly()
        integer, parameter :: nrhs = 7
        integer :: nnz, info, j, solver
        real(wp) :: Lx, Ly, kx, ky, xo, yo, err(nrhs)
        real(wp), allocatable :: f(:), lf(:), ref(:)
        integer :: op(nrhs)
        real(wp) :: xc(nrhs), yc(nrhs)
        type(rbf_fd_workspace) :: ws
        character(256) :: arg
        character(len=*), parameter :: names(nrhs) = [character(len=8) :: &
            "d/dx", "d/dy", "d2/dx2", "d2/dxdy", "d2/dy2", "lap", "interp"]
        ! Loose bounds, about three times the errors of poisson_32_21
        ! with p = 3, q = 3, k = 21 (2e-3, 4e-2 and 1e-4); a wrong sign
        ! or a wrong formula in the right-hand sides fails them by
        ! orders of magnitude
        real(wp), parameter :: tol(nrhs) = [5.0e-3_wp, 5.0e-3_wp, 1.0e-1_wp, 1.0e-1_wp, &
                                            1.0e-1_wp, 1.0e-1_wp, 1.0e-3_wp]

        if (command_argument_count() < 3) then
            print '(a)', "usage: test_rbf_fd <case> <Lx> <Ly>"
            error stop
        end if
        call get_command_argument(1, arg)
        call read_case(trim(arg), nnz)
        call get_command_argument(2, arg)
        read (arg, *) Lx
        call get_command_argument(3, arg)
        read (arg, *) Ly
        box = periodic_box(Lx, Ly)

        print '(a,i0,a,i0,a,i0)', "case: ", npts, " nodes, ", nnz, " nonzeros, k = ", k

        ! The operators: the derivatives at the node, interpolation at
        ! an offset (xo, yo) from it
        xo = 0.3_wp
        yo = -0.2_wp
        op = [OP_DX, OP_DY, OP_DXX, OP_DXY, OP_DYY, OP_LAPLACE, OP_VALUE]
        xc = 0
        yc = 0
        xc(nrhs) = xo
        yc(nrhs) = yo

        allocate(va(nnz, nrhs), va2(nnz, nrhs))

        ws = rbf_fd_workspace(q=3, p=3, nmax=k, nrhs_max=nrhs, solver=SOLVER_LU)
        call rbf_fd_assemble(ws, npts, nrhs, op, xc, yc, gather, scatter, info)
        call check(info == 0, "assembly of the case with LU succeeds")

        ! The smoothest field of the box, mode (1, 1): with ~1.1 node
        ! spacing the case resolves a wavelength of 32 to a few percent
        ! in the second derivatives
        kx = 2*pi/Lx
        ky = 2*pi/Ly
        allocate(f(npts), lf(npts), ref(npts))
        f = sin(kx*x)*cos(ky*y)

        do j = 1, nrhs
            select case (op(j))
            case (OP_DX)
                ref = kx*cos(kx*x)*cos(ky*y)
            case (OP_DY)
                ref = -ky*sin(kx*x)*sin(ky*y)
            case (OP_DXX)
                ref = -kx**2*f
            case (OP_DXY)
                ref = -kx*ky*cos(kx*x)*sin(ky*y)
            case (OP_DYY)
                ref = -ky**2*f
            case (OP_LAPLACE)
                ref = -(kx**2 + ky**2)*f
            case (OP_VALUE)
                ref = sin(kx*(x + xo))*cos(ky*(y + yo))
            end select
            call apply(va(:, j), f, lf)
            err(j) = maxval(abs(lf - ref))/maxval(abs(ref))
            print '(a,a8,a,es10.2)', "  ", names(j), " relative error ", err(j)
            call check(err(j) < tol(j), "accuracy of "//trim(names(j)), err(j))
        end do

        ! The other solvers assemble the same matrices
        do solver = SOLVER_LDLT, SOLVER_RECLU_LDLT
            ws = rbf_fd_workspace(q=3, p=3, nmax=k, nrhs_max=nrhs, solver=solver)
            va2 = 0
            call rbf_fd_assemble(ws, npts, nrhs, op, xc, yc, gather, scatter2, info)
            call check(info == 0, "assembly of the case with "//trim(solver_name(solver))//" succeeds")
            err(1) = maxval(abs(va2 - va))/maxval(abs(va))
            call check(err(1) < 1.0e-9_wp, trim(solver_name(solver))//" and LU assemblies agree", err(1))
        end do

    end subroutine

    ! The callbacks, recursive because rbf_fd_assemble's threads call
    ! them concurrently. The nodes of stencil s as minimum-image
    ! displacements from node s:
    recursive subroutine gather(s, nmax, n, xs, ys)
        integer, intent(in) :: s, nmax
        integer, intent(out) :: n
        real(wp), intent(out) :: xs(nmax), ys(nmax)
        integer :: l, col
        real(wp) :: d(2)
        n = k
        do l = 1, k
            col = ja((s - 1)*k + l)
            d = box%minimum_image([x(col) - x(s), y(col) - y(s)])
            xs(l) = d(1)
            ys(l) = d(2)
        end do
    end subroutine

    ! Row s of the CSR values, one column per operator
    recursive subroutine scatter(s, n, nrhs, w, ldw)
        integer, intent(in) :: s, n, nrhs, ldw
        real(wp), intent(in) :: w(ldw, *)
        va((s - 1)*k + 1:s*k, 1:nrhs) = w(1:n, 1:nrhs)
    end subroutine

    recursive subroutine scatter2(s, n, nrhs, w, ldw)
        integer, intent(in) :: s, n, nrhs, ldw
        real(wp), intent(in) :: w(ldw, *)
        va2((s - 1)*k + 1:s*k, 1:nrhs) = w(1:n, 1:nrhs)
    end subroutine

    ! y = A f for the fixed-row-length CSR values a
    subroutine apply(a, f, lf)
        real(wp), intent(in) :: a(:), f(:)
        real(wp), intent(out) :: lf(:)
        integer :: s, l
        do s = 1, npts
            lf(s) = 0
            do l = (s - 1)*k + 1, s*k
                lf(s) = lf(s) + a(l)*f(ja(l))
            end do
        end do
    end subroutine

    ! <case>.points and <case>.graph through rbf_io; the test wants
    ! k-nearest-neighbour stencils and 1-based indices
    subroutine read_case(prefix, nnz)
        character(*), intent(in) :: prefix
        integer, intent(out) :: nnz
        integer :: n2
        integer, allocatable :: ia(:), ja0(:)

        call read_points(prefix//".points", npts, x, y)
        call read_graph_csr(prefix//".graph", n2, ia, ja0)
        if (n2 /= npts) error stop "points and graph disagree on the node count"
        nnz = ia(npts)
        if (mod(nnz, npts) /= 0) error stop "the test expects k-nearest-neighbour stencils"
        k = nnz/npts
        if (any(ia(1:npts) - ia(0:npts - 1) /= k)) error stop "the test expects rows of equal length"
        allocate(ja(nnz))
        ja = ja0 + 1
    end subroutine

end module

program test_rbf_fd
use test_rbf_fd_cases
implicit none

call test_basis()
call test_one_stencil()
call test_assembly()

if (nfail > 0) then
    print '(a,i0,a)', "test_rbf_fd: ", nfail, " check(s) failed"
    error stop
end if
print '(a)', "test_rbf_fd: all checks passed"

end program
