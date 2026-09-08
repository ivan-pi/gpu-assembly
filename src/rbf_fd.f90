! RBF-FD weights on the host: polyharmonic splines phi(r) = r^q, q odd,
! augmented with the 2-D monomials of degree <= p, one small dense
! saddle-point system per stencil,
!
!     [ A    P ] [ w      ]   [ L phi ]     A(k,l) = phi(|x_l - x_k|)
!     [ P^T  0 ] [ lambda ] = [ L m   ]     P(k,i) = m_i(x_k)
!
! solved with LAPACK. The right-hand side is the operator L applied to
! the basis at an evaluation point; several right-hand sides, one per
! operator, share the factorization. The C++/CUDA twin is
! rbf_operators.h, which fills the same matrix (same monomial order,
! same q) and solves it with cuSolverDx; this module serves the host,
! the tests and the verification of the GPU kernels.
!
! The coordinates handed in are stencil-local: shifted so that the
! stencil's own node sits at the origin, with periodic images already
! resolved (periodic_box%minimum_image). An operator is a code
! (OP_VALUE, OP_DX, ..., OP_LAPLACE) and the point where it is taken
! in that frame: derivatives at the node are taken at (0, 0), an
! interpolation weight at any point of the stencil's footprint. The
! trailing np polynomial coefficients lambda are discarded.
!
!     ws = rbf_fd_workspace(q=3, p=3, nmax=21, nrhs_max=4)
!     call rbf_fd_weights(ws, n, x, y, nrhs, op, xc, yc, w, info)
!
! gives the n x nrhs weights of one stencil, and
!
!     call rbf_fd_assemble(ws, nstencils, nrhs, op, xc, yc, gather, scatter, info)
!
! runs over all stencils in parallel, asking gather(s, ...) for the
! stencil-local nodes of stencil s and handing its weights to
! scatter(s, ...), which stores them in whatever sparse layout the
! caller keeps. Internal procedures serve as the callbacks; they see
! the caller's arrays by host association.
module rbf_fd
use rbf_precision, only: wp
implicit none
private

public :: rbf_fd_workspace
public :: rbf_fd_basis, rbf_fd_weights, rbf_fd_assemble
public :: rbf_fd_gather, rbf_fd_scatter
public :: npoly

public :: OP_VALUE, OP_DX, OP_DY, OP_DXX, OP_DXY, OP_DYY, OP_LAPLACE
public :: SOLVER_LU, SOLVER_LDLT

! The operators the right-hand sides can carry
integer, parameter :: OP_VALUE = 0      ! interpolation
integer, parameter :: OP_DX = 1, OP_DY = 2
integer, parameter :: OP_DXX = 3, OP_DXY = 4, OP_DYY = 5
integer, parameter :: OP_LAPLACE = 6

! Factorization of the saddle-point matrix: LU with partial pivoting
! (dgesv), or the Bunch-Kaufman LDL^T of the symmetric indefinite
! matrix (dsytrf/dsytrs), which does half the flops.
integer, parameter :: SOLVER_LU = 1, SOLVER_LDLT = 2

! The parameters of the approximation and the work arrays of one
! stencil solve: sized once by the constructor for stencils of up to
! nmax nodes and nrhs_max right-hand sides, then reused. Not shared
! between threads; rbf_fd_assemble gives each thread a copy.
type :: rbf_fd_workspace
    integer :: q = 3            ! PHS exponent, phi(r) = r^q
    integer :: p = 2            ! degree of the polynomial augmentation
    integer :: np = 6           ! number of monomials, npoly(p)
    integer :: nmax = 0         ! largest stencil
    integer :: nrhs_max = 0     ! most right-hand sides
    integer :: solver = SOLVER_LU
    integer, allocatable :: ix(:), iy(:)    ! exponents of the monomials
    real(wp), allocatable :: A(:,:), B(:,:) ! the system and its right-hand sides
    integer, allocatable :: ipiv(:)
    real(wp), allocatable :: work(:)        ! dsytrf's workspace
end type

interface rbf_fd_workspace
    module procedure :: rbf_fd_workspace_constructor
end interface

abstract interface
    ! The nodes of stencil s, at most nmax of them, in the
    ! stencil-local frame: x(1:n), y(1:n) are the displacements from
    ! the stencil's node, periodic images resolved.
    subroutine rbf_fd_gather(s, nmax, n, x, y)
        import wp
        integer, intent(in) :: s, nmax
        integer, intent(out) :: n
        real(wp), intent(out) :: x(nmax), y(nmax)
    end subroutine

    ! The weights of stencil s: w(k, j) multiplies the value at the
    ! k-th node gather delivered, for the j-th operator.
    subroutine rbf_fd_scatter(s, n, nrhs, w)
        import wp
        integer, intent(in) :: s, n, nrhs
        real(wp), intent(in) :: w(:,:)
    end subroutine
end interface

! LAPACK, double precision
interface
    subroutine dgesv(n, nrhs, a, lda, ipiv, b, ldb, info)
        import wp
        integer, intent(in) :: n, nrhs, lda, ldb
        real(wp), intent(inout) :: a(lda,*), b(ldb,*)
        integer, intent(out) :: ipiv(*), info
    end subroutine
    subroutine dsytrf(uplo, n, a, lda, ipiv, work, lwork, info)
        import wp
        character, intent(in) :: uplo
        integer, intent(in) :: n, lda, lwork
        real(wp), intent(inout) :: a(lda,*)
        integer, intent(out) :: ipiv(*), info
        real(wp), intent(out) :: work(*)
    end subroutine
    subroutine dsytrs(uplo, n, nrhs, a, lda, ipiv, b, ldb, info)
        import wp
        character, intent(in) :: uplo
        integer, intent(in) :: n, nrhs, lda, ldb
        real(wp), intent(in) :: a(lda,*)
        integer, intent(in) :: ipiv(*)
        real(wp), intent(inout) :: b(ldb,*)
        integer, intent(out) :: info
    end subroutine
end interface

contains

    ! Number of 2-D monomials of degree <= p
    elemental function npoly(p) result(np)
        integer, intent(in) :: p
        integer :: np
        np = (p + 1)*(p + 2)/2
    end function

    function rbf_fd_workspace_constructor(q, p, nmax, nrhs_max, solver) result(ws)
        integer, intent(in) :: q, p, nmax, nrhs_max
        integer, intent(in), optional :: solver
        type(rbf_fd_workspace) :: ws

        integer :: d, j, k, nt, info
        real(wp) :: query(1)

        if (q < 3 .or. mod(q, 2) == 0) then
            error stop "rbf_fd: the PHS exponent q must be an odd integer >= 3"
        end if
        if (p < 0) error stop "rbf_fd: the polynomial degree p must not be negative"
        if (nrhs_max < 1) error stop "rbf_fd: nrhs_max must be at least 1"

        ws%q = q
        ws%p = p
        ws%np = npoly(p)
        ws%nmax = nmax
        ws%nrhs_max = nrhs_max
        if (present(solver)) ws%solver = solver
        if (ws%solver /= SOLVER_LU .and. ws%solver /= SOLVER_LDLT) then
            error stop "rbf_fd: solver must be SOLVER_LU or SOLVER_LDLT"
        end if

        ! Unisolvency: P must have full column rank, so at least np nodes
        if (nmax < ws%np) then
            error stop "rbf_fd: a stencil needs at least npoly(p) nodes"
        end if

        ! The monomials x^i y^j by total degree, the order of
        ! PolyBasis<P> in rbf_operators.h: 1, x, y, x^2, xy, y^2, ...
        allocate(ws%ix(ws%np), ws%iy(ws%np))
        k = 0
        do d = 0, p
            do j = 0, d
                k = k + 1
                ws%ix(k) = d - j
                ws%iy(k) = j
            end do
        end do

        nt = nmax + ws%np
        allocate(ws%A(nt, nt), ws%B(nt, nrhs_max), ws%ipiv(nt))

        ! dsytrf's blocked factorization wants a workspace whose size
        ! only it knows; ask once, for the largest system
        if (ws%solver == SOLVER_LDLT) then
            call dsytrf('L', nt, ws%A, nt, ws%ipiv, query, -1, info)
            if (info /= 0) error stop "rbf_fd: dsytrf workspace query failed"
            allocate(ws%work(max(1, int(query(1)))))
        end if

    end function

    ! The basis, with the operator op applied, at the point (xc, yc) of
    ! the stencil-local frame:
    !
    !     b(k)    = L phi(|(xc, yc) - (x(k), y(k))|),   k = 1, n
    !     b(n+i)  = L m_i(xc, yc),                      i = 1, np
    !
    ! With phi = r^q, q odd, and d = (xc, yc) - x_k:
    !
    !     phi_x    = q r^(q-2) dx
    !     phi_xx   = q r^(q-4) ((q-1) dx^2 + dy^2)
    !     phi_xy   = q (q-2) r^(q-4) dx dy
    !     lap phi  = q^2 r^(q-2)
    !
    ! all of which vanish at r = 0 for q >= 3.
    subroutine rbf_fd_basis(ws, n, x, y, op, xc, yc, b)
        type(rbf_fd_workspace), intent(in) :: ws
        integer, intent(in) :: n
        real(wp), intent(in) :: x(n), y(n)
        integer, intent(in) :: op
        real(wp), intent(in) :: xc, yc
        real(wp), intent(out) :: b(*)

        real(wp) :: dx, dy, r2, r, rq3, q
        real(wp) :: xp(0:ws%p), yp(0:ws%p)
        integer :: k, i, j

        q = real(ws%q, wp)

        do k = 1, n
            dx = xc - x(k)
            dy = yc - y(k)
            r2 = dx*dx + dy*dy
            r = sqrt(r2)
            rq3 = r2**((ws%q - 3)/2)    ! r^(q-3), an integer power of r^2

            select case (op)
            case (OP_VALUE)
                b(k) = rq3*r2*r
            case (OP_DX)
                b(k) = q*rq3*r*dx
            case (OP_DY)
                b(k) = q*rq3*r*dy
            case (OP_LAPLACE)
                b(k) = q*q*rq3*r
            case (OP_DXX, OP_DXY, OP_DYY)
                if (r2 > 0) then
                    select case (op)
                    case (OP_DXX)
                        b(k) = q*rq3*((q - 1)*dx*dx + dy*dy)/r
                    case (OP_DXY)
                        b(k) = q*(q - 2)*rq3*dx*dy/r
                    case (OP_DYY)
                        b(k) = q*rq3*(dx*dx + (q - 1)*dy*dy)/r
                    end select
                else
                    b(k) = 0
                end if
            case default
                error stop "rbf_fd: unknown operator code"
            end select
        end do

        ! Powers of the evaluation point, then the monomials x^i y^j
        xp(0) = 1
        yp(0) = 1
        do k = 1, ws%p
            xp(k) = xp(k - 1)*xc
            yp(k) = yp(k - 1)*yc
        end do

        do k = 1, ws%np
            i = ws%ix(k)
            j = ws%iy(k)
            select case (op)
            case (OP_VALUE)
                b(n + k) = xp(i)*yp(j)
            case (OP_DX)
                b(n + k) = i*mono(i - 1, j)
            case (OP_DY)
                b(n + k) = j*mono(i, j - 1)
            case (OP_DXX)
                b(n + k) = i*(i - 1)*mono(i - 2, j)
            case (OP_DXY)
                b(n + k) = i*j*mono(i - 1, j - 1)
            case (OP_DYY)
                b(n + k) = j*(j - 1)*mono(i, j - 2)
            case (OP_LAPLACE)
                b(n + k) = i*(i - 1)*mono(i - 2, j) + j*(j - 1)*mono(i, j - 2)
            end select
        end do

    contains

        ! x^i y^j at the evaluation point, zero for a negative exponent
        ! (whose coefficient, i or i-1, is zero anyway)
        pure function mono(i, j)
            integer, intent(in) :: i, j
            real(wp) :: mono
            if (i < 0 .or. j < 0) then
                mono = 0
            else
                mono = xp(i)*yp(j)
            end if
        end function

    end subroutine

    ! The weights of one stencil of n nodes at (x(k), y(k)), k = 1, n,
    ! in the stencil-local frame, for the nrhs operators op(j) taken at
    ! (xc(j), yc(j)): on return w(k, j), k = 1, n, is the weight of
    ! node k for operator j. w may be larger than n x nrhs; only that
    ! block is written. info is LAPACK's: 0 on success, > 0 if the
    ! matrix is singular (coincident nodes, or too few of them for
    ! the polynomial degree).
    subroutine rbf_fd_weights(ws, n, x, y, nrhs, op, xc, yc, w, info)
        type(rbf_fd_workspace), intent(inout) :: ws
        integer, intent(in) :: n
        real(wp), intent(in) :: x(n), y(n)
        integer, intent(in) :: nrhs
        integer, intent(in) :: op(nrhs)
        real(wp), intent(in) :: xc(nrhs), yc(nrhs)
        real(wp), intent(out) :: w(:,:)
        integer, intent(out) :: info

        integer :: k, nt, lda

        if (n > ws%nmax) error stop "rbf_fd: stencil larger than the workspace's nmax"
        if (n < ws%np) error stop "rbf_fd: a stencil needs at least npoly(p) nodes"
        if (nrhs > ws%nrhs_max) error stop "rbf_fd: more right-hand sides than nrhs_max"
        if (size(w, 1) < n .or. size(w, 2) < nrhs) error stop "rbf_fd: w is too small"

        nt = n + ws%np
        lda = size(ws%A, 1)

        ! Column l of the value basis at node l is both A(:, l) and,
        ! in its polynomial rows, column l of P^T
        do k = 1, n
            call rbf_fd_basis(ws, n, x, y, OP_VALUE, x(k), y(k), ws%A(:, k))
        end do
        ws%A(1:n, n + 1:nt) = transpose(ws%A(n + 1:nt, 1:n))
        ws%A(n + 1:nt, n + 1:nt) = 0

        do k = 1, nrhs
            call rbf_fd_basis(ws, n, x, y, op(k), xc(k), yc(k), ws%B(:, k))
        end do

        select case (ws%solver)
        case (SOLVER_LU)
            call dgesv(nt, nrhs, ws%A, lda, ws%ipiv, ws%B, lda, info)
        case (SOLVER_LDLT)
            call dsytrf('L', nt, ws%A, lda, ws%ipiv, ws%work, size(ws%work), info)
            if (info == 0) then
                call dsytrs('L', nt, nrhs, ws%A, lda, ws%ipiv, ws%B, lda, info)
            end if
        end select
        if (info /= 0) return

        w(1:n, 1:nrhs) = ws%B(1:n, 1:nrhs)

    end subroutine

    ! The weights of every stencil, through the two callbacks: gather
    ! delivers the stencil-local nodes of stencil s, scatter receives
    ! its n x nrhs weights. All stencils share the operators op(j) at
    ! (xc(j), yc(j)). The loop is parallel over stencils, each thread
    ! with its own copy of ws, so scatter must be safe to call from
    ! several threads at once, which it is when stencil s writes only
    ! its own row. info is 0, or the lowest index of a stencil whose
    ! system was singular; scatter is not called for those.
    subroutine rbf_fd_assemble(ws, nstencils, nrhs, op, xc, yc, gather, scatter, info)
        type(rbf_fd_workspace), intent(in) :: ws
        integer, intent(in) :: nstencils, nrhs
        integer, intent(in) :: op(nrhs)
        real(wp), intent(in) :: xc(nrhs), yc(nrhs)
        procedure(rbf_fd_gather) :: gather
        procedure(rbf_fd_scatter) :: scatter
        integer, intent(out) :: info

        integer :: first_bad

        first_bad = nstencils + 1

        !$omp parallel default(shared) reduction(min:first_bad)
        block
            type(rbf_fd_workspace) :: wsl
            real(wp), allocatable :: x(:), y(:), w(:,:)
            integer :: s, n, ierr

            wsl = ws
            allocate(x(ws%nmax), y(ws%nmax), w(ws%nmax, nrhs))

            !$omp do schedule(static)
            do s = 1, nstencils
                call gather(s, ws%nmax, n, x, y)
                call rbf_fd_weights(wsl, n, x, y, nrhs, op, xc, yc, w, ierr)
                if (ierr /= 0) then
                    first_bad = min(first_bad, s)
                else
                    call scatter(s, n, nrhs, w(1:n, 1:nrhs))
                end if
            end do
            !$omp end do
        end block
        !$omp end parallel

        info = 0
        if (first_bad <= nstencils) info = first_bad

    end subroutine

end module
