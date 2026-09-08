! RBF-FD weights on the host: polyharmonic splines phi(r) = r^q, q odd,
! augmented with the 2-D monomials of degree <= p, one small dense
! saddle-point system per stencil,
!
!     [ A    P ] [ w      ]   [ L phi ]     A(k,l) = phi(|x_l - x_k|)
!     [ P^T  0 ] [ lambda ] = [ L m   ]     P(k,i) = m_i(x_k)
!
! solved with LAPACK, or with the tuned small-matrix kernels of
! third_party/reclu. The right-hand side is the operator L applied to
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
!     call rbf_fd_weights(ws, n, x, y, nrhs, op, xc, yc, info)
!
! leaves the n x nrhs weights of one stencil in ws%B, and
!
!     call rbf_fd_assemble(ws, nstencils, nrhs, op, xc, yc, gather, scatter, info)
!
! runs over all stencils in parallel, asking gather(s, ...) for the
! stencil-local nodes of stencil s and handing its weights to
! scatter(s, ...), which stores them in whatever sparse layout the
! caller keeps. Internal procedures serve as the callbacks; they see
! the caller's arrays by host association.
!
! The per-stencil routines check nothing: the driver validates its
! arguments once, and the size of every stencil as gather returns it.
! They are declared recursive because the driver's OpenMP threads
! invoke them concurrently, which the standard allows only of
! recursive procedures; callbacks should be declared so too.
module rbf_fd
use, intrinsic :: iso_c_binding, only: c_int, c_double, c_char, c_loc, c_intptr_t
use rbf_precision, only: wp
implicit none
private

public :: rbf_fd_workspace
public :: rbf_fd_basis, rbf_fd_fill, rbf_fd_solve, rbf_fd_weights, rbf_fd_assemble
public :: rbf_fd_gather, rbf_fd_scatter
public :: npoly

public :: OP_VALUE, OP_DX, OP_DY, OP_DXX, OP_DXY, OP_DYY, OP_LAPLACE
public :: SOLVER_LU, SOLVER_LDLT, SOLVER_RECLU_LU, SOLVER_RECLU_LDLT, SOLVER_DEFAULT
public :: solver_name

! The operators the right-hand sides can carry
integer, parameter :: OP_VALUE = 0      ! interpolation
integer, parameter :: OP_DX = 1, OP_DY = 2
integer, parameter :: OP_DXX = 3, OP_DXY = 4, OP_DYY = 5
integer, parameter :: OP_LAPLACE = 6
integer, parameter :: OP_LAST = OP_LAPLACE

! Factorization of the saddle-point matrix, which is symmetric
! indefinite: LU with partial pivoting, or the Bunch-Kaufman LDL^T,
! which does half the flops and needs only the lower triangle filled.
! Each from LAPACK (the unblocked dgetf2 with dgetrs, since at these
! sizes dgesv's recursive dgetrf2 is markedly slower; dsytrf with
! dsytrs) or from reclu, whose AVX2 kernels are tuned for exactly
! these sizes and fall back to plain loops elsewhere. Timings in
! docs/rbf_fd.md.
integer, parameter :: SOLVER_LU = 1, SOLVER_LDLT = 2
integer, parameter :: SOLVER_RECLU_LU = 3, SOLVER_RECLU_LDLT = 4
integer, parameter :: SOLVER_DEFAULT = SOLVER_RECLU_LDLT

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
    integer :: solver = SOLVER_DEFAULT
    integer :: lda = 0          ! leading dimension of A and B, nmax + np padded to 8
    ! The monomials m_k = x^ix(k) y^iy(k), by total degree, and the
    ! recurrence that evaluates them: m_1 = 1, m_k = t * m_parent(k)
    ! with t = x for axis(k) = 1 and t = y for axis(k) = 2
    integer, allocatable :: ix(:), iy(:), parent(:), axis(:)
    ! The operators on the monomials, two terms each (only the
    ! Laplacian needs both): L m_k = sum_t pc(k,t,op) x^px(k,t,op) y^py(k,t,op)
    real(wp), allocatable :: pc(:,:,:)
    integer, allocatable :: px(:,:,:), py(:,:,:)
    ! The system, lda x lda column-major from abuf(a0) on, with a0
    ! chosen at every fill so that the matrix starts on a 32-byte
    ! boundary (the reclu kernels are store-bound and want it); the
    ! right-hand sides, lda x nrhs_max, after a solve holding the
    ! weights in B(1:n, 1:nrhs)
    real(wp), allocatable :: abuf(:)
    integer :: a0 = 1
    real(wp), allocatable :: B(:,:)
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
    ! k-th node gather delivered, for the j-th operator. The array is
    ! the workspace's B, of leading dimension ldw.
    subroutine rbf_fd_scatter(s, n, nrhs, w, ldw)
        import wp
        integer, intent(in) :: s, n, nrhs, ldw
        real(wp), intent(in) :: w(ldw, *)
    end subroutine
end interface

! LAPACK, double precision
interface
    subroutine dgetf2(m, n, a, lda, ipiv, info)
        import wp
        integer, intent(in) :: m, n, lda
        real(wp), intent(inout) :: a(lda,*)
        integer, intent(out) :: ipiv(*), info
    end subroutine
    subroutine dgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info)
        import wp
        character, intent(in) :: trans
        integer, intent(in) :: n, nrhs, lda, ldb
        real(wp), intent(in) :: a(lda,*)
        integer, intent(in) :: ipiv(*)
        real(wp), intent(inout) :: b(ldb,*)
        integer, intent(out) :: info
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

! reclu (third_party/reclu/reclu.h): LAPACK conventions, 1-based ipiv
interface
    subroutine reclu_lu_opt(m, n, a, lda, ipiv, info) bind(c, name="reclu_lu_opt")
        import c_int, c_double
        integer(c_int), value :: m, n, lda
        real(c_double), intent(inout) :: a(lda,*)
        integer(c_int), intent(out) :: ipiv(*), info
    end subroutine
    subroutine reclu_solve_opt(m, nrhs, lu, lda, ipiv, b, ldb) bind(c, name="reclu_solve_opt")
        import c_int, c_double
        integer(c_int), value :: m, nrhs, lda, ldb
        real(c_double), intent(in) :: lu(lda,*)
        integer(c_int), intent(in) :: ipiv(*)
        real(c_double), intent(inout) :: b(ldb,*)
    end subroutine
    subroutine reclu_sytf2_opt(uplo, n, a, lda, ipiv, info) bind(c, name="reclu_sytf2_opt")
        import c_int, c_double, c_char
        character(kind=c_char), value :: uplo
        integer(c_int), value :: n, lda
        real(c_double), intent(inout) :: a(lda,*)
        integer(c_int), intent(out) :: ipiv(*), info
    end subroutine
    subroutine reclu_sytrs_opt(uplo, n, nrhs, a, lda, ipiv, b, ldb) bind(c, name="reclu_sytrs_opt")
        import c_int, c_double, c_char
        character(kind=c_char), value :: uplo
        integer(c_int), value :: n, nrhs, lda, ldb
        real(c_double), intent(in) :: a(lda,*)
        integer(c_int), intent(in) :: ipiv(*)
        real(c_double), intent(inout) :: b(ldb,*)
    end subroutine
end interface

contains

    ! Number of 2-D monomials of degree <= p
    elemental function npoly(p) result(np)
        integer, intent(in) :: p
        integer :: np
        np = (p + 1)*(p + 2)/2
    end function

    function solver_name(solver) result(name)
        integer, intent(in) :: solver
        character(len=12) :: name
        select case (solver)
        case (SOLVER_LU)
            name = "LU"
        case (SOLVER_LDLT)
            name = "LDL^T"
        case (SOLVER_RECLU_LU)
            name = "reclu LU"
        case (SOLVER_RECLU_LDLT)
            name = "reclu LDL^T"
        case default
            name = "?"
        end select
    end function

    function rbf_fd_workspace_constructor(q, p, nmax, nrhs_max, solver) result(ws)
        integer, intent(in) :: q, p, nmax, nrhs_max
        integer, intent(in), optional :: solver
        type(rbf_fd_workspace) :: ws

        integer :: d, i, j, k, op, nt, info
        real(wp) :: query(1)

        if (q < 3 .or. mod(q, 2) == 0) then
            error stop "rbf_fd: the PHS exponent q must be an odd integer >= 3"
        end if
        if (p < 0) error stop "rbf_fd: the polynomial degree p must not be negative"
        if (nrhs_max < 1) error stop "rbf_fd: nrhs_max must be at least 1"
        if (kind(0) /= c_int) error stop "rbf_fd: the default integer is not C's int"

        ws%q = q
        ws%p = p
        ws%np = npoly(p)
        ws%nmax = nmax
        ws%nrhs_max = nrhs_max
        if (present(solver)) ws%solver = solver
        if (ws%solver < SOLVER_LU .or. ws%solver > SOLVER_RECLU_LDLT) then
            error stop "rbf_fd: unknown solver code"
        end if

        ! Unisolvency: P must have full column rank, so at least np nodes
        if (nmax < ws%np) then
            error stop "rbf_fd: a stencil needs at least npoly(p) nodes"
        end if
        nt = nmax + ws%np
        if (ws%solver >= SOLVER_RECLU_LU .and. nt > 128) then
            error stop "rbf_fd: the reclu solvers take systems of up to 128 rows"
        end if

        ! The monomials x^i y^j by total degree, the order of
        ! PolyBasis<P> in rbf_operators.h: 1, x, y, x^2, xy, y^2, ...
        ! Each but the first is x or y times an earlier one: x^i y^j
        ! comes from x^(i-1) y^j, or from y^(j-1) when i = 0.
        allocate(ws%ix(ws%np), ws%iy(ws%np), ws%parent(ws%np), ws%axis(ws%np))
        k = 0
        do d = 0, p
            do j = 0, d
                k = k + 1
                i = d - j
                ws%ix(k) = i
                ws%iy(k) = j
                if (d == 0) then
                    ws%parent(k) = 1
                    ws%axis(k) = 1
                else if (i > 0) then
                    ws%parent(k) = index_of(i - 1, j)
                    ws%axis(k) = 1
                else
                    ws%parent(k) = index_of(0, j - 1)
                    ws%axis(k) = 2
                end if
            end do
        end do

        ! The operators on the monomials, as coefficient and exponents
        ! of up to two terms; a term whose exponent would go negative
        ! has coefficient zero and is pointed at x^0 y^0
        allocate(ws%pc(ws%np, 2, 0:OP_LAST), ws%px(ws%np, 2, 0:OP_LAST), ws%py(ws%np, 2, 0:OP_LAST))
        ws%pc = 0
        ws%px = 0
        ws%py = 0
        do op = 0, OP_LAST
            do k = 1, ws%np
                i = ws%ix(k)
                j = ws%iy(k)
                select case (op)
                case (OP_VALUE)
                    call term(k, 1, op, 1, i, j)
                case (OP_DX)
                    call term(k, 1, op, i, i - 1, j)
                case (OP_DY)
                    call term(k, 1, op, j, i, j - 1)
                case (OP_DXX)
                    call term(k, 1, op, i*(i - 1), i - 2, j)
                case (OP_DXY)
                    call term(k, 1, op, i*j, i - 1, j - 1)
                case (OP_DYY)
                    call term(k, 1, op, j*(j - 1), i, j - 2)
                case (OP_LAPLACE)
                    call term(k, 1, op, i*(i - 1), i - 2, j)
                    call term(k, 2, op, j*(j - 1), i, j - 2)
                end select
            end do
        end do

        ! The leading dimension: the largest system, padded to a
        ! multiple of 8 doubles so every column starts on a 64-byte
        ! line once the matrix itself is aligned; the buffer has the
        ! slack that takes
        ws%lda = 8*((nt + 7)/8)
        allocate(ws%abuf(ws%lda*ws%lda + 3), ws%B(ws%lda, nrhs_max), ws%ipiv(ws%lda))
        ws%a0 = aligned_start(ws)

        ! dsytrf's blocked factorization wants a workspace whose size
        ! only it knows; ask once, for the largest system
        if (ws%solver == SOLVER_LDLT) then
            call dsytrf('L', nt, ws%abuf(ws%a0), ws%lda, ws%ipiv, query, -1, info)
            if (info /= 0) error stop "rbf_fd: dsytrf workspace query failed"
            allocate(ws%work(max(1, int(query(1)))))
        end if

    contains

        ! Position of x^i y^j in the ordering by total degree
        pure function index_of(i, j) result(k)
            integer, intent(in) :: i, j
            integer :: k
            k = (i + j)*(i + j + 1)/2 + j + 1
        end function

        ! Term t of operator op on monomial k: c x^i y^j
        subroutine term(k, t, op, c, i, j)
            integer, intent(in) :: k, t, op, c, i, j
            if (i < 0 .or. j < 0) return
            ws%pc(k, t, op) = c
            ws%px(k, t, op) = i
            ws%py(k, t, op) = j
        end subroutine

    end function

    ! Where in its buffer the matrix starts so as to sit on a 32-byte
    ! boundary. Recomputed by every fill from the buffer's current
    ! address, because a copy of the workspace (an assignment, the
    ! constructor's result landing in a variable, the driver's
    ! per-thread copies) puts the buffer at an address of its own.
    function aligned_start(ws) result(a0)
        type(rbf_fd_workspace), intent(in), target :: ws
        integer :: a0
        integer(c_intptr_t) :: addr
        addr = transfer(c_loc(ws%abuf(1)), addr)
        a0 = 1 + int(mod(4 - mod(addr/8, 4_c_intptr_t), 4_c_intptr_t))
    end function

    ! The PHS part of the basis, with the operator op applied, at the
    ! point (xc, yc) of the stencil-local frame:
    !
    !     b(k) = L phi(|(xc, yc) - (x(k), y(k))|),   k = 1, n
    !
    ! With phi = r^q, q odd, and d = (xc, yc) - x_k:
    !
    !     phi_x    = q r^(q-2) dx
    !     phi_xx   = q r^(q-4) ((q-1) dx^2 + dy^2)
    !     phi_xy   = q (q-2) r^(q-4) dx dy
    !     lap phi  = q^2 r^(q-2)
    !
    ! all of which vanish at r = 0 for q >= 3. The second derivatives
    ! divide by r^2, so there the divisor is clamped to the smallest
    ! normal number; the numerator is zero and so is the quotient.
    recursive subroutine phs_column(q, n, x, y, op, xc, yc, b)
        integer, intent(in) :: q, n, op
        real(wp), intent(in) :: x(n), y(n), xc, yc
        real(wp), intent(out) :: b(n)

        real(wp) :: dx(n), dy(n), r2(n), rp(n), qr
        integer :: k

        do k = 1, n
            dx(k) = xc - x(k)
            dy(k) = yc - y(k)
            r2(k) = dx(k)*dx(k) + dy(k)*dy(k)
        end do

        ! rp = r^(q-2): r for the cubic, else r times an integer power of r^2
        select case (q)
        case (3)
            rp = sqrt(r2)
        case (5)
            rp = sqrt(r2)*r2
        case default
            rp = sqrt(r2)*r2**((q - 3)/2)
        end select

        qr = real(q, wp)
        select case (op)
        case (OP_VALUE)
            b = rp*r2
        case (OP_DX)
            b = qr*rp*dx
        case (OP_DY)
            b = qr*rp*dy
        case (OP_DXX)
            b = qr*rp*((qr - 1)*dx*dx + dy*dy)/max(r2, tiny(qr))
        case (OP_DXY)
            b = qr*(qr - 2)*rp*dx*dy/max(r2, tiny(qr))
        case (OP_DYY)
            b = qr*rp*(dx*dx + (qr - 1)*dy*dy)/max(r2, tiny(qr))
        case (OP_LAPLACE)
            b = qr*qr*rp
        case default
            error stop "rbf_fd: unknown operator code"
        end select

    end subroutine

    ! The polynomial part of the basis with the operator op applied,
    ! at (xc, yc): b(k) = L m_k(xc, yc), k = 1, np, from the tables of
    ! the constructor
    recursive subroutine poly_column(ws, op, xc, yc, b)
        type(rbf_fd_workspace), intent(in) :: ws
        integer, intent(in) :: op
        real(wp), intent(in) :: xc, yc
        real(wp), intent(out) :: b(ws%np)

        real(wp) :: xp(0:ws%p), yp(0:ws%p)
        integer :: k

        xp(0) = 1
        yp(0) = 1
        do k = 1, ws%p
            xp(k) = xp(k - 1)*xc
            yp(k) = yp(k - 1)*yc
        end do

        do k = 1, ws%np
            b(k) = ws%pc(k, 1, op)*xp(ws%px(k, 1, op))*yp(ws%py(k, 1, op)) &
                 + ws%pc(k, 2, op)*xp(ws%px(k, 2, op))*yp(ws%py(k, 2, op))
        end do

    end subroutine

    ! The whole basis, with the operator op applied, at the point
    ! (xc, yc) of the stencil-local frame: b(1:n) the PHS part,
    ! b(n+1:n+np) the monomials. What the right-hand sides are made of.
    recursive subroutine rbf_fd_basis(ws, n, x, y, op, xc, yc, b)
        type(rbf_fd_workspace), intent(in) :: ws
        integer, intent(in) :: n
        real(wp), intent(in) :: x(n), y(n)
        integer, intent(in) :: op
        real(wp), intent(in) :: xc, yc
        real(wp), intent(out) :: b(*)

        call phs_column(ws%q, n, x, y, op, xc, yc, b)
        call poly_column(ws, op, xc, yc, b(n + 1))

    end subroutine

    ! The matrix of one stencil, into A: the PHS block, the monomial
    ! blocks and the zero block. For the LDL^T solvers only the lower
    ! triangle is filled.
    recursive subroutine fill_matrix(ws, n, x, y, A, lda)
        type(rbf_fd_workspace), intent(in) :: ws
        integer, intent(in) :: n, lda
        real(wp), intent(in) :: x(n), y(n)
        real(wp), intent(inout) :: A(lda, *)

        integer :: k, l, i, j, l0, nt, np, m
        real(wp) :: t(2), r2
        logical :: lower

        np = ws%np
        nt = n + np
        lower = ws%solver == SOLVER_LDLT .or. ws%solver == SOLVER_RECLU_LDLT

        ! The PHS block, column k = phi(|x_l - x_k|) for the nodes l
        ! from 1, or from k for the lower triangle only: one loop nest
        ! per exponent, r^q = r^2 ... r^2 sqrt(r^2)
        select case (ws%q)
        case (3)
            do k = 1, n
                l0 = merge(k, 1, lower)
                do l = l0, n
                    r2 = (x(l) - x(k))**2 + (y(l) - y(k))**2
                    A(l, k) = r2*sqrt(r2)
                end do
            end do
        case (5)
            do k = 1, n
                l0 = merge(k, 1, lower)
                do l = l0, n
                    r2 = (x(l) - x(k))**2 + (y(l) - y(k))**2
                    A(l, k) = r2*r2*sqrt(r2)
                end do
            end do
        case default
            m = (ws%q - 1)/2
            do k = 1, n
                l0 = merge(k, 1, lower)
                do l = l0, n
                    r2 = (x(l) - x(k))**2 + (y(l) - y(k))**2
                    A(l, k) = r2**m*sqrt(r2)
                end do
            end do
        end select

        ! The monomials at node k, rows n+1 to nt of column k: P^T(:, k)
        do k = 1, n
            t(1) = x(k)
            t(2) = y(k)
            A(n + 1, k) = 1
            do i = 2, np
                A(n + i, k) = A(n + ws%parent(i), k)*t(ws%axis(i))
            end do
        end do

        ! The upper-right block P, column by column by the same
        ! recurrence, each a coordinate times an earlier column
        if (.not. lower) then
            A(1:n, n + 1) = 1
            do i = 2, np
                j = n + ws%parent(i)
                if (ws%axis(i) == 1) then
                    A(1:n, n + i) = A(1:n, j)*x
                else
                    A(1:n, n + i) = A(1:n, j)*y
                end if
            end do
        end if

        A(n + 1:nt, n + 1:nt) = 0

    end subroutine

    ! The system of one stencil of n nodes at (x(k), y(k)), k = 1, n,
    ! in the stencil-local frame, into the workspace's matrix, and the
    ! nrhs right-hand sides, operator op(j) at (xc(j), yc(j)), into ws%B
    recursive subroutine rbf_fd_fill(ws, n, x, y, nrhs, op, xc, yc)
        type(rbf_fd_workspace), intent(inout), target :: ws
        integer, intent(in) :: n
        real(wp), intent(in) :: x(n), y(n)
        integer, intent(in) :: nrhs
        integer, intent(in) :: op(nrhs)
        real(wp), intent(in) :: xc(nrhs), yc(nrhs)

        integer :: j

        ws%a0 = aligned_start(ws)
        call fill_matrix(ws, n, x, y, ws%abuf(ws%a0), ws%lda)

        do j = 1, nrhs
            call phs_column(ws%q, n, x, y, op(j), xc(j), yc(j), ws%B(1, j))
            call poly_column(ws, op(j), xc(j), yc(j), ws%B(n + 1, j))
        end do

    end subroutine

    ! Solves the system rbf_fd_fill set up, overwriting the matrix with
    ! its factorization and ws%B with the solution: the weights in
    ! ws%B(1:n, 1:nrhs), the polynomial coefficients below them. info
    ! is LAPACK's: 0 on success, > 0 if the factorization met an exact
    ! zero pivot (coincident nodes, or too few of them for the
    ! polynomial degree); rounding may turn such a pivot into a tiny
    ! nonzero one instead, so a singular stencil is not guaranteed to
    ! be reported.
    recursive subroutine rbf_fd_solve(ws, n, nrhs, info)
        type(rbf_fd_workspace), intent(inout) :: ws
        integer, intent(in) :: n, nrhs
        integer, intent(out) :: info

        integer :: nt

        nt = n + ws%np
        associate (A => ws%abuf(ws%a0:), lda => ws%lda)
            select case (ws%solver)
            case (SOLVER_LU)
                call dgetf2(nt, nt, A, lda, ws%ipiv, info)
                if (info == 0) call dgetrs('N', nt, nrhs, A, lda, ws%ipiv, ws%B, lda, info)
            case (SOLVER_LDLT)
                call dsytrf('L', nt, A, lda, ws%ipiv, ws%work, size(ws%work), info)
                if (info == 0) call dsytrs('L', nt, nrhs, A, lda, ws%ipiv, ws%B, lda, info)
            case (SOLVER_RECLU_LU)
                call reclu_lu_opt(nt, nt, A, lda, ws%ipiv, info)
                if (info == 0) call reclu_solve_opt(nt, nrhs, A, lda, ws%ipiv, ws%B, lda)
            case (SOLVER_RECLU_LDLT)
                call reclu_sytf2_opt('L', nt, A, lda, ws%ipiv, info)
                if (info == 0) call reclu_sytrs_opt('L', nt, nrhs, A, lda, ws%ipiv, ws%B, lda)
            end select
        end associate

    end subroutine

    ! The weights of one stencil: rbf_fd_fill, then rbf_fd_solve. On
    ! return with info = 0, ws%B(k, j), k = 1, n, is the weight of node
    ! k for operator j. Preconditions, unchecked here: n <= ws%nmax,
    ! n >= ws%np, nrhs <= ws%nrhs_max.
    recursive subroutine rbf_fd_weights(ws, n, x, y, nrhs, op, xc, yc, info)
        type(rbf_fd_workspace), intent(inout) :: ws
        integer, intent(in) :: n
        real(wp), intent(in) :: x(n), y(n)
        integer, intent(in) :: nrhs
        integer, intent(in) :: op(nrhs)
        real(wp), intent(in) :: xc(nrhs), yc(nrhs)
        integer, intent(out) :: info

        call rbf_fd_fill(ws, n, x, y, nrhs, op, xc, yc)
        call rbf_fd_solve(ws, n, nrhs, info)

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

        if (nrhs < 1 .or. nrhs > ws%nrhs_max) then
            error stop "rbf_fd: nrhs must be between 1 and the workspace's nrhs_max"
        end if
        if (any(op < 0 .or. op > OP_LAST)) error stop "rbf_fd: unknown operator code"

        first_bad = nstencils + 1

        !$omp parallel default(shared) reduction(min:first_bad)
        block
            type(rbf_fd_workspace) :: wsl
            real(wp), allocatable :: x(:), y(:)
            integer :: s, n, ierr

            wsl = ws
            allocate(x(ws%nmax), y(ws%nmax))

            !$omp do schedule(static)
            do s = 1, nstencils
                call gather(s, ws%nmax, n, x, y)
                if (n > ws%nmax .or. n < ws%np) then
                    error stop "rbf_fd: gather returned a stencil outside [npoly(p), nmax]"
                end if
                call rbf_fd_weights(wsl, n, x, y, nrhs, op, xc, yc, ierr)
                if (ierr /= 0) then
                    first_bad = min(first_bad, s)
                else
                    call scatter(s, n, nrhs, wsl%B, wsl%lda)
                end if
            end do
            !$omp end do
        end block
        !$omp end parallel

        info = 0
        if (first_bad <= nstencils) info = first_bad

    end subroutine

end module
