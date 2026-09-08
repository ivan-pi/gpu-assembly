! Readers of the points file and the graph file (docs/file_formats.md),
! the Fortran side of rbf_io.h, for the Fortran tests and benchmarks.
! Both stop with a message naming the file on anything they cannot
! read. Indices are returned as the file has them, 0-based, and the row
! pointer ia(0:n) with them, the layout the rest of the Fortran and the
! C++ use.
module rbf_io
use rbf_precision, only: wp
implicit none
private

public :: read_points, read_graph_csr

contains

    ! The point count, then one coordinate pair per line
    subroutine read_points(fname, n, x, y)
        character(*), intent(in) :: fname
        integer, intent(out) :: n
        real(wp), allocatable, intent(out) :: x(:), y(:)

        integer :: iu, ios, i

        open (newunit=iu, file=fname, status="old", action="read", iostat=ios)
        if (ios /= 0) call fail(fname, "cannot open")
        read (iu, *, iostat=ios) n
        if (ios /= 0 .or. n < 0) call fail(fname, "bad point count")
        allocate(x(n), y(n))
        do i = 1, n
            read (iu, *, iostat=ios) x(i), y(i)
            if (ios /= 0) call fail(fname, "short file or bad coordinate")
        end do
        close (iu)

    end subroutine

    ! The stencils in compressed sparse row form, "n nnz" then one row
    ! per node of any length: ia(0:n) and ja(0:nnz-1), 0-based. With k
    ! given, every row must have exactly k entries.
    subroutine read_graph_csr(fname, n, ia, ja, k)
        character(*), intent(in) :: fname
        integer, intent(out) :: n
        integer, allocatable, intent(out) :: ia(:), ja(:)
        integer, intent(in), optional :: k

        integer, parameter :: maxline = 65536
        character(len=maxline) :: line
        integer :: iu, ios, i, nnz, count, pos

        open (newunit=iu, file=fname, status="old", action="read", iostat=ios)
        if (ios /= 0) call fail(fname, "cannot open")
        read (iu, *, iostat=ios) n, nnz
        if (ios /= 0 .or. n < 0 .or. nnz < 0) call fail(fname, "bad header")
        if (present(k)) then
            if (nnz /= n*k) call fail(fname, "nnz is not n times the row length asked for")
        end if

        allocate(ia(0:n), ja(0:max(nnz, 1) - 1))
        ia(0) = 0
        pos = 0
        do i = 1, n
            read (iu, '(a)', iostat=ios) line
            if (ios /= 0) call fail(fname, "fewer rows than the header says")
            count = tokens(line)
            if (count < 1) call fail(fname, "a row without entries")
            if (present(k)) then
                if (count /= k) call fail(fname, "a row of the wrong length")
            end if
            if (pos + count > nnz) call fail(fname, "more entries than the header says")
            read (line, *, iostat=ios) ja(pos:pos + count - 1)
            if (ios /= 0) call fail(fname, "bad index")
            pos = pos + count
            ia(i) = pos
        end do
        close (iu)
        if (pos /= nnz) call fail(fname, "fewer entries than the header says")
        if (any(ja(0:nnz - 1) < 0 .or. ja(0:nnz - 1) >= n)) call fail(fname, "index out of range")

    end subroutine

    ! Blank-separated tokens on a line
    pure function tokens(line) result(count)
        character(*), intent(in) :: line
        integer :: count
        integer :: i
        logical :: inside
        count = 0
        inside = .false.
        do i = 1, len_trim(line)
            if (line(i:i) == ' ' .or. line(i:i) == char(9)) then
                inside = .false.
            else if (.not. inside) then
                inside = .true.
                count = count + 1
            end if
        end do
    end function

    subroutine fail(fname, what)
        character(*), intent(in) :: fname, what
        print '(a,a,a,a)', "rbf_io: ", what, ": ", fname
        error stop
    end subroutine

end module
