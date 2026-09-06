! Space-filling-curve keys for node renumbering.
!
! Part of the mixed C++/Fortran build: the C++ side (rbf_reorder.h)
! calls the bind(c) entry points below and does the sorting itself,
! so this module only maps points to curve keys.
!
! Keys are int64: each subdivision level contributes 2 bits, so
! ndiv <= 31. The point (x,y) must lie inside bbox.

module rbf_ordering

  use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_double
  implicit none
  private

  public :: bbox2d
  public :: morton_keys, hilbert_keys
  public :: zidx, hidx

  integer, parameter :: wp = c_double
  integer, parameter :: i8 = c_int64_t

  ! Interoperable with rbf::BBox2<double> on the C++ side.
  type, bind(c) :: bbox2d
    real(c_double) :: xmin, ymin, xmax, ymax
  end type

contains

  ! Morton (Z-curve) keys for n points.
  subroutine morton_keys(n,x,y,bbox,ndiv,keys) bind(c,name="rbf_morton_keys")
    integer(c_int), intent(in) :: n
    real(wp), intent(in) :: x(n), y(n)
    type(bbox2d), intent(in) :: bbox
    integer(c_int), intent(in) :: ndiv
    integer(i8), intent(out) :: keys(n)

    real(wp) :: bb(4)
    integer :: i

    bb = [bbox%xmin, bbox%ymin, bbox%xmax, bbox%ymax]

    ! Fixed-depth iterations, so a static schedule balances fine
    !$omp parallel do schedule(static)
    do i = 1, n
      keys(i) = zidx([x(i),y(i)],bb,ndiv)
    end do

  end subroutine

  ! Hilbert-curve keys for n points; same interface as morton_keys.
  subroutine hilbert_keys(n,x,y,bbox,ndiv,keys) bind(c,name="rbf_hilbert_keys")
    integer(c_int), intent(in) :: n
    real(wp), intent(in) :: x(n), y(n)
    type(bbox2d), intent(in) :: bbox
    integer(c_int), intent(in) :: ndiv
    integer(i8), intent(out) :: keys(n)

    real(wp) :: bb(4)
    integer :: i

    bb = [bbox%xmin, bbox%ymin, bbox%xmax, bbox%ymax]

    !$omp parallel do schedule(static)
    do i = 1, n
      keys(i) = hidx([x(i),y(i)],bb,ndiv)
    end do

  end subroutine


  pure function zidx(vertex,bbox,ndiv) result(z)

    real(wp), intent(in) :: vertex(2)
      !! A two-dimensional Cartesian point.
    real(wp), intent(in) :: bbox(4)
      !! Bounding box of the domain, specified as the axis-aligned
      !! values (xmin,ymin,xmax,ymax) corresponding to the bottom-left,
      !! and top-right corners.
    integer(c_int), intent(in) :: ndiv
      !! Number of bounding box sub-divisions.

    integer(i8) :: z
      !! The z-order index.

    integer :: i, d
    real(wp) :: center(2), bb(4)

    bb = bbox
    z = 0
    do i = 1, ndiv

#if defined(__NVCOMPILER) && __NVCOMPILER_MAJOR < 23
! nvfortran releases before 23.x lack the F2008 bit-shifting intrinsics
      z = lshift(z,2)
#else
      z = shiftl(z,2)
#endif

      center = midpoint(bb)

      ! Set the quadrant using Morton order
      !
      !   2 | 3
      ! ----|----
      !   0 | 1
      !
      if (vertex(1) > center(1)) z = z + 1
      if (vertex(2) > center(2)) z = z + 2

      ! Update the bounding box to the appropriate quadrant
      do d = 1, 2
        if (vertex(d) > center(d)) then
          bb(d) = center(d)
        else
          bb(2+d) = center(d)
        end if
      end do

    end do

  end function


  pure function hidx(vertex,bbox,ndiv) result(h)

    real(wp), intent(in) :: vertex(2)
      !! A two-dimensional Cartesian point.
    real(wp), intent(in) :: bbox(4)
      !! Bounding box of the domain, specified as the axis-aligned
      !! values (xmin,ymin,xmax,ymax) corresponding to the bottom-left,
      !! and top-right corners.
    integer(c_int), intent(in) :: ndiv
      !! Number of bounding box sub-divisions.

    integer(i8) :: h
      !! The Hilbert-order index.

    ! A  00  0
    ! B  01  1
    ! C  10  2
    ! D  11  3

    integer, parameter :: table(0:3,0:3) = reshape([0,3,1,2, &
                                                    2,3,1,0, &
                                                    2,1,3,0, &
                                                    0,1,3,2], [4,4])

    !    rot
    ! q / 0  1  2  3
    ! --------------
    ! 0 | D  B  C  A
    ! 1 | B  A  C  D
    ! 2 | A  B  D  C
    ! 3 | A  C  B  D

    integer, parameter :: tr(0:3,0:3) = reshape([ &
        3,1,0,0, &
        1,0,1,2, &
        2,2,3,1, &
        0,3,2,3], [4,4])

    integer :: i, d, rot, q
    real(wp) :: center(2), bb(4)

    bb = bbox
    h = 0
    rot = 0

    do i = 1, ndiv

#if defined(__NVCOMPILER) && __NVCOMPILER_MAJOR < 23
      h = lshift(h,2)
#else
      h = shiftl(h,2)
#endif
      center = midpoint(bb)

      q = 0
      if (vertex(1) > center(1)) q = q + 1
      if (vertex(2) > center(2)) q = q + 2

      h = h + table(q,rot)

      rot = tr(q,rot)

      ! Update the bounding box to the appropriate quadrant
      do d = 1, 2
        if (vertex(d) > center(d)) then
          bb(d) = center(d)
        else
          bb(2+d) = center(d)
        end if
      end do

    end do

  end function


  ! Midpoint of a bounding box [xmin,ymin,xmax,ymax]
  pure function midpoint(bb) result(center)
    real(wp), intent(in) :: bb(4)
    real(wp) :: center(2)
    center(1) = bb(1) + 0.5_wp*(bb(3) - bb(1))
    center(2) = bb(2) + 0.5_wp*(bb(4) - bb(2))
  end function

end module
