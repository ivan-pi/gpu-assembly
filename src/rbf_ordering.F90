! Space-filling-curve keys for node renumbering.
!
! Part of the mixed C++/Fortran build: the C++ side (rbf_reorder.h)
! calls the bind(c) entry points below and does the sorting itself,
! so this module only maps points to curve keys.
!
! Keys are int64: each subdivision level contributes 2 bits, so
! ndiv <= 31. The point (x,y) must lie inside bbox.

module rbf_ordering

  use, intrinsic :: iso_c_binding, only: c_int32_t, c_int64_t, c_double
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
    integer(c_int32_t), intent(in) :: n
    real(wp), intent(in) :: x(n), y(n)
    type(bbox2d), intent(in) :: bbox
    integer(c_int32_t), intent(in) :: ndiv
    integer(i8), intent(out) :: keys(n)

    integer :: i

    call check_ndiv(ndiv)

    ! Fixed-depth iterations, so a static schedule balances fine
    !$omp parallel do schedule(static)
    do i = 1, n
      keys(i) = zidx(x(i),y(i),bbox,ndiv)
    end do

  end subroutine

  ! Hilbert-curve keys for n points; same interface as morton_keys.
  subroutine hilbert_keys(n,x,y,bbox,ndiv,keys) bind(c,name="rbf_hilbert_keys")
    integer(c_int32_t), intent(in) :: n
    real(wp), intent(in) :: x(n), y(n)
    type(bbox2d), intent(in) :: bbox
    integer(c_int32_t), intent(in) :: ndiv
    integer(i8), intent(out) :: keys(n)

    integer :: i

    call check_ndiv(ndiv)

    !$omp parallel do schedule(static)
    do i = 1, n
      keys(i) = hidx(x(i),y(i),bbox,ndiv)
    end do

  end subroutine

  ! The key functions zidx/hidx are elemental (hence pure), and error
  ! stop in a pure procedure requires F2018, which not all compilers
  ! we target support. The check therefore lives here, in the non-pure
  ! wrappers, before the loop; anyone calling the elemental functions
  ! directly must validate ndiv at the call site.
  subroutine check_ndiv(ndiv)
    integer(c_int32_t), intent(in) :: ndiv
    if (ndiv < 1 .or. ndiv > 31) then
      error stop "rbf_ordering: ndiv must be between 1 and 31"
    end if
  end subroutine


  elemental function zidx(px,py,bbox,ndiv) result(z)

    real(wp), intent(in) :: px, py
      !! A two-dimensional Cartesian point, which must lie inside bbox.
    type(bbox2d), intent(in) :: bbox
      !! Axis-aligned bounding box of the domain.
    integer(c_int32_t), intent(in) :: ndiv
      !! Number of bounding box sub-divisions, 1 <= ndiv <= 31.
      !! Not validated here (an elemental function is pure and cannot
      !! portably error stop); validate at the call site, as the
      !! wrapper subroutines above do.

    integer(i8) :: z
      !! The z-order index.

    integer :: i
    real(wp) :: cx, cy, x0, y0, x1, y1

    x0 = bbox%xmin
    y0 = bbox%ymin
    x1 = bbox%xmax
    y1 = bbox%ymax

    z = 0
    do i = 1, ndiv

#if defined(__NVCOMPILER) && __NVCOMPILER_MAJOR < 23
! nvfortran releases before 23.x lack the F2008 bit-shifting intrinsics
      z = lshift(z,2)
#else
      z = shiftl(z,2)
#endif

      cx = x0 + 0.5_wp*(x1 - x0)
      cy = y0 + 0.5_wp*(y1 - y0)

      ! Set the quadrant using Morton order
      !
      !   2 | 3
      ! ----|----
      !   0 | 1
      !
      if (px > cx) z = z + 1
      if (py > cy) z = z + 2

      ! Shrink the bounding box to the chosen quadrant
      if (px > cx) then
        x0 = cx
      else
        x1 = cx
      end if
      if (py > cy) then
        y0 = cy
      else
        y1 = cy
      end if

    end do

  end function


  elemental function hidx(px,py,bbox,ndiv) result(h)

    real(wp), intent(in) :: px, py
      !! A two-dimensional Cartesian point, which must lie inside bbox.
    type(bbox2d), intent(in) :: bbox
      !! Axis-aligned bounding box of the domain.
    integer(c_int32_t), intent(in) :: ndiv
      !! Number of bounding box sub-divisions, 1 <= ndiv <= 31.
      !! Not validated here (an elemental function is pure and cannot
      !! portably error stop); validate at the call site, as the
      !! wrapper subroutines above do.

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

    integer :: i, rot, q
    real(wp) :: cx, cy, x0, y0, x1, y1

    x0 = bbox%xmin
    y0 = bbox%ymin
    x1 = bbox%xmax
    y1 = bbox%ymax

    h = 0
    rot = 0

    do i = 1, ndiv

#if defined(__NVCOMPILER) && __NVCOMPILER_MAJOR < 23
      h = lshift(h,2)
#else
      h = shiftl(h,2)
#endif
      cx = x0 + 0.5_wp*(x1 - x0)
      cy = y0 + 0.5_wp*(y1 - y0)

      q = 0
      if (px > cx) q = q + 1
      if (py > cy) q = q + 2

      h = h + table(q,rot)

      rot = tr(q,rot)

      ! Shrink the bounding box to the chosen quadrant
      if (px > cx) then
        x0 = cx
      else
        x1 = cx
      end if
      if (py > cy) then
        y0 = cy
      else
        y1 = cy
      end if

    end do

  end function

end module
