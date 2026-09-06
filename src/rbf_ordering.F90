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

  public :: fill_morton_keys, fill_hilbert_keys
  public :: zidx, hidx

  integer, parameter :: wp = c_double
  integer, parameter :: i8 = c_int64_t

contains

  ! Morton (Z-curve) keys for n points.
  ! bbox = [xmin,ymin,xmax,ymax], ndiv = number of quadtree subdivisions.
  subroutine fill_morton_keys(n,x,y,bbox,ndiv,keys) bind(c,name="rbf_fill_morton_keys")
    integer(c_int), intent(in) :: n
    real(wp), intent(in) :: x(n), y(n)
    real(wp), intent(in) :: bbox(4)
    integer(c_int), intent(in) :: ndiv
    integer(i8), intent(out) :: keys(n)

    integer :: i

    do concurrent (i = 1:n)
      keys(i) = zidx([x(i),y(i)],bbox,ndiv)
    end do

  end subroutine

  ! Hilbert-curve keys for n points; same interface as fill_morton_keys.
  subroutine fill_hilbert_keys(n,x,y,bbox,ndiv,keys) bind(c,name="rbf_fill_hilbert_keys")
    integer(c_int), intent(in) :: n
    real(wp), intent(in) :: x(n), y(n)
    real(wp), intent(in) :: bbox(4)
    integer(c_int), intent(in) :: ndiv
    integer(i8), intent(out) :: keys(n)

    integer :: i

    do concurrent (i = 1:n)
      keys(i) = hidx([x(i),y(i)],bbox,ndiv)
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

#ifdef __NVCOMPILER
! nvfortran v22.7 lacks the F2008 bit-shifting intrinsics
      z = lshift(z,2)
#else
      z = shiftl(z,2)
#endif

      center = bbox_center(bb)

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

#ifdef __NVCOMPILER
      h = lshift(h,2)
#else
      h = shiftl(h,2)
#endif
      center = bbox_center(bb)

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


  pure function bbox_center(bbox) result(center)
    real(wp), intent(in) :: bbox(4)
    real(wp) :: center(2)
    center(1) = bbox(1) + 0.5_wp*(bbox(3) - bbox(1))
    center(2) = bbox(2) + 0.5_wp*(bbox(4) - bbox(2))
  end function

end module
