! [0, Lx) x [0, Ly) with periodic images: the box of the periodic
! benchmarks and of the minimum-image displacement in stencil assembly.
! The mode (nx, ny) has the wave number (2 pi nx/Lx, 2 pi ny/Ly); in
! lattice units Lx = nx cells.
module rbf_periodic_box
use rbf_precision, only: wp, pi
implicit none
private

public :: periodic_box

type :: periodic_box
    real(wp) :: Lx, Ly
contains
    procedure :: wavenumber => box_wavenumber
    procedure :: wrap => box_wrap
    procedure :: minimum_image => box_minimum_image
end type

interface periodic_box
    module procedure :: periodic_box_constructor
end interface

contains

    function periodic_box_constructor(Lx,Ly) result(box)
        real(wp), intent(in) :: Lx, Ly
        type(periodic_box) :: box
        if (Lx <= 0 .or. Ly <= 0) error stop "periodic_box: box sides must be positive"
        box%Lx = Lx
        box%Ly = Ly
    end function

    pure function box_wavenumber(box,nx,ny) result(k)
        class(periodic_box), intent(in) :: box
        integer, intent(in) :: nx, ny
        real(wp) :: k(2)
        k = [2*pi*nx/box%Lx, 2*pi*ny/box%Ly]
    end function

    ! The point mapped into the box
    pure function box_wrap(box,xy) result(w)
        class(periodic_box), intent(in) :: box
        real(wp), intent(in) :: xy(2)
        real(wp) :: w(2)
        w(1) = xy(1) - box%Lx*floor(xy(1)/box%Lx)
        w(2) = xy(2) - box%Ly*floor(xy(2)/box%Ly)
    end function

    ! The shortest of the displacement and its periodic images
    pure function box_minimum_image(box,d) result(m)
        class(periodic_box), intent(in) :: box
        real(wp), intent(in) :: d(2)
        real(wp) :: m(2)
        m(1) = d(1) - box%Lx*anint(d(1)/box%Lx)
        m(2) = d(2) - box%Ly*anint(d(2)/box%Ly)
    end function

end module
