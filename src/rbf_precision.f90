module rbf_precision
use, intrinsic :: iso_c_binding, only: c_float, c_double
implicit none
private
public :: wp, pi
integer, parameter :: wp = c_double
real(wp), parameter :: pi = 4.0_wp*atan(1.0_wp)
end module
