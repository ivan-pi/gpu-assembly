module rbf_precision
use, intrinsic :: iso_c_binding, only: c_float, c_double
implicit none
private
public :: wp
integer, parameter :: wp = c_double
end module
