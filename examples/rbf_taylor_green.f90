! Taylor-Green vortex on a lattice grid: nx by ny cells with the field
! sampled at the cell centres (i - 1/2, j - 1/2), i.e. in lattice units,
! so the box is nx by ny and the modes (mx, my) are the fundamental
! (1, 1) by default. Arrays are laid out as (y, x). The formulas come
! from the taylor_green() case of rbf_benchmarks; this module only adds
! the grid.
module rbf_taylor_green

   use rbf_precision, only: wp
   use rbf_benchmarks, only: shear_modes, taylor_green, periodic_box, pi

   implicit none
   private

   public :: taylor_green_t
   public :: pi

   type :: taylor_green_t
      integer :: nx, ny
      type(shear_modes) :: point   ! the point-based case on this grid
   contains
      procedure :: eval => taylor_green_eval
      procedure :: time_constant => taylor_green_time_constant
      procedure :: decay_time => taylor_green_decay_time
   end type

   interface taylor_green_t
      module procedure :: taylor_green_t_constructor
   end interface

contains

   function taylor_green_t_constructor(nx,ny,umax,nu,mx,my) result(this)
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: umax, nu
      integer, intent(in), optional :: mx, my
      type(taylor_green_t) :: this
      integer :: mx_, my_

      mx_ = 1; if (present(mx)) mx_ = mx
      my_ = 1; if (present(my)) my_ = my

      this%nx = nx
      this%ny = ny
      this%point = taylor_green(periodic_box(real(nx,wp), real(ny,wp)), mx_, my_, umax, nu)
   end function

   ! Time constant 1/(nu*(kx^2 + ky^2)) of the velocity decay
   function taylor_green_time_constant(self) result(tc)
      class(taylor_green_t), intent(in) :: self
      real(wp) :: tc
      tc = self%point%time_constant()
   end function

   ! Time at which the amplitude has dropped to the fraction frac of umax
   function taylor_green_decay_time(self,frac) result(t)
      class(taylor_green_t), intent(in) :: self
      real(wp), intent(in) :: frac
      real(wp) :: t
      t = self%point%decay_time(frac)
   end function

   subroutine taylor_green_eval(self,t,p,ux,uy,S)
      class(taylor_green_t), intent(in) :: self
      real(wp), intent(in) :: t
      real(wp), intent(out) :: p(:,:), ux(:,:), uy(:,:)
      real(wp), intent(out), optional :: S(self%ny,self%nx,3)

      integer :: x, y
      real(wp) :: xx(self%ny), yy(self%ny)

      yy = [(y - 0.5_wp, y = 1, self%ny)]

      do x = 1, self%nx
         xx = x - 0.5_wp
         call self%point%fields(xx,yy,p(:,x),ux(:,x),uy(:,x),t)
         if (present(S)) then
            call self%point%stress_tensor(xx,yy,S(:,x,1),S(:,x,2),S(:,x,3),t)
         end if
      end do

   end subroutine

end module
