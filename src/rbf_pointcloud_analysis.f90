module rbf_pointcloud_analysis

implicit none
private

public :: cache_sim

   interface
      ! Count number of unique integers; array a is rearranged
      function count_unique(a,n) bind(c)
         ! See count_unique.cpp for implementation details
         use, intrinsic :: iso_c_binding, only: c_int
         integer(c_int), value :: n
         integer(c_int), intent(inout) :: a(n)
         integer(c_int) :: count_unique
      end function
   end interface

contains

    ! Calculate some point-cloud cache-related statistics
    !
    ! The stencil distance is a measure how local the stencil is
    ! in memory. Large values mean that the points are far away in memory
    !
    ! The array clines is the number of different cache-lines participating
    ! in the stencil, assuming that the cache will load from the lowest
    ! addressable memory chunk containing a given element.
    subroutine cache_sim(n,ia,ja,dist,clines,cdist)

        integer, intent(in) :: n
        integer, intent(in) :: ia(0:n), ja(0:*)
        integer, intent(out) :: dist(0:n-1), clines(0:n-1),cdist(0:n-1)

        integer :: i, iaa, iab

        integer, allocatable :: tmp(:)

        do i = 0, n-1

            iaa = ia(i)
            iab = ia(i+1) - 1

            ! How far is furthest element
            dist(i) = maxval(abs(ja(iaa:iab) - i))

            ! How many unique cache lines (i.e. blocks of 8 elements)
            ! are used by the stencil
            associate(curr => i / 8)
               tmp = ja(iaa:iab) / 8
               clines(i) = count_unique(tmp,size(tmp))
               cdist(i) = maxval(abs(tmp - curr))
            end associate

        end do

    end subroutine

end module