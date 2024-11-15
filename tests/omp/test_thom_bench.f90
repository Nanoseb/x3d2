program test_thom

  use mpi
  use omp_lib
  use m_common, only: dp, pi
  use m_omp_common, only: SZ
  use m_tdsops, only: tdsops_t, tdsops_init
  use m_exec_thom, only: exec_thom_tds_compact
  
  implicit none
  
  integer :: n_glob, n, n_groups
  integer :: n_iters
  integer :: ndof

  real(dp), dimension(:, :, :), allocatable :: u, du
  real(dp) :: dx, dx_per

  integer :: i, j, k
  integer :: pow
  integer :: ierr, nrank

  type(tdsops_t) :: tdsops

  real(dp) :: tstart, tend
  real(dp) :: mem_use

  logical :: allpass

  allpass = .true.
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, nrank, ierr)

  mem_use = 24.0d0 ! Memory to use in GiB
  
  !! Performance test
  do pow=5, 13 
    n_glob = 2**pow
    ! n_groups = 64*192 / SZ
    n_groups = mem_use/(2.00d0*n_glob*SZ*8/1024/1024/1024)
    n_iters = 100

    if (nrank == 0) then
      print *, "----------------------------------"
      print *, "pow:", pow
      print *, "n_glob:", n_glob
      print *, "n_groups:", n_groups
      print *, "SZ:", SZ
      print *, "n_iters:", n_iters
      print *, "memuse per rank [GiB]", 2.00d0*n_glob*n_groups*SZ*8/1024/1024/1024
    end if

    !! Verification test
    ! n_glob = 1024
    ! n_groups = 64 * 64 / SZ
    ! n_iters = 1

    n = n_glob
    ndof = n_glob * n_groups * SZ

    allocate(u(SZ, n, n_groups), du(SZ, n, n_groups))

    dx_per = 2 * pi / n_glob
    dx = 2 * pi / (n_glob - 1)

    !! Periodic case
    if (nrank == 0) print *, "=== Testing periodic case ==="

    do k = 1, n_groups
      do j = 1, n
        do i = 1, SZ
          u(i, j, k) = sin((j - 1) * dx_per)
        end do
      end do
    end do

    tdsops = tdsops_init(n, dx_per, &
                         operation = "second-deriv", scheme = "compact6", &
                         bc_start = "periodic", bc_end = "periodic")

    tstart = omp_get_wtime()
    do i = 1, n_iters
      call exec_thom_tds_compact(du, u, tdsops)
    end do
    tend = omp_get_wtime()
    print *, nrank, "Total time Periodic", tend - tstart

    call checkperf(tend - tstart, n_iters, ndof, 3.0_dp)
    !call checkerr(u, du, 1.0e-8_dp)

    !! Dirichlet case
    if (nrank == 0) print *, "=== Testing Dirichlet case ==="
    
    do k = 1, n_groups
      do j = 1, n
        do i = 1, SZ
          u(i, j, k) = sin((j - 1) * dx)
        end do
      end do
    end do

    tdsops = tdsops_init(n, dx, &
                         operation = "second-deriv", scheme = "compact6", &
                         bc_start = "dirichlet", bc_end = "dirichlet")

    tstart = omp_get_wtime()
    do i = 1, n_iters
      call exec_thom_tds_compact(du, u, tdsops)
    end do
    tend = omp_get_wtime()
    print *, nrank, "Total time Dirichlet", tend - tstart

    call checkperf(tend - tstart, n_iters, ndof, 3.0_dp)
    !call checkerr(u, du, 1.0e-8_dp)

    deallocate(u)
    deallocate(du)
  end do
  call MPI_Finalize(ierr)

contains

  subroutine checkperf(trun, n_iters, ndof, consumed_bw)

    real(dp), intent(in) :: trun
    integer, intent(in) :: n_iters
    integer, intent(in) :: ndof
    real(dp), intent(in) :: consumed_bw

    real(dp) :: nbytes
    real(dp) :: achievedBW

    real(dp) :: memClockRt, memBusWidth, deviceBW

    if (dp == kind(0.0d0)) then
      nbytes = 8_dp
    else
      nbytes = 4_dp
    end if
    achievedBW = consumed_bw * n_iters * ndof * nbytes / trun

    print *, nrank, "Achieved BW: ", achievedBW / (2**30), " GiB/s"

    memClockRt = 3200
    memBusWidth = 64
    deviceBW = 2 * memBusWidth / nbytes * memClockRt * (10**6)

    !print *, "Available BW: ", deviceBW / 2**30, " GiB/s / NUMA zone on ARCHER2"
    !print *, "Utilised BW: ", 100 * (achievedBW / deviceBW), " %"
    
  end subroutine checkperf

  subroutine checkerr(u, du, tol)

    real(dp), dimension(:, :, :), intent(in) :: u, du
    real(dp), intent(in) :: tol
    
    real(dp) :: norm_du

    norm_du = sum((u + du)**2) / n_glob / n_groups / SZ
    norm_du = sqrt(norm_du)

    print *, minval(u + du), maxval(u+du)
    
    print *, "error norm", norm_du

    if (norm_du > tol) then
      print *, "Check second derivatives... FAILED"
      allpass = .false.
    else
      print *,  "Check second derivatives... PASSED"
    end if
    
  end subroutine checkerr
  
end program test_thom
