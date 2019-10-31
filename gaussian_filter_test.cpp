#include <iostream>
#include <gaussian_filter.h>
#include <mpi.h>
#include <adios2.h>

using namespace GaussianFilter;
using namespace std;

void init_data(vector<double>& d, int d_size)
{
  int index;
  index = 30 + 20*d_size + 10*d_size*d_size;
  d[index] = 5.3;
  index = 0 + 0*d_size + 0*d_size*d_size;
  d[index] = 10.5;
  index = 40 + 40*d_size + 0*d_size*d_size;
  d[index] = - 11.3;
}

void to_adios2(Kokkos::View<double***, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged> > data, MPI_Comm *comm, std::string stream_name)
{
  adios2::ADIOS ad ("adios2.xml", *comm, adios2::DebugON);
  adios2::IO writer_io = ad.DeclareIO(stream_name);
  adios2::Engine writer =
    writer_io.Open(stream_name + ".bp",
		   adios2::Mode::Write, *comm);

  adios2::Variable<double> var;

  size_t L0 = data.extent(0);
  size_t L1 = data.extent(1);
  size_t L2 = data.extent(2);  

  var =
    writer_io.DefineVariable<double> ("data",
      { L0, L1, L2 },
      { 0, 0, 0 },
      { L0, L1, L2 } );

  writer.BeginStep ();
  writer.Put<double> (var, data.data());
  writer.EndStep();

  writer.Close();  
}

int main(int argc, char ** argv)
{
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int rank, comm_size, wrank;

  MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
  const unsigned int color = 2;
  MPI_Comm comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, wrank, &comm);


  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_size);
  
  int L = 256;
  int gw = 3;
  int l = 2*gw + 1;
  double sigma = 0.1;
  int tile0=8, tile1=8, tile2=8;


  Kokkos::initialize( argc, argv );
  {

  Kokkos::Timer timer;
  double time;

  ViewMatrixType g("gaussian", l, l, l);
  generate_gaussian(sigma,  g);
  ViewMatrixConstType gg = g;

  Kokkos::fence();
  time = timer.seconds();
  std::cout<<"generate_gaussian " << time << std::endl;
  timer.reset();

  vector<double> v_data(L*L*L, 0);
  init_data(v_data, L);  
  Kokkos::View<double***, Kokkos::HostSpace, 
	       Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_data(v_data.data(), L, L, L); 

  Kokkos::fence();
  time = timer.seconds();
  std::cout<<"init_data " << time << std::endl;
  timer.reset();


  auto d_data = Kokkos::create_mirror_view_and_copy(Kokkos::Cuda(), h_data);

  Kokkos::fence();
  time = timer.seconds();
  std::cout<<"h_data -> d_data " << time << std::endl;
  timer.reset();

  Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::Cuda> data("data", L, L, L);
  Kokkos::deep_copy(data, d_data);

  Kokkos::fence();
  time = timer.seconds();
  std::cout<<"d_data -> data " << time << std::endl;
  timer.reset();

  Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::Cuda> result("result", L, L, L);
  apply_kernel(data, result, gg, tile0, tile1, tile2);

  Kokkos::fence();
  time = timer.seconds();
  std::cout<<"apply_kernel " << time << std::endl;
  timer.reset();

  Kokkos::deep_copy(d_data, result);

  Kokkos::fence();
  time = timer.seconds();
  std::cout<<"result -> d_data " << time << std::endl;
  timer.reset();

  Kokkos::deep_copy(h_data, d_data);

  Kokkos::fence();
  time = timer.seconds();
  std::cout<<"d_data -> h_data " << time << std::endl;
  timer.reset();

  to_adios2(h_data, &comm, "data");

  Kokkos::fence();
  time = timer.seconds();
  std::cout<<"to_adios2 " << time << std::endl;
  timer.reset();
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
