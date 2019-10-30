#include <iostream>
#include <gaussian_filter.h>
#include <mpi.h>
#include <adios2.h>

using namespace GaussianFilter;
using namespace std;

void init_data(ViewMatrixType::HostMirror d, int d_size)
{
  d(10, 20, 30) = 5.3;
  d(0,0,0) = 10.5;
  d(0,40,40) = - 11.3;
}

void to_adios2(ViewMatrixType::HostMirror data, MPI_Comm *comm, std::string stream_name)
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
  
  Kokkos::initialize( argc, argv );
  {
    ViewMatrixType g("gaussian", l, l, l);
    ViewMatrixType data("data", L, L, L);
    ViewMatrixType result("result", L, L, L);
    ViewMatrixType::HostMirror h_result = Kokkos::create_mirror_view( result );
    ViewMatrixType::HostMirror h_data = Kokkos::create_mirror_view( data );
    

    Kokkos::Timer timer;
    double time;
    
    init_data(h_data, L);

    time = timer.seconds();
    std::cout<<"init_data " << time << std::endl;

    timer.reset();
    Kokkos::deep_copy(data, h_data);
    time = timer.seconds();
    std::cout<<"h_data -> data " << time << std::endl;
    
    timer.reset();
    generate_gaussian(sigma,  g);
    time = timer.seconds();
    std::cout<<"generate_gaussian  " << time << std::endl;

    ViewMatrixConstType gg = g;
    
    timer.reset();
    apply_kernel(data, result, gg, int(8), int(8), int(8));
    Kokkos::fence();
    time = timer.seconds();
    std::cout<<"apply_kernel  " << time << std::endl;
    

    timer.reset();
    Kokkos::deep_copy(h_result, result);
    time = timer.seconds();
    std::cout<<"result -> h_result  " << time << std::endl;

    timer.reset();    
    to_adios2(h_result, &comm, "data");
    time = timer.seconds();
    std::cout<<"adios2  " << time << std::endl;    
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
