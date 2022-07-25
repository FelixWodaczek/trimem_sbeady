/** \file energy.cpp
 * \brief Helfrich Energy functional on a OpenMesh::TriMesh.
 */
#include "energy.h"

#include "mesh_tether.h"
#include "nlists/nlist.h"
#include "mesh_repulsion.h"
#include "kernel.h"

namespace trimem {

EnergyManager::EnergyManager(const TriMesh& mesh,
                             const EnergyParams& energy_params) :
  params(energy_params)
{
    // setup bond potential
    bonds = make_bonds(params.bond_params);

    // setup neighbour list
    nlist = make_nlist(mesh, params);

    // setup mesh repulsion
    repulse = make_repulsion(*nlist, params.repulse_params);

    // evaluate properties from mesh
    initial_props = properties(mesh);
}

VertexProperties EnergyManager::interpolate_reference_properties() const
{
    auto& cparams = params.continuation_params;

    real i_af = 1.0 - params.area_frac;
    real i_vf = 1.0 - params.volume_frac;
    real i_cf = 1.0 - params.curvature_frac;
    const real& lam = cparams.lambda;

    VertexProperties ref_props{0, 0, 0, 0, 0, 0};
    ref_props.area      = ( 1.0 - lam * i_af ) * initial_props.area;
    ref_props.volume    = ( 1.0 - lam * i_vf ) * initial_props.volume;
    ref_props.curvature = ( 1.0 - lam * i_cf ) * initial_props.curvature;

    return ref_props;
}

void EnergyManager::update_reference_properties()
{
    auto& cparams = params.continuation_params;

    if (cparams.lambda < 1.0)
    {
        cparams.lambda += cparams.delta;
    }
}

void EnergyManager::update_repulsion(const TriMesh& mesh)
{
    nlist   = make_nlist(mesh, params);
    repulse = make_repulsion(*nlist, params.repulse_params);
}

VertexProperties EnergyManager::properties(const TriMesh& mesh)
{
    const size_t n = mesh.n_vertices();

    VertexProperties props{ 0, 0, 0, 0, 0, 0};
    std::vector<VertexProperties> vprops(n, props);

    EvaluateProperties eval_kernel(params, mesh, *bonds, *repulse, vprops);
    parallel_for(n, eval_kernel);

    ReduceProperties reduce_kernel(vprops);
    parallel_reduction(n, reduce_kernel, props);

    return props;
}

real EnergyManager::energy(const TriMesh& mesh)
{
    auto ref_props = interpolate_reference_properties();
    auto props     = properties(mesh);

    return trimem_energy(params, props, ref_props);
}

real EnergyManager::energy(const VertexProperties& props)
{
    auto ref_props = interpolate_reference_properties();
    return trimem_energy(params, props, ref_props);
}

std::vector<Point> EnergyManager::gradient(const TriMesh& mesh)
{
    const size_t n = mesh.n_vertices();

    // update properties
    VertexProperties props{ 0, 0, 0, 0, 0, 0};
    std::vector<VertexProperties> vprops(n, props);

    EvaluateProperties eval_kernel(params, mesh, *bonds, *repulse, vprops);
    parallel_for(n, eval_kernel);

    ReduceProperties reduce_kernel(vprops);
    parallel_reduction(n, reduce_kernel, props);

    // reference properties
    auto ref_props = interpolate_reference_properties();

    // properties gradients
    VertexPropertiesGradient zeros
      { Point(0), Point(0), Point(0), Point(0), Point(0), Point(0) };
    std::vector<VertexPropertiesGradient> gprops(n, zeros);
    EvaluatePropertiesGradient pg_kernel(mesh, *bonds, *repulse, gprops);
    parallel_for(n, pg_kernel);

    // evaluate gradient
    std::vector<Point> gradient(n, Point(0));
    EvaluateGradient g_kernel(params, props, ref_props, gprops, gradient);
    parallel_for(n, g_kernel);

    return gradient;
}

void EnergyManager::print_info(const TriMesh& mesh)
{
  auto props     = properties(mesh);
  auto ref_props = interpolate_reference_properties();

  std::ostream& out = std::cout;

  out << "----- EnergyManager info\n";
  out << "reference properties:\n";
  out << "  area:      " << ref_props.area << "\n";
  out << "  volume:    " << ref_props.volume << "\n";
  out << "  curvature: " << ref_props.curvature << "\n";
  out << "current properties:\n";
  out << "  area:      " << props.area << "\n";
  out << "  volume:    " << props.volume << "\n";
  out << "  curvature: " << props.curvature << "\n";
  out << "energies:\n";
  out << "  area:      " << area_penalty(params, props, ref_props) << "\n";
  out << "  volume:    " << volume_penalty(params, props, ref_props) << "\n";
  out << "  area diff: " << curvature_penalty(params, props, ref_props) << "\n";
  out << "  bending:   " << helfrich_energy(params, props) << "\n";
  out << "  tether:    " << tether_penalty(params, props) << "\n";
  out << "  repulsion: " << repulsion_penalty(params, props) << "\n";
  out << "  total:     " << trimem_energy(params, props, ref_props) << "\n";
  out << std::endl;
}


}
