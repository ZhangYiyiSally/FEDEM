// Gmsh project created on Mon Dec 09 16:12:13 2024
SetFactory("OpenCASCADE");
//+
Box(1) = {0, 0, 0, 4, 1, 1};
//+
Physical Volume("all_volume", 13) = {1};
//+
Physical Surface("bc_Dirichlet", 14) = {1};
//+
Physical Surface("bc_Neumann", 15) = {2};
//+
Physical Surface("all_boundaries", 16) = {1, 3, 2, 4, 6, 5};
